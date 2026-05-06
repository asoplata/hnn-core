"""Post hoc extracellular decomposition from HNN transmembrane-current recordings.

This module is designed for exploratory work on top of the transmembrane-current
recording branch discussed in PR #1212 (and ####). It reconstructs approximate LFP/CSD
contributions from:

1. intrinsic / transmembrane current channels recorded per segment
2. synaptic currents recorded at section midpoints, using a midpoint-segment
   approximation

Assumptions
-----------
- This code assumes the PR branch stores recordings under
  ``net.cell_response.transmembrane_currents`` and ``net.cell_response.isec``.
- Intrinsic currents are assumed to be recorded as current densities
  (mA / cm^2) except for ``agg_i_mem``, which is assumed to already be in nA.
- Synaptic currents are approximated as being located on the midpoint segment of
  the section. For even ``nseg``, the current is split equally over the two
  central segments by default.

This is a standalone analysis helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt

from hnn_core.cells_default import pyramidal
from hnn_core.extracellular import _transfer_resistance, calculate_csd2d
from hnn_core.network_builder import load_custom_mechanisms

from numpy.linalg import norm

# Channels exposed by the tm_currents branch. These are the currents in Jones 2009.
DENSITY_CHANNELS = {
    "agg_ina",
    "agg_ik",
    "agg_i_cap",
    "ina_hh2",
    "ik_hh2",
    "ik_kca",
    "ik_km",
    "ica_ca",
    "ica_cat",
    "il_hh2",
    "i_ar",
}
ABSOLUTE_NA_CHANNELS = {"agg_i_mem"}
SUPPORTED_INTRINSIC_CHANNELS = DENSITY_CHANNELS | ABSOLUTE_NA_CHANNELS


@dataclass(frozen=True)
class SourceInfo:
    """Metadata container for each extracellular current source to use in the reconstruction.

    Each source corresponds to one segment for intrinsic currents,
    and one segment (the midpoint segment) for synaptic currents.

    The ``area_um2`` and ``weight_fraction`` fields are used to convert current
    densities to absolute currents and to implement the midpoint-segment
    approximation for synaptic currents, respectively.
    """
    kind: str  # "intrinsic" or "synaptic"
    cell_type: str
    gid: int
    section: str
    segment_index: int
    segment_x: float
    label: str
    area_um2: float | None = None
    syn_name: str | None = None
    weight_fraction: float = 1.0


def _segment_xs_for_section(nseg: int) -> np.ndarray:
    """Return normalized NEURON x locations for segment centers.

    For a section with ``nseg`` segments, NEURON places segment centers evenly
    between 0 and 1, excluding the endpoints. For example, ``nseg=5`` gives
    x locations ``[0.1, 0.3, 0.5, 0.7, 0.9]``.

    These x locations are used for segment area lookup and for assigning
    recorded currents to segment-level extracellular sources.
    """
    return np.asarray([(i - 0.5) / nseg for i in range(1, nseg + 1)], dtype=float)


def _segment_area_um2(template_cell, section: str, segment_x: float) -> float:
    """Returns the membrane area of one segment, given section and segment location"""
    sec = template_cell.sections[section]
    return float(np.pi * sec.diam * (sec.L / sec.nseg))


def _density_to_nA(current_density, area_um2: float) -> np.ndarray:
    """Convert current density mA/cm^2 to nA using segment area in um^2."""
    return np.asarray(current_density, dtype=float) * area_um2 * 1e-2


def _ensure_2d_timeseries(matrix_like: list[np.ndarray]) -> np.ndarray:
    """Convert list of 1D arrays to 2D array with shape (n_sources, n_times).
    
    Each input array is assumed to have shape (n_times,). The output has shape (n_sources, n_times).

    If the input list is empty, returns an empty array with shape (0, 0).
    """
    if len(matrix_like) == 0:
        return np.empty((0, 0), dtype=float)
    return np.vstack([np.asarray(x, dtype=float)[None, :] for x in matrix_like])


def _pick_midpoint_segments(nseg: int, mode: str = "split_even"):
    """Choose which segment or segments should represent the midpoint of a section 
    for synaptic current assignment, and how to weight them if there are multiple. 
    For even nseg, the default is to split evenly between the two central segments, 
    but other options are provided.
    
    Return [(segment_index, weight_fraction), ...] for midpoint approximation.
    """
    if nseg < 1:
        raise ValueError("nseg must be >= 1")

    if nseg % 2 == 1:
        center = nseg // 2
        return [(center, 1.0)]

    left = (nseg // 2) - 1
    right = nseg // 2
    if mode == "nearest_lower":
        return [(left, 1.0)]
    if mode == "nearest_upper":
        return [(right, 1.0)]
    if mode == "split_even":
        return [(left, 0.5), (right, 0.5)]
    raise ValueError(
        "midpoint_mode must be one of {'split_even', 'nearest_lower', 'nearest_upper'}"
    )


def _channel_data_for_gid(net, trial_idx: int, gid: int, channel: str):
    return net.cell_response.transmembrane_currents[channel][trial_idx][gid]


def _synaptic_data_for_gid(net, trial_idx: int, gid: int):
    return net.cell_response.isec[trial_idx][gid]


def collect_intrinsic_sources(
    net,
    trial_idx: int = 0,
    cell_types: Sequence[str] = ("L2_pyramidal", "L5_pyramidal"),
    channels: Sequence[str] | None = None,
    cap_current_sign: float = -1.0,
):
    """Collect intrinsic current sources as post hoc source vectors.

    More specifically, create a list of SourceInfo objects,
    one for each intrinsic current source that will be used in the reconstruction.

    Returns
    -------
    sources : list[SourceInfo]
    current_matrix_nA : ndarray, shape (n_sources, n_times)
    """
    channels = tuple(channels or sorted(SUPPORTED_INTRINSIC_CHANNELS))
    bad = sorted(set(channels) - SUPPORTED_INTRINSIC_CHANNELS)
    if bad:
        raise ValueError(f"Unsupported intrinsic channels: {bad}")

    sources: list[SourceInfo] = []
    currents_nA = []

    for cell_type in cell_types:
        template_cell = net.cell_types[cell_type]["cell_object"]
        
        for gid in net.gid_ranges[cell_type]:
            for channel in channels:
                channel_data = _channel_data_for_gid(net, trial_idx, gid, channel)
                for section, seg_dict in channel_data.items():
                    nseg = len(seg_dict)
                    seg_xs = _segment_xs_for_section(nseg)
                    seg_keys = list(seg_dict.keys())

                    for seg_idx, (seg_key, seg_x) in enumerate(zip(seg_keys, seg_xs)):
                        values = np.asarray(seg_dict[seg_key], dtype=float)
                        area_um2 = _segment_area_um2(template_cell, section, seg_x)

                        if channel in ABSOLUTE_NA_CHANNELS:
                            values_nA = values.copy()
                        else:
                            values_nA = _density_to_nA(values, area_um2)
                            if channel == "agg_i_cap":
                                values_nA = cap_current_sign * values_nA

                        sources.append(
                            SourceInfo(
                                kind="intrinsic",
                                cell_type=cell_type,
                                gid=int(gid),
                                section=section,
                                segment_index=int(seg_idx),
                                segment_x=float(seg_x),
                                label=channel,
                                area_um2=float(area_um2),
                            )
                        )
                        currents_nA.append(values_nA)

    return sources, _ensure_2d_timeseries(currents_nA)


def collect_synaptic_sources(
    net,
    trial_idx: int = 0,
    cell_types: Sequence[str] = ("L2_pyramidal", "L5_pyramidal"),
    #template_builders: dict[str, Callable] | None = None,
    midpoint_mode: str = "split_even",
):
    """Collect synaptic current sources using midpoint-segment approximation.

    Each recorded synapse current is assigned to the segment corresponding to the
    section midpoint. For even nseg, it is split between the two central segments
    unless a different midpoint_mode is requested.
    """
    sources: list[SourceInfo] = []
    currents_nA = []

    for cell_type in cell_types:
        template_cell = net.cell_types[cell_type]["cell_object"]
        for gid in net.gid_ranges[cell_type]:
            syn_data = _synaptic_data_for_gid(net, trial_idx, gid)
            for section, syn_dict in syn_data.items():
                #nseg = template_cell._nrn_sections[section].nseg
                nseg = template_cell.sections[section].nseg
                seg_xs = _segment_xs_for_section(nseg)
                midpoint_targets = _pick_midpoint_segments(nseg, mode=midpoint_mode)

                for syn_name, vec in syn_dict.items():
                    syn_current_nA = np.asarray(vec, dtype=float)
                    for seg_idx, weight in midpoint_targets:
                        seg_x = seg_xs[seg_idx]
                        area_um2 = _segment_area_um2(template_cell, section, seg_x)
                        sources.append(
                            SourceInfo(
                                kind="synaptic",
                                cell_type=cell_type,
                                gid=int(gid),
                                section=section,
                                segment_index=int(seg_idx),
                                segment_x=float(seg_x),
                                label="isec",
                                syn_name=str(syn_name),
                                area_um2=float(area_um2),
                                weight_fraction=float(weight),
                            )
                        )
                        currents_nA.append(weight * syn_current_nA)

    return sources, _ensure_2d_timeseries(currents_nA)


def _transfer_resistance_postproc(
    sec_start,
    sec_end,
    nseg,
    L,
    electrode_pos,
    conductivity,
    method,
    min_distance=0.5,
):
    """Postprocessing equivalent of _transfer_resistance in extracellular.py.

    Parameters
    ----------
    sec_start : array-like, shape (3,)
        Global 3D coordinates of section start (um).
    sec_end : array-like, shape (3,)
        Global 3D coordinates of section end (um).
    nseg : int
        Number of segments in the section.
    L : float
        Section length (um).
    electrode_pos : array-like, shape (3,)
        Electrode position in global coordinates (um).
    conductivity : float
        Extracellular conductivity (S/m).
    method : str
        'psa' or 'lsa'.
    min_distance : float
        Minimum distance in um.

    Returns
    -------
    xfer : ndarray, shape (nseg,)
        Transfer resistance for each segment in the section.
    """
    electrode_pos = np.asarray(electrode_pos, dtype=float)
    sec_start = np.asarray(sec_start, dtype=float)
    sec_end = np.asarray(sec_end, dtype=float)

    sec_vec = sec_end - sec_start
    sec_norm = norm(sec_vec)
    if sec_norm == 0:
        raise ValueError("Section start and end points are identical.")

    seg_ctr = np.zeros((nseg, 3), dtype=float)
    line_lens = np.zeros(nseg + 2, dtype=float)

    # Match NEURON segment centers: nseg=5 -> 0.1, 0.3, 0.5, 0.7, 0.9
    seg_xs = [(2 * ii + 1) / (2 * nseg) for ii in range(nseg)]

    for ii, seg_x in enumerate(seg_xs):
        seg_ctr[ii, :] = sec_start + seg_x * sec_vec
        line_lens[ii + 1] = seg_x * L

    line_lens[-1] = L
    line_lens = np.diff(line_lens)
    first_len = line_lens[0]
    line_lens = np.array([first_len] + list(line_lens[2:]))

    if method == "psa": # point source approximation. Each segment is treated as a point source at its center.
        dis = norm(np.tile(electrode_pos, (nseg, 1)) - seg_ctr, axis=1)
        dis = np.maximum(dis, min_distance)
        phi = 1.0 / dis

    elif method == "lsa": # line-source approximation. Each segment is treated as a line source along the section.
        phi = np.zeros(nseg, dtype=float)
        sec_unit = sec_vec / sec_norm

        for idx, (ctr, line_len) in enumerate(zip(seg_ctr, line_lens)):
            start = ctr - line_len * sec_unit
            end = ctr + line_len * sec_unit

            a = end - start
            norm_a = norm(a)
            b = electrode_pos - end
            H = np.dot(b, a) / norm_a
            Lpar = H + norm_a
            R2 = np.dot(b, b) - H**2
            R2 = np.maximum(R2, min_distance**2)

            if Lpar < 0 and H < 0:
                num = np.sqrt(H**2 + R2) - H
                denom = np.sqrt(Lpar**2 + R2) - Lpar
            elif Lpar > 0 and H < 0:
                num = (np.sqrt(H**2 + R2) - H) * (Lpar + np.sqrt(Lpar**2 + R2))
                denom = R2
            else:
                num = np.sqrt(Lpar**2 + R2) + Lpar
                denom = np.sqrt(H**2 + R2) + H

            phi[idx] = np.log(num / denom) / norm_a
    else:
        raise ValueError(f"Unknown method: {method}")

    return 1000.0 * phi / (4.0 * np.pi * conductivity)


def _get_gid_soma_pos(net, cell_type, gid):
    """Return gid-specific soma position."""
    start_gid = net.gid_ranges[cell_type][0]
    return np.asarray(net.pos_dict[cell_type][gid - start_gid], dtype=float)



def _get_global_section_geometry(net, template_cells, src):
    """Return global section endpoints and geometry for one source."""
    soma_pos = _get_gid_soma_pos(net, src.cell_type, src.gid)
    sec = template_cells[src.cell_type].sections[src.section]

    end_pts = np.asarray(sec._end_pts, dtype=float)
    sec_start = end_pts[0] + soma_pos
    sec_end = end_pts[1] + soma_pos

    return sec_start, sec_end, sec.nseg, sec.L


# note: HNN has pairwise transfer resistance calculation via _transfer_resistance() in extracellular.py.
# That function computes the transfer resistance between one NEURON segment 
# and one electrode position. That value is multiplied by the transmembrane current
# to get the extracellular potential.
# The following function does the same, just not on the fly but as a post hoc calculation, 
# and it caches transfer resistances for sections that are reused across multiple sources.
def build_transfer_resistance_matrix_for_sources(
    net,
    sources: Sequence[SourceInfo],
    array_name: str = "probe1",
):
    """Build electrode x source transfer matrix for postprocessed sources.
    
    Output T (n_contacts, n_sources): one column per source. Each source corresponds to one segment for intrinsic currents, 
    and one (or two) segments for synaptic currents depending on the midpoint_mode. 

    To avoid recomputing the same geometry repeatedly, transfer values are cached by (cell_type, gid, section)
    as arrays with shape (n_contacts, nseg). For each source, we then select the
    transfer-resistance column corresponding to src.segment_index.
    """
    array = net.rec_arrays[array_name]
    cell_types = sorted({src.cell_type for src in sources})

    template_cells = {
        cell_type: net.cell_types[cell_type]["cell_object"]
        for cell_type in cell_types
    }

    T = np.zeros((len(array.positions), len(sources)), dtype=float)

    # Reuse section-level transfer vectors for repeated sources on the same
    # gid/section. Cached array has shape (n_contacts, nseg).
    cache = {}

    for col, src in enumerate(sources):
        key = (src.cell_type, src.gid, src.section)

        if key not in cache:
            sec_start, sec_end, nseg, L = _get_global_section_geometry(
                net, template_cells, src
            )

            xfers = []
            for pos in array.positions:
                xfer = _transfer_resistance_postproc(
                    sec_start=sec_start,
                    sec_end=sec_end,
                    nseg=nseg,
                    L=L,
                    electrode_pos=pos,
                    conductivity=array.conductivity,
                    method=array.method,
                    min_distance=array.min_distance,
                )
                xfers.append(np.asarray(xfer, dtype=float))

            cache[key] = np.vstack(xfers)  # shape: (n_contacts, nseg)

        T[:, col] = cache[key][:, src.segment_index]

    return T


def reconstruct_lfp_from_sources(
    transfer_matrix: np.ndarray,
    current_matrix_nA: np.ndarray,
) -> np.ndarray:
    """Return LFP (uV), shape (n_contacts, n_times).
    
    It reconstructs the LFP time series by multiplying the transfer resistance matrix (electrodes x sources)
    with the current matrix (sources x time) to get the LFP at each electrode over time. The result is in microvolts (uV).

    Check _nrn_r_transfer * _nrn_imem_vec in hnn_core.extracellular._transfer_resistance for HNN's version"""
    if transfer_matrix.shape[1] != current_matrix_nA.shape[0]:
        raise ValueError(
            "transfer_matrix columns must match current_matrix_nA rows: "
            f"got {transfer_matrix.shape[1]} vs {current_matrix_nA.shape[0]}"
        )
    return transfer_matrix @ current_matrix_nA


def reconstruct_intrinsic_lfp(
    net,
    trial_idx: int = 0,
    cell_types: Sequence[str] = ("L2_pyramidal", "L5_pyramidal"),
    channels: Sequence[str] | None = None,
    array_name: str = "probe1",
    cap_current_sign: float = -1.0,
):
    """ High-level wrapper to go from net and source selection parameters all the way to reconstructed LFP. 
    # Returns the LFP, the list of sources, the transfer matrix, and the current matrix for further analysis.
    """
    sources, I_nA = collect_intrinsic_sources(
        net,
        trial_idx=trial_idx,
        cell_types=cell_types,
        channels=channels,
        cap_current_sign=cap_current_sign,
    )
    T = build_transfer_resistance_matrix_for_sources(net, sources, array_name=array_name) #(n_contacts x n_sources)
    lfp = reconstruct_lfp_from_sources(T, I_nA) # (n_contacts x n_times); lfp = T @ I_nA
    return lfp, sources, T, I_nA


def reconstruct_synaptic_lfp(
    net,
    trial_idx: int = 0,
    cell_types: Sequence[str] = ("L2_pyramidal", "L5_pyramidal"),
    array_name: str = "probe1",
    midpoint_mode: str = "split_even",
):
    sources, I_nA = collect_synaptic_sources(
        net,
        trial_idx=trial_idx,
        cell_types=cell_types,
        midpoint_mode=midpoint_mode,
    )
    T = build_transfer_resistance_matrix_for_sources(net, sources, array_name=array_name)
    lfp = reconstruct_lfp_from_sources(T, I_nA)
    return lfp, sources, T, I_nA

def reconstruct_synaptic_lfp_by_name(
    net,
    trial_idx=0,
    cell_types=("L2_pyramidal", "L5_pyramidal"),
    syn_names=None,
    array_name="probe1",
    midpoint_mode="split_even",
):
    sources, I_syn = collect_synaptic_sources(
        net, trial_idx=trial_idx, cell_types=cell_types,
        midpoint_mode=midpoint_mode,
    )
    if syn_names is not None:
        sources, I_syn = filter_sources(sources, I_syn, syn_names=syn_names)
    if len(sources) == 0:
        raise ValueError(f"No synaptic sources matched syn_names={syn_names!r}.")
    T_syn = build_transfer_resistance_matrix_for_sources(
        net, sources, array_name=array_name)
    
    return reconstruct_lfp_from_sources(T_syn, I_syn), sources, T_syn, I_syn

    
def reconstruct_csd(net, lfp: np.ndarray, array_name: str = "probe1") -> np.ndarray:
    z_coords = np.asarray(net.rec_arrays[array_name].positions)[:, 2]
    if len(z_coords) < 2:
        raise ValueError("Need at least 2 contacts to compute laminar CSD.")
    delta = float(np.abs(z_coords[1] - z_coords[0]))
    return calculate_csd2d(lfp, delta=delta)


# e.g., ica_sources, ica_I = filter_sources(
#     sources, I_nA, labels=["ica_cat"])
# to then reconstruct the LFP contribution from just the ica_cat channel.
def filter_sources(
    sources: Sequence[SourceInfo],
    current_matrix_nA: np.ndarray,
    *,
    cell_types: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    sections: Sequence[str] | None = None,
    gid_subset: Iterable[int] | None = None,
    syn_names: Sequence[str] | None = None,
):
    """Filter an existing source/current collection without rerunning anything."""
    keep = np.ones(len(sources), dtype=bool)
    if cell_types is not None:
        cell_types = set(cell_types)
        keep &= np.array([src.cell_type in cell_types for src in sources])
    if labels is not None:
        labels = set(labels)
        keep &= np.array([src.label in labels for src in sources])
    if sections is not None:
        sections = set(sections)
        keep &= np.array([src.section in sections for src in sources])
    if gid_subset is not None:
        gid_subset = set(int(g) for g in gid_subset)
        keep &= np.array([src.gid in gid_subset for src in sources])
    #if syn_names is not None:
    #    syn_names = set(syn_names)
    #    #keep &= np.array([src.syn_name in syn_names for src in sources])
    #    keep = np.array([src.syn_name.endswith("_ampa") for src in sources])
    #if syn_names is not None:
    #    keep &= np.array([
    #        any(src.syn_name.endswith(f"_{syn}") for syn in syn_names)
    #        for src in sources
    #    ])
    if syn_names is not None:
        syn_set = set(syn_names)
        keep &= np.array([
            src.syn_name is not None
            and (src.syn_name in syn_set
                or any(src.syn_name.endswith(f"_{s}") for s in syn_set))
            for src in sources
        ])

    filt_sources = [src for src, k in zip(sources, keep) if k]
    filt_I = current_matrix_nA[keep]
    return filt_sources, filt_I


def summarize_sources(sources: Sequence[SourceInfo]):
    """Make a count summary of the sources, grouped by:
    - src.kind (intrinsic vs synaptic)
    - src.cell_type (e.g., L2_pyramidal)
    - src.label (e.g., "ina_hh2" or "isec")
    question: how many sources do I have of each type, and how does that relate to the LFP/CSD contributions?
    """
    summary = {}
    for src in sources:
        key = (src.kind, src.cell_type, src.label)
        summary[key] = summary.get(key, 0) + 1
    return summary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from hnn_core.viz import plot_laminar_lfp, plot_laminar_csd, plt_show


def plot_stacked_traces(
    ax,
    t,
    data,
    depths,
    color="k",
    lw=0.8,
    alpha=1.0,
    scale=1.0,
    baseline_samps=None,
    labels=None,
):
    """Plot traces stacked by cortical depth.

    Each trace is optionally baseline-corrected and then drawn around its
    corresponding depth value, so the y-coordinate of each plotted point is
    ``depth_z - scale * trace`` (the negative sign keeps positive deflections
    pointing toward the cortical surface when ``invert_yaxis=True``).

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    t : array-like, shape (n_times,)
        Time vector.
    data : array-like, shape (n_traces, n_times)
        One row per trace.
    depths : array-like, shape (n_traces,)
        Depth value for each trace, in the same units as the y-axis.
    color : matplotlib color
        Color used for all traces drawn in this call.
    lw, alpha : float
        Line width and alpha forwarded to ``ax.plot``.
    scale : float
        Multiplier applied to each trace before subtracting it from its depth.
    baseline_samps : int | None
        If not None, subtract the mean of the first ``baseline_samps`` samples
        from each trace before stacking. If None, no baseline correction.
    labels : list[str] | None
        Optional per-trace labels for the legend.
    """
    depths = np.asarray(depths, dtype=float)
    data = np.asarray(data)

    for i, (z, tr) in enumerate(zip(depths, data)):
        if baseline_samps is not None:
            tr0 = tr - np.mean(tr[:baseline_samps])
        else:
            tr0 = tr
        lab = labels[i] if labels is not None else None
        ax.plot(t, z - scale * tr0, color=color, lw=lw, alpha=alpha, label=lab)


def _draw_cell_morphology(ax, net, cell_type='L5_pyramidal', gid=None,
                          color_by_region=True, center_x=True,
                          diam_scale=5.0, x_shift=0.0):
    """Draw one cell's sections as rectangles in the (x, z) plane.

    Parameters
    ----------
    x_shift : float
        Extra horizontal offset (in µm) added to every section. Useful when
        drawing several cells side by side on the same axis.
    """
    template = net.cell_types[cell_type]['cell_object']
    if gid is None:
        gid = list(net.gid_ranges[cell_type])[0]
    start_gid = net.gid_ranges[cell_type][0]
    soma_pos = np.asarray(net.pos_dict[cell_type][gid - start_gid], dtype=float)

    region_colors = {
        'soma':           'black',
        'apical_trunk':   '#1f4e79',
        'apical_oblique': '#5b9bd5',
        'apical_1':       '#1f4e79',
        'apical_2':       '#1f4e79',
        'apical_tuft':    '#0b2540',
        'basal_1':        '#a85432',
        'basal_2':        '#d08770',
        'basal_3':        '#d08770',
    }
    x_off = soma_pos[0] if center_x else 0.0

    for name, sec in template.sections.items():
        pts = np.asarray(sec._end_pts, dtype=float) + soma_pos
        x0, z0 = pts[0, 0] - x_off + x_shift, pts[0, 2]
        x1, z1 = pts[1, 0] - x_off + x_shift, pts[1, 2]
        diam = float(sec.diam) * diam_scale  # exaggerate for visibility

        dx, dz = x1 - x0, z1 - z0
        L = np.hypot(dx, dz)
        if L < 1e-6:
            # zero-length projection (rare): draw a small axis-aligned square
            h = diam / 2.0
            corners = [(x0 - h, z0 - h), (x0 + h, z0 - h),
                       (x0 + h, z0 + h), (x0 - h, z0 + h)]
        else:
            # unit vector along section, then perpendicular (90° in-plane)
            ux, uz = dx / L, dz / L
            px, pz = -uz, ux
            hx, hz = px * diam / 2.0, pz * diam / 2.0
            # Square (butt) ends — no rounded caps
            corners = [(x0 + hx, z0 + hz),
                       (x1 + hx, z1 + hz),
                       (x1 - hx, z1 - hz),
                       (x0 - hx, z0 - hz)]

        face = region_colors.get(name, 'gray') if color_by_region else 'lightgray'
        ax.add_patch(Polygon(
            corners, closed=True,
            facecolor=face, edgecolor='black',
            lw=0.4, joinstyle='miter',          # sharp corners
        ))
    return gid, soma_pos


def plot_lfp_and_csd(times, lfp, csd, net=None,
                     cell_type='L5_pyramidal', gid=None,
                     contact_positions=None,
                     csd_vmin=None, csd_vmax=None,
                     titles=("LFP", "CSD"),
                     diam_scale=5.0):
    """LFP | morphology | CSD based on plot_laminar_lfp and plot_laminar_csd, with an optional morphology in the middle."""
    if contact_positions is None:
        contact_labels = np.arange(lfp.shape[0], dtype=int)
    else:
        contact_labels = np.asarray([p[2] for p in contact_positions], dtype=int)

    if net is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
        ax_morph, ax_lfp, ax_csd = None, axes[0], axes[1]
    else:
        fig = plt.figure(figsize=(18, 5), constrained_layout=True)
        # LFP | morphology | CSD
        gs = fig.add_gridspec(1, 3, width_ratios=[7.0, 1.2, 8.0])
        ax_lfp   = fig.add_subplot(gs[0, 0])
        ax_morph = fig.add_subplot(gs[0, 1])
        ax_csd   = fig.add_subplot(gs[0, 2])

    plot_laminar_lfp(times, lfp, contact_labels=contact_labels,
                     ax=ax_lfp, show=False)
    ax_lfp.set_title(titles[0])

    plot_laminar_csd(times, csd, contact_labels=contact_labels, ax=ax_csd,
                     vmin=csd_vmin, vmax=csd_vmax, show=False)
    ax_csd.set_title(titles[1])

    if ax_morph is not None:
        gid_used, _ = _draw_cell_morphology(
            ax_morph, net, cell_type=cell_type, gid=gid, diam_scale=diam_scale,
        )
        ax_morph.set_title(f'{cell_type}\n(gid={gid_used})', fontsize=10)
        ax_morph.set_xlabel('x (µm)')

        # Match depth range to the probe / CSD axis
        z_min = float(min(contact_labels.min(), -125))
        z_max = float(max(contact_labels.max(), 2050))
        ax_morph.set_ylim(z_min, z_max)

        # Autoscale x to fit the cell + a little padding
        ax_morph.relim()
        ax_morph.autoscale(axis='x', tight=False)

        # Hide the morph y-tick labels — the CSD on the right shows the depth
        ax_morph.tick_params(axis='y', labelleft=False)

        # Light contact-position guides
        for z in contact_labels:
            ax_morph.axhline(z, color='lightgray', lw=0.3, zorder=0)

    plt.show()
    return fig, (ax_morph, ax_lfp, ax_csd)


# to be swapped with the one in hnn_core.viz once happy with it
# it allows to keep using the old version of plot_laminar_lfp if voltage_offset is not None
# it introduces an option depth-based mode
def plot_laminar_lfp_AC(
    times,
    data,
    contact_labels,
    tmin=None,
    tmax=None,
    ax=None,
    decim=None,
    color="cividis",
    voltage_offset=50,
    voltage_scalebar=200,
    show=True,
    depth_plot=False,
    baseline_samps=None,
    scale=1.0,
    invert_yaxis=False,
):
    """Plot laminar extracellular electrode array voltage time series.

    This is a drop-in extension of :func:`hnn_core.viz.plot_laminar_lfp` that
    keeps the legacy ``voltage_offset``-based behavior and adds an optional
    ``depth_plot`` mode in which traces are plotted around their actual
    cortical depth values via :func:`plot_stacked_traces`.

    Parameters
    ----------
    times : array-like, shape (n_times,)
        Sampling times (in ms).
    data : Two-dimensional Numpy array
        The extracellular voltages as an (n_contacts, n_times) array.
    contact_labels : list
        In ``voltage_offset`` mode: tick labels for the contacts. In
        ``depth_plot`` mode: the actual depth (y) values for each contact.
    ax : instance of matplotlib figure | None
        The matplotlib axis.
    decim : int | list of int | None (default)
        Optional (integer) factor by which to decimate the raw dipole traces.
        The SciPy function :func:`~scipy.signal.decimate` is used, which
        recommends values <13. To achieve higher decimation factors, a list of
        ints can be provided. These are applied successively.
    color : str | array of floats | ``matplotlib.colors.ListedColormap``
        The colormap to use for plotting. The usual Matplotlib standard
        colormap strings may be used (e.g., 'jetblue'). A color can also be
        defined as an RGBA-quadruplet, or an array of RGBA-values (one for each
        electrode contact trace to plot). An instance of
        :class:`~matplotlib.colors.ListedColormap` may also be provided.
    voltage_offset : float | None (optional)
        Amount to offset traces by on the voltage-axis. Useful for plotting
        laminar arrays. Ignored when ``depth_plot=True``.
    voltage_scalebar : float | None (optional)
        Height, in units of uV, of a scale bar to plot in the top-left corner
        of the plot. Disabled when ``depth_plot=True``.
    show : bool
        If True, show the figure.
    depth_plot : bool
        If True, draw stacked traces around their actual cortical depth (taken
        from ``contact_labels``) using :func:`plot_stacked_traces`. The
        ``voltage_offset`` / ``voltage_scalebar`` machinery is bypassed.
    baseline_samps : int | None
        Only used when ``depth_plot=True``. If not None, subtract the mean of
        the first ``baseline_samps`` samples from each trace before stacking.
    scale : float
        Only used when ``depth_plot=True``. Trace amplitude multiplier applied
        before subtracting from the depth value.
    invert_yaxis : bool
        Only used when ``depth_plot=True``. If True, invert the depth axis so
        that the cortical surface is at the top.

    Returns
    -------
    fig : instance of plt.fig
        The matplotlib figure handle into which time series were plotted.
    """
    import warnings
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # --- Type coercion / validation ---
    if isinstance(times, list):
        times = np.array(times)
    if isinstance(data, list):
        data = np.array(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if len(times) != data.shape[1]:
        raise ValueError(
            f"length of times ({len(times)}) and data ({len(data)}) do not match"
        )

    n_contacts = data.shape[0]

    # --- depth_plot validation ---
    if depth_plot:
        # In depth-plot mode contact_labels are actual y-values, not labels.
        depths = np.asarray(contact_labels, dtype=float)
        if len(depths) != n_contacts:
            raise ValueError(
                f"contact_labels is length {len(depths)}, "
                f"but data has {n_contacts} contacts"
            )

    # --- Color parsing ---
    if color is not None:
        if isinstance(color, (tuple, list)):
            if (
                not np.all([isinstance(c, float) for c in color])
                or len(color) < 3
                or len(color) > 4
            ):
                raise ValueError(f"color must be length 3 or 4, got {color}")
        elif isinstance(color, np.ndarray):
            if color.shape[0] != n_contacts or (
                color.shape[1] < 3 or color.shape[1] > 4
            ):
                raise ValueError(f"color must be n_contacts x (3 or 4), got {color}")
        elif isinstance(color, ListedColormap):
            if color.N != n_contacts:
                raise ValueError(
                    f"ListedColormap has N={color.N}, but "
                    f"there are {n_contacts} contacts"
                )
        elif isinstance(color, str):
            color = plt.get_cmap(color, len(contact_labels))

    if ax is None:
        _, ax = plt.subplots(1, 1)

    n_offsets = n_contacts
    trace_offsets = np.zeros((n_offsets, 1))
    if voltage_offset is not None and not depth_plot:
        trace_offsets = np.arange(n_offsets)[:, np.newaxis] * voltage_offset

    # --- Per-contact loop: just the plotting ---
    for contact_no, trace in enumerate(np.atleast_2d(data)):
        plot_data = trace
        plot_times = times

        if decim is not None:
            plot_data, plot_times = _decimate_plot_data(decim, plot_data, plot_times)

        if isinstance(color, np.ndarray):
            col = color[contact_no]
        elif isinstance(color, ListedColormap):
            col = color(contact_no)
        else:
            col = color

        if depth_plot:
            # Delegate the stacking to plot_stacked_traces, one row at a time
            # so each contact can have its own color from the colormap.
            plot_stacked_traces(
                ax=ax,
                t=plot_times,
                data=plot_data[None, :],
                depths=[depths[contact_no]],
                color=col,
                scale=scale,
                baseline_samps=baseline_samps,
                labels=[f"C{contact_no}"],
            )
        else:
            ax.plot(
                plot_times,
                plot_data + trace_offsets[contact_no],
                label=f"C{contact_no}",
                color=col,
            )

    # --- xlim (set once, outside the loop) ---
    if tmin is not None or tmax is not None:
        ax.set_xlim(left=tmin, right=tmax)
        warnings.warn(
            "tmin and tmax are deprecated and will be "
            "removed in future releases of hnn-core. Please"
            "use matplotlib plt.xlim to set tmin and tmax.",
            DeprecationWarning,
        )
    else:
        ax.set_xlim(left=times[0], right=times[-1])

    # --- y-axis setup (set once, outside the loop) ---
    if depth_plot:
        ylabel = "Depth (µm)"

        ax.set_yticks(depths)
        ax.set_yticklabels([f"{d:g}" for d in depths])

        if len(depths) > 1:
            depth_spacing = np.median(np.abs(np.diff(np.sort(depths))))
        else:
            depth_spacing = 100

        ax.set_ylim(depths[0] - depth_spacing, depths[-1] + depth_spacing)

        if invert_yaxis:
            ax.invert_yaxis()

        # No voltage scalebar in depth mode.
        voltage_scalebar = None

    elif voltage_offset is not None:
        ax.set_ylim(-voltage_offset, n_offsets * voltage_offset)
        ylabel = "Individual contact traces"

        if len(contact_labels) != n_offsets:
            raise ValueError(
                f"contact_labels is length {len(contact_labels)},"
                f" but {n_offsets} contacts to be plotted"
            )
        else:
            trace_ticks = np.arange(
                0, len(contact_labels) * voltage_offset, voltage_offset
            )
            ax.set_yticks(trace_ticks)
            ax.set_yticklabels(contact_labels)

        if voltage_scalebar is None:
            voltage_scalebar = voltage_offset
    else:
        ylabel = r"Electric potential ($\mu V$)"
        ax.ticklabel_format(axis="both", scilimits=(-2, 3))

    # --- voltage scalebar (depth_plot already set this to None) ---
    if voltage_scalebar is not None:
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        scalebar = AnchoredSizeBar(
            ax.transData,
            1,
            f"{voltage_scalebar:.0f} " + r"$\mu V$",
            "upper left",
            size_vertical=voltage_scalebar,
            pad=0.1,
            color="black",
            label_top=False,
            frameon=False,
        )
        ax.add_artist(scalebar)

    ax.set_ylabel(ylabel, multialignment="center")
    ax.set_xlabel("Time (ms)")

    plt_show(show)
    return ax.get_figure()

def plot_laminar_csd_AC(
    times,
    data,
    contact_labels,
    ax=None,
    colorbar=True,
    vmin=None,
    vmax=None,
    sink="b",
    interpolation="spline",
    show=True,
):
    """Plot laminar current source density (CSD) estimation from LFP array.

    Parameters
    ----------
    times : Numpy array, shape (n_times,)
        Sampling times (in ms).
    data : array-like, shape (n_channels, n_times)
        CSD data, channels x time.
    ax : instance of matplotlib figure | None
        The matplotlib axis.
    colorbar : bool
        If True (default), adjust figure to include colorbar.
    contact_labels : list
        Labels associated with the contacts to plot. Passed as-is to
        :func:`~matplotlib.axes.Axes.set_yticklabels`.
    vmin: float, optional
        lower bound of the color axis.
        Will be set automatically of None.
    vmax: float, optional
        upper bound of the color axis.
        Will be set automatically of None.
    sink : str
        If set to 'blue' or 'b', plots sinks in blue and sources in red,
        if set to 'red' or 'r', sinks plotted in red and sources blue.
    interpolation : str | None
        If 'spline', will smoothen the CSD using spline method,
        if None, no smoothing will be applied.

    show : bool
        If True, show the plot.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    if sink[0].lower() == "b":
        cmap = "jet"
    elif sink[0].lower() == "r":
        cmap = "jet_r"
    elif sink[0].lower() != "b" or sink[0].lower() != "r":
        raise RuntimeError(
            'Please use sink = "b" or sink = "r".'
            ' Only colormap "jet" is supported for CSD.'
        )

    if interpolation == "spline":
        # create interpolation function
        interp_data = RectBivariateSpline(times, contact_labels, data.T)
        # increase number of contacts
        new_depths = np.linspace(
            contact_labels[0],
            contact_labels[-1],
            contact_labels[-1] - contact_labels[0],
        )
        # interpolate
        data = interp_data(times, new_depths).T
    elif interpolation is None:
        data = data
        new_depths = contact_labels

    # if vmin and vmax are both None, set colormap such that green = zero
    if vmin is None and vmax is None:
        vmin = -np.max(np.abs(data))
        vmax = np.max(np.abs(data))

    im = ax.pcolormesh(
        times, new_depths, data, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("electrode depth")
    if colorbar:
        color_axis = ax.inset_axes([1.05, 0, 0.02, 1], transform=ax.transAxes)
        plt.colorbar(im, ax=ax, cax=color_axis).set_label(r"$CSD (uV/um^{2})$")

    plt.tight_layout()
    plt_show(show)

    return ax.get_figure()


def _relabel_axis_as_depth(ax, z_surface_um, side="left",
                            ylabel="Depth from surface (µm)"):
    """Relabel an axis whose y-data are HNN z values to display depth-from-surface.

    Tick *positions* stay at the same HNN z values (so plotted data aligns),
    but tick *labels* show ``z_surface_um - y``. Matplotlib's auto tick locator
    is preserved — we attach a FuncFormatter so labels stay correct after any
    later autoscale or pan/zoom.
    """
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(lambda y, pos: f"{int(round(z_surface_um - y))}")
    if side == "left":
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(ylabel)
    else:
        # Apply on the right (e.g., for an existing twin/secondary axis).
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(ylabel)


def plot_lfp_and_csd_AC(times, lfp, csd, net=None,
                     dpl=None,
                     cell_types_morph=('L2_pyramidal', 'L5_pyramidal'),
                     cell_x_shifts=(-150.0, 150.0),
                     gid=None,
                     contact_positions=None,
                     csd_vmin=None, csd_vmax=None,
                     titles=("LFP", "CSD"),
                     diam_scale=5.0,
                     z_surface_um=None,
                     show_hnn_axis=True,
                     trial_idx=None,
                     trim_csd_edges=True,
                     lfp_scale=0.8,
                     overlay_csd_traces=True,
                     csd_traces_scale=5e7,
                     show_inputs=False):
    """Dipole/raster (provisional) | LFP | morph | CSD plot with cortical-depth labels.

    The y-axes of the LFP and CSD panels are relabeled to show
    depth-from-surface (0 at the top, positive going down), while the data is
    still drawn in HNN z internally so the panels align with the morphology.
    A secondary axis on the right of the LFP and CSD panels shows the
    original HNN z values for reference.

    When ``dpl`` is provided, a top row is added: dipole on the left and
    spike raster on the right. Both share the same column widths as the
    LFP/CSD row so the panels line up.

    Parameters
    ----------
    times, lfp, csd : array-like
        See plot_laminar_lfp_AC / plot_laminar_csd.
    net : Network | None
        If provided, a morphology cartoon is drawn between the LFP and CSD,
        and the spike raster is drawn in the top row when ``dpl`` is supplied.
    dpl : Dipole | list of Dipole | None
        Dipole object(s) to plot in the top-left panel. If None, no top row.
    cell_types_morph : tuple of str
        Cell types to draw in the morphology panel, in left-to-right order.
    cell_x_shifts : tuple of float
        Per-cell horizontal offsets (in µm) used when drawing the morphologies
        side by side. Must match the length of ``cell_types_morph``.
    gid : int | None
        gid to draw for each cell type. If None, the first gid of each cell
        type is used.
    contact_positions : iterable | None
        (x, y, z) tuples for each contact. Used to derive HNN z labels.
    csd_vmin, csd_vmax : float | None
        Color limits for the CSD heatmap.
    titles : tuple of str
        Titles for the LFP and CSD panels.
    z_surface_um : float | None
        z value (in HNN coords) that corresponds to depth = 0. If None, the
        maximum contact z is used.
    show_hnn_axis : bool
        If True, add a right-side secondary y-axis on the LFP and CSD panels
        showing HNN z values.
    trial_idx : int | list of int | None
        Forwarded to plot_spikes_raster.
    trim_csd_edges : bool
        If True (default), drop the topmost and bottommost rows of ``csd``
        from the CSD panel. Those two rows are the linearly-extrapolated
        borders added inside ``calculate_csd2d`` (see ``reconstruct_csd``)
        and can produce edge artifacts. The LFP and morphology panels still
        use the full set of contacts.
    lfp_scale : float
        Trace amplitude scale for the stacked LFP traces (forwarded to
        ``plot_laminar_lfp_AC`` as ``scale``).
    overlay_csd_traces : bool
        If True (default), overlay the CSD traces on the heatmap as stacked
        traces using ``plot_stacked_traces``.
    csd_traces_scale : float
        Trace amplitude scale for the overlaid CSD traces. Default 5e7 is
        tuned for typical CSD magnitudes (~1e-6 µV/µm²) and 100-µm contact
        spacing so peak deflections are ~50 µm.
    show_inputs : bool
        If True, add a top row of histogram panels above the LFP and CSD
        panels showing the per-drive event-time histograms via
        ``hnn_core.viz.plot_spikes_hist``. Requires ``net``. Default False.
    """
    from hnn_core.viz import plot_dipole, plot_spikes_raster, plot_spikes_hist

    if contact_positions is None:
        contact_labels = np.arange(lfp.shape[0], dtype=int)
    else:
        contact_labels = np.asarray([p[2] for p in contact_positions], dtype=int)

    if z_surface_um is None:
        z_surface_um = float(np.max(contact_labels))

    if len(cell_x_shifts) != len(cell_types_morph):
        raise ValueError(
            f"cell_x_shifts (len {len(cell_x_shifts)}) must match "
            f"cell_types_morph (len {len(cell_types_morph)})"
        )

    if show_inputs and net is None:
        raise ValueError("show_inputs=True requires `net` (uses net.cell_response).")

    # --- Main figure: LFP | morph | CSD ---
    # LFP and CSD get equal width. When show_inputs=True, an extra top row
    # holds histogram strips above LFP and CSD (the morph column up there is
    # left empty so the panels align).
    ax_hist_lfp = None
    ax_hist_csd = None
    if net is None:
        fig = plt.figure(figsize=(15, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[8.0, 8.0])
        ax_lfp   = fig.add_subplot(gs[0, 0])
        ax_morph = None
        ax_csd   = fig.add_subplot(gs[0, 1])
    elif show_inputs:
        fig = plt.figure(figsize=(17, 6), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, width_ratios=[8.0, 1.2, 8.0],
                              height_ratios=[1.0, 4.0])
        ax_hist_lfp = fig.add_subplot(gs[0, 0])
        # gs[0, 1] (above the morph) intentionally left empty
        ax_hist_csd = fig.add_subplot(gs[0, 2])
        ax_lfp     = fig.add_subplot(gs[1, 0])
        ax_morph   = fig.add_subplot(gs[1, 1])
        ax_csd     = fig.add_subplot(gs[1, 2])
    else:
        fig = plt.figure(figsize=(17, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[8.0, 1.2, 8.0])
        ax_lfp   = fig.add_subplot(gs[0, 0])
        ax_morph = fig.add_subplot(gs[0, 1])
        ax_csd   = fig.add_subplot(gs[0, 2])

    # --- LFP ---
    plot_laminar_lfp_AC(
        times,
        lfp,
        contact_labels=contact_labels,
        ax=ax_lfp,
        depth_plot=True,
        baseline_samps=20,
        scale=lfp_scale,
        invert_yaxis=False,
        show=False,
    )
    ax_lfp.set_title(titles[0])
    _relabel_axis_as_depth(ax_lfp, z_surface_um, side="left")

    if show_hnn_axis:
        secax_lfp = ax_lfp.secondary_yaxis(
            "right", functions=(lambda y: y, lambda y: y)
        )
        secax_lfp.set_ylabel("HNN z (µm)")

    # --- Top-row input histograms (optional) ---
    if show_inputs:
        # Group every evprox* drive as "Proximal" (red) and every evdist*
        # drive as "Distal" (green) — plot_spikes_hist matches by prefix.
        hist_groups = {"Proximal": ["evprox"], "Distal": ["evdist"]}
        hist_colors = {"Proximal": "red", "Distal": "green"}
        for ax_hist, ax_main in ((ax_hist_lfp, ax_lfp), (ax_hist_csd, ax_csd)):
            plot_spikes_hist(net.cell_response, ax=ax_hist,
                             trial_idx=trial_idx,
                             spike_types=hist_groups,
                             color=hist_colors,
                             show=False)
            # Sync time x-axis with the corresponding panel below.
            ax_hist.sharex(ax_main)
            # The LFP/CSD panels below show the time axis, so suppress here.
            ax_hist.set_xlabel("")
            ax_hist.tick_params(axis="x", labelbottom=False)

    # --- CSD ---
    # Suppress plot_laminar_csd's built-in colorbar so we can put the HNN-z
    # secondary axis at the right edge of the panel and place a custom
    # colorbar further out.
    if trim_csd_edges and len(contact_labels) >= 4:
        # Drop the linearly-extrapolated top and bottom rows of the CSD
        # (those are the boundary rows added inside calculate_csd2d).
        csd_plot = np.asarray(csd)[1:-1]
        csd_labels = np.asarray(contact_labels)[1:-1]
    else:
        csd_plot = csd
        csd_labels = contact_labels

    plot_laminar_csd_AC(times, csd_plot, contact_labels=csd_labels, ax=ax_csd,
                     vmin=csd_vmin, vmax=csd_vmax, colorbar=True, show=False)
    ax_csd.set_title(titles[1])
    _relabel_axis_as_depth(ax_csd, z_surface_um, side="left")

    # Match the CSD y-range AND y-tick positions to the LFP exactly. With
    # trim_csd_edges=True the CSD has 2 fewer rows than the LFP, so the
    # trimmed contact strips at the very top and bottom appear blank.
    ax_csd.set_ylim(ax_lfp.get_ylim())
    ax_csd.set_yticks(ax_lfp.get_yticks())

    #if show_hnn_axis:
    #    secax_csd = ax_csd.secondary_yaxis(
    #        "right", functions=(lambda y: y, lambda y: y)
    #    )
    #    secax_csd.set_ylabel("HNN z (µm)")

    # Place the colorbar in the dedicated gridspec column. Force scientific
    # notation so the tiny CSD values (typical: ~1e-6 µV/µm²) render with a
    # visible exponent rather than as a clipped offset.
    
    '''
    csd_im = next(
        (c for c in ax_csd.collections
         if c.__class__.__name__ in ("QuadMesh", "Collection")
         and c.get_array() is not None),
        None,
    )
    if csd_im is not None:
        plt.colorbar(csd_im, cax=cbar_ax)
        #from matplotlib.ticker import ScalarFormatter
        #fmt = ScalarFormatter(useMathText=True)
        #fmt.set_scientific(False)
        #fmt.set_powerlimits((-2, 2))
        #cbar = plt.colorbar(csd_im, cax=cbar_ax, format=fmt)
        #cbar.set_label(r"$CSD\ (\mu V/\mu m^{2})$")
    else:
        cbar_ax.axis("off")
    '''

    # Overlay stacked CSD traces on top of the heatmap.
    # Note: data is negated so the standard CSD convention applies — sinks
    # (negative CSD) deflect downward, sources (positive CSD) upward.
    if overlay_csd_traces:
        plot_stacked_traces(
            ax=ax_csd,
            t=times,
            data=-np.asarray(csd_plot),
            depths=csd_labels,
            color="k",
            lw=0.6,
            alpha=0.7,
            scale=csd_traces_scale,
            baseline_samps=20,
        )

    # --- Morphology (in HNN z, aligned with LFP/CSD data coords) ---
    if ax_morph is not None:
        gid_labels = []
        for cell_type, x_shift in zip(cell_types_morph, cell_x_shifts):
            gid_used, _ = _draw_cell_morphology(
                ax_morph, net, cell_type=cell_type, gid=gid,
                diam_scale=diam_scale, x_shift=x_shift,
            )
            gid_labels.append(f"{cell_type}\n(gid={gid_used})")

        ax_morph.set_title(" | ".join(gid_labels), fontsize=8)
        ax_morph.set_xlabel('x (µm)')

        # Match depth range to the probe / CSD axis
        z_min = float(min(contact_labels.min(), -125)) # why these????
        z_max = float(max(contact_labels.max(), 2050)) # why L2 cell thinner??
        ax_morph.set_ylim(z_min, z_max)

        # Autoscale x to fit both cells + a little padding
        ax_morph.relim()
        ax_morph.autoscale(axis='x', tight=False)

        # Hide the morph x and y-tick labels — depth/HNN labels are on LFP and CSD
        ax_morph.tick_params(axis='y', labelleft=False)
        ax_morph.tick_params(axis='x', labelbottom=False)

        # Light contact-position guides
        for z in contact_labels:
            ax_morph.axhline(z, color='lightgray', lw=0.3, zorder=0)

    # --- Optional separate "Full network" figure: dipole | raster ---
    fig_full = None
    ax_dpl = None
    ax_raster = None
    if dpl is not None:
        fig_full = plt.figure(figsize=(14, 4), constrained_layout=True)
        gs_full = fig_full.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
        ax_dpl    = fig_full.add_subplot(gs_full[0, 0])
        ax_raster = fig_full.add_subplot(gs_full[0, 1])

        plot_dipole(dpl, ax=ax_dpl, show=False)
        ax_dpl.set_title("Dipole")

        if net is not None:
            plot_spikes_raster(net.cell_response, trial_idx=trial_idx,
                               ax=ax_raster, show=False)
            ax_raster.set_title("Spike raster")
        else:
            ax_raster.text(0.5, 0.5, "(no net provided)",
                           ha="center", va="center")
            ax_raster.set_axis_off()

        fig_full.suptitle("Full network", fontweight="bold")

    plt.show()
    return fig, (ax_dpl, ax_raster, ax_lfp, ax_morph, ax_csd)



'''
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline
from matplotlib.ticker import MultipleLocator
from hnn_core.extracellular import _get_laminar_z_coords


def plot_stacked_traces(
    ax,
    t,
    data,
    depths,
    color="k",
    lw=0.8,
    alpha=1.0,
    scale=1.0,
    baseline_samps=30, # check this
    labels=None,
):
    """Plot traces stacked by cortical depth.

    Each trace is baseline-corrected (if applicable) and then plotted around its corresponding
    depth value.
    """
    depths = np.asarray(depths).astype(float)
    data = np.asarray(data)

    for i, (z, tr) in enumerate(zip(depths, data)):
        tr0 = tr - np.mean(tr[:baseline_samps])
        lab = labels[i] if labels is not None else None
        ax.plot(t, z - scale * tr0, color=color, lw=lw, alpha=alpha, label=lab)

def plot_lfp_and_csd_AC(
    times,
    lfp,
    csd,
    net=None,
    contact_positions=None,
    array_name="probe1",
    titles=("LFP", "CSD"),
    baseline_samps=20,
    lfp_scale=0.8,
    csd_scale=1.1,
    xlim=(0, 250),
    time_offset_ms=0.0,
    sink="b",                 # <-- new: "b" => sinks blue, "r" => sinks red
    interpolation="spline",   # <-- new: "spline" or None
    vmin=None,
    vmax=None,
    invert_yaxis=False,
):
    """Plot stacked LFP traces and CSD heatmap (+ stacked CSD traces).

    Mirrors the conventions of hnn_core.viz.plot_laminar_csd:
      - `sink` selects 'jet' (blue sinks) or 'jet_r' (red sinks)
      - if vmin/vmax are both None, color limits are symmetric around zero
      - `interpolation` controls depth-axis spline smoothing
    """
    times = np.asarray(times)
    lfp = np.asarray(lfp)
    csd = np.asarray(csd)

    # --- Sink convention -> colormap (matches plot_laminar_csd) -------------
    if sink[0].lower() == "b":
        cmap = "jet"
    elif sink[0].lower() == "r":
        cmap = "jet_r"
    else:
        raise ValueError('Use sink="b" or sink="r".')

    # --- Contact geometry ----------------------------------------------------
    if contact_positions is None:
        if net is None:
            raise ValueError("Provide either net or contact_positions.")
        contact_positions = net.rec_arrays[array_name].positions

    contact_labels, _ = _get_laminar_z_coords(contact_positions)
    contact_labels = np.asarray(contact_labels)
    depths_lfp = contact_labels

    if csd.shape[0] == len(contact_labels) - 2:
        depths_csd = contact_labels[1:-1]
    elif csd.shape[0] == len(contact_labels):
        depths_csd = contact_labels
    else:
        raise ValueError(
            f"csd has {csd.shape[0]} depth rows, but contact_labels has "
            f"{len(contact_labels)} entries. Expected either n_contacts or "
            f"n_contacts - 2."
        )

    # --- Baseline correction on CSD -----------------------------------------
    csd_bc = csd - np.mean(csd[:, :baseline_samps], axis=1, keepdims=True)

    # --- Time vector ---------------------------------------------------------
    t = times + time_offset_ms

    # --- Optional spline smoothing over depth (matches reference) -----------
    if interpolation == "spline":
        interp_data = RectBivariateSpline(t, depths_csd, csd_bc.T)
        new_depths = np.linspace(
            depths_csd[0],
            depths_csd[-1],
            int(abs(depths_csd[-1] - depths_csd[0])),
        )
        data_csd_plot = interp_data(t, new_depths).T
    elif interpolation is None:
        data_csd_plot = csd_bc
        new_depths = depths_csd
    else:
        raise ValueError('interpolation must be "spline" or None')

    # --- Symmetric color limits if both unset (matches reference) -----------
    if vmin is None and vmax is None:
        vmax = float(np.nanmax(np.abs(data_csd_plot)))
        vmin = -vmax

    # --- Figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Left: stacked LFP
    plot_stacked_traces(
        ax=ax[0], t=t, data=lfp, depths=depths_lfp,
        color="k", lw=0.8, alpha=1.0,
        scale=lfp_scale, baseline_samps=baseline_samps,
    )
    ax[0].set_title(titles[0])
    ax[0].set_ylabel("Depth (µm)")
    ax[0].set_xlim(xlim)

    # Right: CSD heatmap + stacked CSD traces overlay
    im = ax[1].pcolormesh(
        t, new_depths, data_csd_plot,
        cmap=cmap, shading="auto", rasterized=False,
        vmin=vmin, vmax=vmax,
    )
    plot_stacked_traces(
        ax=ax[1], t=t, data=csd_bc, depths=depths_csd,
        color="k", lw=0.7, alpha=0.9,
        scale=csd_scale, baseline_samps=baseline_samps,
    )
    ax[1].set_title(titles[1])
    ax[1].set_ylabel("Depth (µm)")
    ax[1].set_xlim(xlim)

    cbar = fig.colorbar(im, ax=ax[1])
    cbar.set_label(r"CSD (uV/µm$^2$)")

    for axis in ax:
        axis.xaxis.set_major_locator(MultipleLocator(25))
        axis.xaxis.set_minor_locator(MultipleLocator(5))
        axis.set_xlabel("Time (ms)")
        if invert_yaxis:
            axis.invert_yaxis()

    plt.tight_layout()
    return fig, ax



def plot_lfp_and_csd_AC(
    times,
    lfp,
    csd,
    net=None,
    contact_positions=None,
    array_name="probe1",
    titles=("LFP", "CSD"),
    baseline_samps=20, # 100 in NHP
    lfp_scale=0.8,
    csd_scale=1.1,
    xlim=(0, 250),
    time_offset_ms=0.0,
    cmap="jet_r",
    vmin=None,
    vmax=None,
    invert_yaxis=False,
):
    """Plot stacked LFP traces and CSD heatmap with stacked CSD traces.
    Not based on current HNN plotting functions.

    Parameters
    ----------
    times : array-like
        Time vector, expected in ms.
    lfp : array, shape (n_contacts, n_times)
        LFP data.
    csd : array, shape (n_csd_contacts, n_times)
        CSD data. Usually this has two fewer depth rows than LFP if computed
        with a second spatial derivative.
    net : hnn_core.Network | None
        Network containing the electrode array. Used to infer contact labels
        if contact_positions is not provided.
    contact_positions : array-like | None
        Electrode positions. If provided, used to infer contact labels.
    array_name : str
        Name of the electrode array in net.rec_arrays.
    titles : tuple
        Titles for the LFP and CSD panels.
    baseline_samps : int
        Number of initial samples used for baseline correction.
    lfp_scale : float
        Scale factor for stacked LFP traces.
    csd_scale : float
        Scale factor for stacked CSD traces.
    xlim : tuple
        X-axis limits in ms.
    time_offset_ms : float
        Optional shift applied to the time vector.
    cmap : str
        Colormap for CSD heatmap.
    vmin, vmax : float | None
        Optional color limits for CSD heatmap.
    invert_yaxis : bool
        Whether to invert the depth axis.
    """
    times = np.asarray(times)
    lfp = np.asarray(lfp)
    csd = np.asarray(csd)

    if contact_positions is None:
        if net is None:
            raise ValueError("Provide either net or contact_positions.")
        contact_positions = net.rec_arrays[array_name].positions

    contact_labels, _ = _get_laminar_z_coords(contact_positions)
    contact_labels = np.asarray(contact_labels)

    depths_lfp = contact_labels

    # If CSD has two fewer contacts than LFP, use the inner contacts.
    if csd.shape[0] == len(contact_labels) - 2:
        depths_csd = contact_labels[1:-1]
    elif csd.shape[0] == len(contact_labels):
        depths_csd = contact_labels
    else:
        raise ValueError(
            f"csd has {csd.shape[0]} depth rows, but contact_labels has "
            f"{len(contact_labels)} entries. Expected either n_contacts or "
            f"n_contacts - 2."
        )

    # Baseline correction for CSD before interpolation.
    csd_bc = csd - np.mean(csd[:, :baseline_samps], axis=1, keepdims=True) # CHECK THIS!

    # Time vector in ms, with optional offset.
    t = times + time_offset_ms

    # Interpolate CSD over depth for smooth heatmap.
    new_depths = np.linspace(
        depths_csd[0],
        depths_csd[-1],
        int(abs(depths_csd[-1] - depths_csd[0])) + 1,
    )

    interp_data = RectBivariateSpline(
        t,
        depths_csd,
        csd_bc.T,
    )

    data_csd_interp = interp_data(t, new_depths).T

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
    )

    # Left: stacked LFP traces
    plot_stacked_traces(
        ax=ax[0],
        t=t,
        data=lfp,
        depths=depths_lfp,
        color="k",
        lw=0.8,
        alpha=1.0,
        scale=lfp_scale,
        baseline_samps=baseline_samps,
    )

    ax[0].set_title(titles[0])
    ax[0].set_ylabel("Depth (µm)")
    ax[0].set_xlim(xlim)

    # Right: CSD heatmap + stacked CSD traces
    im = ax[1].pcolormesh(
        t,
        new_depths,
        data_csd_interp,
        cmap=cmap,
        shading="auto",
        rasterized=False,
        vmin=vmin,
        vmax=vmax,
    )
    #if colorbar:
    #    color_axis = ax.inset_axes([1.05, 0, 0.02, 1], transform=ax.transAxes)
    #    plt.colorbar(im, ax=ax, cax=color_axis).set_label(r"$CSD (uV/um^{2})$")

    #plt.tight_layout()
    

    plot_stacked_traces(
        ax=ax[1],
        t=t,
        data=csd_bc,
        depths=depths_csd,
        color="k",
        lw=0.7,
        alpha=0.9,
        scale=csd_scale,
        baseline_samps=baseline_samps,
    )

    ax[1].set_title(titles[1])
    ax[1].set_ylabel("Depth (µm)")
    ax[1].set_xlim(xlim)

    cbar = fig.colorbar(im, ax=ax[1])
    cbar.set_label(r"CSD")

    for axis in ax:
        axis.xaxis.set_major_locator(MultipleLocator(25))
        axis.xaxis.set_minor_locator(MultipleLocator(5))
        axis.set_xlabel("Time (ms)")

        if invert_yaxis:
            axis.invert_yaxis()

    plt.tight_layout()

    return fig, ax
'''