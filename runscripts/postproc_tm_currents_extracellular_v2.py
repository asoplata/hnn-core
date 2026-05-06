"""Post hoc extracellular decomposition from HNN transmembrane-current recordings.

This module is designed for exploratory work on top of the transmembrane-current
recording branch discussed in HNN PR #1212 (and ####). It reconstructs approximate LFP/CSD
contributions from:

1. intrinsic / transmembrane current channels recorded per segment
2. synaptic currents recorded at section midpoints, using a midpoint-segment
   approximation

Assumptions / limitations
-------------------------
- This code assumes the PR branch stores recordings under
  ``net.cell_response.transmembrane_currents`` and ``net.cell_response.isec``.
- Intrinsic/mechanism currents are assumed to be recorded as current densities
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

# Channels exposed by the tm_currents branch
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


def _default_template_builder(cell_type: str):
    """Build a template cell for geometry/area lookup.

    Currently supports HNN pyramidal cell types. Basket cells, being single-compartment, don't contribute to LFP/CSD
    """
    load_custom_mechanisms()
    if cell_type in {"L2_pyramidal", "L5_pyramidal"}:
        cell = pyramidal(cell_name=cell_type)
        cell.build(sec_name_apical="apical_trunk")
        return cell
    raise NotImplementedError(
        f"No default template builder for {cell_type!r}. "
    )


#def _get_template_cells(cell_types):
#    return {
#        cell_type: _default_template_builder(cell_type)
#        for cell_type in cell_types
#    }


# returns 3D position of the soma for the given gid, used as reference point for extracellular calculations
def _get_gid_soma_pos(net, cell_type: str, gid: int) -> np.ndarray:
    # gid_ranges tell where the gids for this cell type start
    start_gid = net.gid_ranges[cell_type][0]
    return np.asarray(net.pos_dict[cell_type][gid - start_gid], dtype=float)

#returns standard segment-center x locations for a section with nseg segments, used for area lookup and midpoint-segment approximation
def _segment_xs_for_section(nseg: int) -> np.ndarray:
    return np.asarray([(i - 0.5) / nseg for i in range(1, nseg + 1)], dtype=float)


# returns the membrane area of one segment, given section and segment location
#def _segment_area_um2(template_cell, section: str, segment_x: float) -> float:
#    #seg = template_cell._nrn_sections[section](float(segment_x))
#    seg = template_cell.sections[section](float(segment_x))
#    return float(seg.area())
def _segment_area_um2(template_cell, section: str, segment_x: float) -> float:
    sec = template_cell.sections[section]
    return float(np.pi * sec.diam * (sec.L / sec.nseg))

def _density_to_nA(current_density, area_um2: float) -> np.ndarray:
    """Convert current density mA/cm^2 -> nA using segment area in um^2."""
    return np.asarray(current_density, dtype=float) * area_um2 * 1e-2


# Stack a list of 1D time series into a 2D array, ensuring the result is always 2D even if the input is empty or has only one source.
def _ensure_2d_timeseries(matrix_like: list[np.ndarray]) -> np.ndarray:
    if len(matrix_like) == 0:
        return np.empty((0, 0), dtype=float)
    return np.vstack([np.asarray(x, dtype=float)[None, :] for x in matrix_like])


# choose which segment or segments should represent the midpoint of a section for synaptic current assignment, and how to weight them if there are multiple. For even nseg, the default is to split evenly between the two central segments, but other options are provided.
def _pick_midpoint_segments(nseg: int, mode: str = "split_even"):
    """Return [(segment_index, weight_fraction), ...] for midpoint approximation."""
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
    """Collect intrinsic/mechanism current sources as post hoc source vectors.

    Returns
    -------
    sources : list[SourceInfo]
    current_matrix_nA : ndarray, shape (n_sources, n_times)
    """
    channels = tuple(channels or sorted(SUPPORTED_INTRINSIC_CHANNELS))
    bad = sorted(set(channels) - SUPPORTED_INTRINSIC_CHANNELS)
    if bad:
        raise ValueError(f"Unsupported intrinsic channels: {bad}")

    #template_cells = _get_template_cells(cell_types, template_builders)
    #template_cells = net.cell_types[cell_type]["cell_object"]
    sources: list[SourceInfo] = []
    currents_nA = []

    for cell_type in cell_types:
        template_cell = net.cell_types[cell_type]["cell_object"]
        #template_cell = template_cells[cell_type]
        
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
        #template_cell = template_cells[cell_type]
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

    #print(sorted({s.syn_name for s in sources}))

    return sources, _ensure_2d_timeseries(currents_nA)


# note: trying to not use _transfer_resistance()

#import numpy as np
#from numpy.linalg import norm

#from end-points
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
    """Postprocessing equivalent of _transfer_resistance.

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

    if method == "psa":
        dis = norm(np.tile(electrode_pos, (nseg, 1)) - seg_ctr, axis=1)
        dis = np.maximum(dis, min_distance)
        phi = 1.0 / dis

    elif method == "lsa":
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


# note: HNN has pairwise transfer resistance calculation via _transfer_resistance().
# That function computes the transfer resistance between one NEURON section 
# and one electrode position. That value is multiplied by the transmembrane current
# to get the extracellular potential.
# The following function does the same, just not on the fly but as a post hoc calculation, and it caches transfer resistances for sections that are reused across multiple sources.
def build_transfer_resistance_matrix_for_sources(
    net,
    sources: Sequence[SourceInfo],
    array_name: str = "probe1",
):
    """Build electrode x source transfer matrix for postprocessed sources."""
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


# reconstructs the LFP time series by multiplying the transfer resistance matrix (electrodes x sources) with the current matrix (sources x time) to get the LFP at each electrode over time. The result is in microvolts (uV).
# check _nrn_r_transfer * _nrn_imem_vec in hnn_core.extracellular._transfer_resistance for HNN's version
def reconstruct_lfp_from_sources(
    transfer_matrix: np.ndarray,
    current_matrix_nA: np.ndarray,
) -> np.ndarray:
    """Return LFP (uV), shape (n_contacts, n_times)."""
    if transfer_matrix.shape[1] != current_matrix_nA.shape[0]:
        raise ValueError(
            "transfer_matrix columns must match current_matrix_nA rows: "
            f"got {transfer_matrix.shape[1]} vs {current_matrix_nA.shape[0]}"
        )
    return transfer_matrix @ current_matrix_nA


# high-level wrapper to go from net and source selection parameters all the way to reconstructed LFP. 
# Returns the LFP, the list of sources, the transfer matrix, and the current matrix for further analysis.
def reconstruct_intrinsic_lfp(
    net,
    trial_idx: int = 0,
    cell_types: Sequence[str] = ("L2_pyramidal", "L5_pyramidal"),
    channels: Sequence[str] | None = None,
    array_name: str = "probe1",
    cap_current_sign: float = -1.0,
):
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

    '''
    lfp_syn, sources, T_syn, I_syn = reconstruct_synaptic_lfp(
        net,
        trial_idx=trial_idx,
        cell_types=cell_types,
        array_name=array_name,
        midpoint_mode=midpoint_mode,
    )

    if syn_names is not None:
        sources, I_syn = filter_sources(
            sources,
            I_syn,
            syn_names=syn_names,
        )
        T_syn = build_transfer_resistance_matrix_for_sources(
            net,
            sources,
            array_name=array_name,
        )
        lfp_syn = reconstruct_lfp_from_sources(T_syn, I_syn)

    return lfp_syn, sources, T_syn, I_syn
    '''

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


# makes a count summary of the sources, grouped by:
# - src.kind (intrinsic vs synaptic)
# - src.cell_type (e.g., L2_pyramidal)
# - src.label (e.g., "ina_hh2" or "isec")
# question: how many sources do I have of each type, and how does that relate to the LFP/CSD contributions?
def summarize_sources(sources: Sequence[SourceInfo]):
    summary = {}
    for src in sources:
        key = (src.kind, src.cell_type, src.label)
        summary[key] = summary.get(key, 0) + 1
    return summary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from hnn_core.viz import plot_laminar_lfp, plot_laminar_csd


def _draw_cell_morphology(ax, net, cell_type='L5_pyramidal', gid=None,
                          color_by_region=True, center_x=True,
                          diam_scale=5.0):
    """Draw one cell's sections as rectangles in the (x, z) plane."""
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
        x0, z0 = pts[0, 0] - x_off, pts[0, 2]
        x1, z1 = pts[1, 0] - x_off, pts[1, 2]
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
    """LFP | morphology | CSD."""
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

'''
import matplotlib.pyplot as plt
import numpy as np
from hnn_core.viz import plot_laminar_lfp, plot_laminar_csd

def plot_lfp_and_csd(times, lfp, csd, contact_positions=None, csd_vmin=None, csd_vmax=None,
                     titles=("LFP", "CSD")):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)

    if contact_positions is None:
        contact_labels = np.arange(lfp.shape[0], dtype=int)
    else:
        contact_labels = np.asarray([pos[2] for pos in contact_positions], dtype=int)

    plot_laminar_lfp(
        times,
        lfp,
        contact_labels=contact_labels,
        ax=axes[0],
        show=False
    )
    axes[0].set_title(titles[0])

    plot_laminar_csd(
        times,
        csd,
        contact_labels=contact_labels,
        ax=axes[1],
        vmin=csd_vmin,
        vmax=csd_vmax,
        show=False
    )
    axes[1].set_title(titles[1])

    plt.show()
    return fig, axes
'''
'''
import matplotlib.pyplot as plt
from hnn_core.viz import plot_laminar_lfp, plot_laminar_csd

def plot_lfp_and_csd(times, lfp, csd, contact_positions=None,
                     titles=("LFP", "CSD")):
    fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    if contact_positions is None:
        contact_labels = [str(ii) for ii in range(lfp.shape[0])]
    else:
        contact_labels = [f"{pos[2]:.0f}" for pos in contact_positions]

    contact_labels = np.asarray(contact_labels, dtype=int)

    plot_laminar_lfp(
        times, lfp, contact_labels,
        ax=axes[0], show=False
    )
    axes[0].set_title(titles[0])

    plot_laminar_csd(
        times, csd, contact_labels,
        ax=axes[1], show=False
    )
    axes[1].set_title(titles[1])

    plt.show()

    return fig, axes
'''
#use net.rec_arrays[array_name].plot_lfp() and .plot_csd() when plotting actual HNN-recorded arrays

'''
def plot_lfp(times, lfp, contact_positions=None, ax=None, voltage_offset=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    lfp = np.asarray(lfp)
    n_contacts = lfp.shape[0]
    if voltage_offset is None:
        voltage_offset = np.nanmax(np.abs(lfp)) * 1.2 if lfp.size else 1.0
    offsets = np.arange(n_contacts)[::-1] * voltage_offset
    for idx in range(n_contacts):
        ax.plot(times, lfp[idx] + offsets[idx])
    if contact_positions is not None:
        contact_positions = np.asarray(contact_positions)
        z = contact_positions[:, 2]
        ax.set_yticks(offsets)
        ax.set_yticklabels([f"{zz:.0f}" for zz in z])
        ax.set_ylabel("z (um)")
    else:
        ax.set_ylabel("contact")
    ax.set_xlabel("time (ms)")
    if title:
        ax.set_title(title)
    return ax


def plot_csd(times, csd, contact_positions=None, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    csd = np.asarray(csd)
    extent = [times[0], times[-1], 0, csd.shape[0] - 1]
    if contact_positions is not None:
        z = np.asarray(contact_positions)[:, 2]
        extent = [times[0], times[-1], z[-1], z[0]]
    im = ax.imshow(csd, aspect="auto", extent=extent)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("z (um)" if contact_positions is not None else "contact")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="CSD")
    return ax


def plot_lfp_and_csd(times, lfp, csd, contact_positions=None, titles=("LFP", "CSD")):
    fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    plot_lfp(times, lfp, contact_positions=contact_positions, ax=axes[0], title=titles[0])
    plot_csd(times, csd, contact_positions=contact_positions, ax=axes[1], title=titles[1])
    return fig, axes
'''

def example_usage(net, trial_idx: int = 0, array_name: str = "probe1"):
    """Minimal examples for interactive use.

    Notes
    -----
    This assumes `net` already contains:
    - `net.cell_response.transmembrane_currents`
    - `net.cell_response.isec`
    - `net.rec_arrays[array_name]`
    """
    times = np.asarray(net.rec_arrays[array_name].times)
    contact_positions = net.rec_arrays[array_name].positions

    # Intrinsic: sodium only in L5 pyramidal cells
    lfp_na_l5, src_na_l5, T_na_l5, I_na_l5 = reconstruct_intrinsic_lfp(
        net,
        trial_idx=trial_idx,
        cell_types=["L5_pyramidal"],
        channels=["ina_hh2"],
        array_name=array_name,
    )
    csd_na_l5 = reconstruct_csd(net, lfp_na_l5, array_name=array_name)

    # Synaptic: midpoint-segment approximation
    lfp_syn, src_syn, T_syn, I_syn = reconstruct_synaptic_lfp(
        net,
        trial_idx=trial_idx,
        cell_types=["L2_pyramidal", "L5_pyramidal"],
        array_name=array_name,
        midpoint_mode="split_even",
    )
    csd_syn = reconstruct_csd(net, lfp_syn, array_name=array_name)

    plot_lfp_and_csd(
        times,
        lfp_na_l5,
        csd_na_l5,
        contact_positions=contact_positions,
        titles=("LFP: L5 ina_hh2", "CSD: L5 ina_hh2"),
    )
    plot_lfp_and_csd(
        times,
        lfp_syn,
        csd_syn,
        contact_positions=contact_positions,
        titles=("LFP: synaptic approx", "CSD: synaptic approx"),
    )

    return {
        "lfp_na_l5": lfp_na_l5,
        "csd_na_l5": csd_na_l5,
        "sources_na_l5": src_na_l5,
        "lfp_syn": lfp_syn,
        "csd_syn": csd_syn,
        "sources_syn": src_syn,
    }
