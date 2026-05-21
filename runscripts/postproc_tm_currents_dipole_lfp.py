"""Post hoc extracellular decomposition from HNN transmembrane-current recordings.

This module is designed for exploratory work on top of the transmembrane-current
recording branch discussed in PR #1212 (and ####). It reconstructs approximate LFP/CSD
contributions from:

1. transmembrane current channels recorded per segment
2. synaptic currents recorded at section midpoints, using a midpoint-segment
   approximation

Assumptions
-----------
- This code assumes the PR branch stores recordings under
  ``net.cell_response.transmembrane_currents`` and ``net.cell_response.isec``.
- Intrinsic currents are assumed to be recorded as current densities
  (mA / cm^2) except for ``agg_i_mem``, which is assumed to already be in nA.
- Synaptic currents located at the midpoint segment of
  the section. 

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
    #"agg_ina",
    #"agg_ik",
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
    (I guess it's never the case) For even nseg, the default is to split evenly between the two central segments, 
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

from collections import defaultdict
import numpy as np

def sum_intrinsic_sources_by_segment(sources_intr, I_intr, to_mA=True):
    """Sum collect_intrinsic_sources output across channels per segment.

    Parameters
    ----------
    sources_intr : list[SourceInfo]
        Sources returned by collect_intrinsic_sources.
    I_intr : ndarray, shape (n_sources, n_times)
        Current matrix returned by collect_intrinsic_sources.
        Expected units: nA.
    to_mA : bool
        If True, convert summed currents from nA to mA.

    Returns
    -------
    keys : list[tuple]
        Segment keys in the order they are assembled.
    I_sum : ndarray, shape (n_segments, n_times)
        Summed current per segment.
        Units are mA if to_mA=True, otherwise nA.
    """
    currents_by_segment = defaultdict(list)

    for src, current in zip(sources_intr, I_intr):
        key = (
            src.gid,
            src.section,
            src.segment_index,
        )
        currents_by_segment[key].append(current)

    keys = list(currents_by_segment.keys())

    I_sum = np.asarray([
        np.sum(currents_by_segment[key], axis=0)
        for key in keys
    ])

    if to_mA:
        I_sum = I_sum / 1e6

    return keys, I_sum


def collect_intrinsic_sources( # capacitive + ionic
    net,
    trial_idx: int = 0,
    cell_types: Sequence[str] = ("L2_pyramidal", "L5_pyramidal"),
    channels: Sequence[str] | None = None,
    cap_current_sign: float = +1.0, # only applied to agg_i_cap
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
                #def _channel_data_for_gid(net, trial_idx: int, gid: int, channel: str):
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

    seg_ctr = np.zeros((nseg, 3), dtype=float) # for segment_center coordinates
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

    # this template_cells construction differs from the one in the transmembrane current PR.
    # Here we don't go through NEURON and build()
    # Going through NEURON is not necessary here as all we need are 
    # sec_start, sec_end, nseg, and L; no need for seg.area()
    template_cells = {
        cell_type: net.cell_types[cell_type]["cell_object"]
        for cell_type in cell_types
    }

    T = np.zeros((len(array.positions), len(sources)), dtype=float)

    # Per-section transfer-resistance cache.
    #
    # Many current sources share the same (cell_type, gid, section) and differ only in
    # segment_index (e.g., for nseg=3, three intrinsic-current sources coexist on
    # one section). Their global section endpoints, nseg, and L are identical, so
    # the full (n_contacts, nseg) transfer-resistance block is computed once per
    # (cell_type, gid, section) and column-sliced per source via segment_index.
    # Note: segment_index is intentionally omitted from the cache key. Cached array has shape (n_contacts, nseg).
    cache = {}

    for col, src in enumerate(sources):
        key = (src.cell_type, src.gid, src.section) # e.g., ('L5_pyramidal', 170, 'soma')

        if key not in cache:
            sec_start, sec_end, nseg, L = _get_global_section_geometry(
                net, template_cells, src
            )

            xfers = []
            for pos in array.positions: # loop over electrode positions
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
    """Return LFP (micro V), shape (n_contacts, n_times).
    
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
    cap_current_sign: float = +1.0, # BE CAREFUL HERE!
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


###########
# FROM tm_currents_utils_forLFP.py (to be deleted once this module is mature and we have a better home for these)
# (slightly modified from Dylan's PR)
###########
def postproc_tm_currents(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """
    Function for processing transmembrane currents to recreate the dipole moment
    calculated from the axial currents in hnn_core. This can be done from either the
    total recorded transmembrane current, or from the constituent components.

    Note: isec (the transmembrane synaptic current) is part of the total transmembrane
    current, but is *not* aggregated with the other component channel currents
    in this function. This is due to the fact that isec contains *section-specific*
    currents (since synapses are placed at the section midpoint), as opposed to
    *segment-specfic* currents, which are required for this method of recreating
    the dipole.

    Parameters
    ----------
    net : Network object
        The network object containing the simulation data.
    trial : int
        The index of the trial to use
    cell_type : str
        The cell type to process
    scaling_factor : float
        The scaling factor to apply to the dipole
    #from_components : bool
    #    if True, use agg_i_mem to reproduce the dipole. if False, use the component
    #    currents for either L5_pyramidal or L2_pyramidal
    from_components : bool
        If True, reconstruct the dipole from the individual transmembrane
        current components for the selected cell type. If False, use
        agg_i_mem, the total aggregated transmembrane current.   

    Returns
    -------
    dipole : np.ndarray
        The reconstructed dipole moment from transmembrane currents.
    """

    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for each section from the template cell
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        # sec.n3d() returns the number of 3D points along a section; essentially len()
        # so "sec.n3d() - 1" is the index of the last 3D point
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_km",
                "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

    # loop through GIDs for the cell_type of interest
    for gid in net.gid_ranges[cell_type]:
        # get the updated soma position for this instantiation of the cell
        # index of the first cell: e.g., 170 for the first L5Pyr cell
        start_index = net.gid_ranges[cell_type][0]
        # get soma position from position dictionary, which uses its own indexing
        # that does not match the GID, hence the "- start_index"
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # create a dictionary of all channel data for the cell
        cell_channels = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }
        # AC: list(cell_channels.keys()) returns 'agg_i_mem']
        # AC: cell_channels['agg_i_mem'].keys() returns dict_keys(['soma', 'apical_trunk', 'apical_dendrite', 'basal_dendrite'])

        # get the cell sections to loop over
        # the key used shouldn't matter, but we don't want to hard code it since
        # we can pass different channels to this function, so we get it dynamically
        first_key = list(cell_channels.keys())[0]
        cell_sections = list(cell_channels[first_key].keys())

        for sec_name in cell_sections:
            # offset the start/end positions by the realized soma position for this
            # cell instantiation
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # get the normalized segment positions along the cell section
            nseg = len(cell_channels[first_key][sec_name])
            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(
                seg_positions,
                cell_channels[first_key][sec_name].keys(),
            ):
                # convert the normalized position to the absolute position
                # via linear interpolation
                abs_pos = start + pos * (end - start)
                # simplification: we are using the z position only here we only
                # need the vertical component of the dipole momen
                # we do *not* need to do geometric projection (via cos_theta)
                # as we do for the dipole calculation from axial currents
                z_i = abs_pos[2]

                # sum all currents for this segment
                I_t = np.zeros_like(
                    np.array(cell_channels[first_key][sec_name][seg_key])
                )
                for ch in all_tm_channels:
                    # get channel data
                    vec = np.array(cell_channels[ch][sec_name][seg_key])

                    # get segment area and convert from µm^2 to cm^2
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8  # cm^2

                    if ch == "agg_i_mem":
                        # agg_i_mem is not recorded continuously as a density; it is
                        # recorded after each timestep. Ergo, the units conversion
                        # here is not necessary as the units are already in nA
                        #
                        # multiplying the contribution by zi in um will give us fAm,
                        # so we will later need to divide by 1e6 to convert to nAm
                        I_abs = vec
                    # convert densities (mA/cm^2) to absolute currents (mA)]
                    else:
                        I_abs = vec * area_cm2  # keep as mA

                        # [WIP]
                        # Should I flip sign for the capacitive currents? I *think*
                        # so, but I haven't found direct confirmation of this ...
                        #
                        # I ideally would want to test the sign flip empirically,
                        # and confirm that we can reproduce i_mem by summing up all
                        # of its constituent components. However, we can't get the
                        # per-segment synaptic currents since we model them as point
                        # processes at the midpoint of the section (not the segment);
                        #
                        # Ergo, we are missing the synaptic piece of the total
                        # transmembrane current needed to reproduce i_mem exactly
                        #
                        # Note: isec is the *per-synapse* current, and not the
                        # *per-segment* current that we need
                        if ch == "agg_i_cap":
                            I_abs = I_abs * -1  # flip sign (?)
                        # [end WIP]

                    I_t += I_abs

                # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
                # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
                # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
                contrib = I_t * z_i

                # for agg_i_mem, divide by 1e6 to convert fAm to nAm
                if not from_components:
                    contrib = contrib / 1e6 * scaling_factor
                else:
                    contrib = contrib * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole

def postproc_soma_dipole_AC(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """ """

    # this function will only handle the "soma", as it's composed of exactly one
    # segment where pos = 0.5
    sec_name = "soma"
    seg_key = "seg_1"
    pos = 0.5

    # load custom mechanisms
    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for the soma
    rel_endpoints = {}
    sec = template_cell._nrn_sections["soma"]
    start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
    end = np.array(
        [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
    )
    rel_endpoints[sec_name] = (start, end)

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_km",
                "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )
    
    I_t_over_gid = None
    I_syn_over_gid = None
    I_cap_intr_over_gid = None #capacitive + intrinsic currents
    # loop through GIDs for the cell_type of interest
    for gid in net.gid_ranges[cell_type]:
        # get the updated soma position for this instantiation of the cell
        # index of the first cell: e.g., 170 for the first L5Pyr cell
        start_index = net.gid_ranges[cell_type][0]
        # get soma position from position dictionary, which uses its own indexing
        # that does not match the GID, hence the "- start_index"
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # create a dictionary of all channel data for the cell
        cell_channels = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # get the cell sections to loop over
        # the key used shouldn't matter, but we don't want to hard code it since
        # we can pass different channels to this function, so we get it dynamically
        first_key = list(cell_channels.keys())[0]

        start_rel, end_rel = rel_endpoints[sec_name]
        start = start_rel + soma_pos
        end = end_rel + soma_pos

        abs_pos = start + pos * (end - start)
        z_i = abs_pos[2]

        # sum all currents for this segment
        I_t = np.zeros_like(
            np.array(cell_channels[first_key][sec_name][seg_key]),
        )

        for ch in all_tm_channels:
            # get channel data
            vec = np.array(cell_channels[ch][sec_name][seg_key])

            # get segment area and convert from µm^2 to cm^2
            seg = template_cell._nrn_sections[sec_name](pos)
            area_um2 = seg.area()  # µm^2
            area_cm2 = area_um2 * 1e-8  # cm^2

            if ch == "agg_i_mem":
                # agg_i_mem is not recorded continuously as a density; it is
                # recorded after each timestep. Ergo, the units conversion
                # here is not necessary as the units are already in nA
                #
                # multiplying the contribution by zi in um will give us fAm,
                # so we will later need to divide by 1e6 to convert to nAm
                I_abs = vec # in nA
            # convert densities (mA/cm^2) to absolute currents (mA)]
            else:
                #sign = -1 if ch == "agg_i_cap" else 1 # flip sign for capacitive current to match convention of inward current as positive
                # sign = 1 # keep original sign for all currents for now
                I_abs = vec * area_cm2  # keep as mA
                if I_cap_intr_over_gid is None:
                    I_cap_intr_over_gid = np.zeros_like(
                        np.array(cell_channels[first_key][sec_name][seg_key]),
                    )
                I_cap_intr_over_gid += I_abs * 1e6 # in nA

            I_t += I_abs # AC: nA for agg_i_mem, mA for the rest

        # around for different structure for isec when reconstructing from components
        if from_components:
            soma_isec = net.cell_response.isec[trial][gid].get(sec_name, {})
            for syn_key in soma_isec:
                # isec is measured in nA, so we need to divide by 1e6 to
                # convert nA to mA before we add to I_t
                I_t += np.array(soma_isec[syn_key]) / 1e6
                if I_syn_over_gid is None:
                    I_syn_over_gid = np.zeros_like(
                        np.array(cell_channels[first_key][sec_name][seg_key]),
                    )
                I_syn_over_gid += np.array(soma_isec[syn_key]) # in nA
        
        if I_t_over_gid is None:
            I_t_over_gid = np.zeros_like(
                np.array(cell_channels[first_key][sec_name][seg_key]),
            )
        
        if from_components:
            I_t_over_gid += I_t * 1e6 # convert mA to nA for I_t_over_gid
        elif not from_components:
            I_t_over_gid += I_t # already in nA

        # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
        # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
        # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
        contrib = I_t * z_i

        # for agg_i_mem, divide by 1e6 to convert fAm to nAm
        if not from_components:
            contrib = contrib / 1e6 * scaling_factor
        else:
            contrib = contrib * scaling_factor

        if dipole is None:
            dipole = contrib.copy()
        else:
            dipole += contrib

    return dipole, I_t_over_gid, I_syn_over_gid, I_cap_intr_over_gid


def check_rmse_and_residuals(
    net,
    trial=0,
    cell_type="L5_pyramidal",
):
    times = net.cell_response.times

    # get dipole from i_mem
    dpl_imem = postproc_soma_dipol_AC(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=False,
    )
    # get the dipole reconstructed from the constituent components
    dpl_comp = postproc_soma_dipole_AC(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=True,
    )

    if dpl_imem is None or dpl_comp is None:
        raise ValueError(
            "Dipole data could not be computed",
        )

    # calculate residual and rmse
    residual = dpl_imem - dpl_comp
    rmse = np.sqrt(np.mean(residual**2))

    # normalize by the peak-to-peak range of agg_i_mem (our "ground truth")
    dpl_range = np.max(dpl_imem) - np.min(dpl_imem)
    nrmse_pct = (rmse / dpl_range) * 100 if dpl_range != 0 else 0

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    # overlay plot
    ax[0].plot(
        times[1:],
        dpl_imem[1:],
        label="From agg_i_mem",
        alpha=0.8,
    )
    ax[0].plot(
        times[1:],
        dpl_comp[1:],
        label="From components",
        linestyle="--",
        alpha=0.8,
    )

    # text box for rmse
    error_text = f" RMSE: {rmse:.2f}\nNRMSE:  {nrmse_pct:.2f}%"
    ax[0].text(
        x=0.05,
        y=0.95,
        s=error_text,
        transform=ax[0].transAxes,
        verticalalignment="top",
        fontsize=12,
        fontfamily="Menlo",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.2,
        ),
    )

    ax[0].set_title(
        f"Computed Dipole Comparison for {cell_type.replace('_', ' ').title()} Soma",
    )
    ax[0].set_ylabel("nAm")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right")

    ymin, ymax = ax[0].get_ylim()
    ylim = (max(abs(ymin), abs(ymax))) * 1.05
    ax[0].set_ylim(-ylim, ylim)

    # residual dipole plot
    ax[1].plot(
        times[1:],
        residual[1:],
        color="red",
        label="(i_mem - reconstructed), scaled",
    )
    ax[1].set_title(
        "Residual (Error)",
    )
    ax[1].set_ylabel("nAm")
    ax[1].set_ylim(-ylim, ylim)
    ax[1].grid(True, alpha=0.3)
    # ax[1].legend()

    rightax = ax[1].twinx()
    rightax.plot(
        times[1:],
        residual[1:],
        alpha=0.2,
        label="(i_mem - reconstructed), zoomed",
    )
    ymin, ymax = rightax.get_ylim()
    ylim = max(abs(ymin), abs(ymax))
    rightax.set_ylim(-ylim, ylim)

    lines1, labels1 = ax[1].get_legend_handles_labels()
    lines2, labels2 = rightax.get_legend_handles_labels()

    ax[1].legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()






###########
# UPDATED LFP/CSD PLOTTING FUNCTIONS
###########


def plot_stacked_traces(
    ax,
    t,
    data,
    depths,
    color="k",
    lw=0.8,
    alpha=1.0,
    scale=1.0,
    baseline_samps=None, # use a value for experimental data
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
