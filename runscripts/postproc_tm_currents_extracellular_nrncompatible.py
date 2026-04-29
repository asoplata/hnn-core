"""Post hoc extracellular decomposition from HNN transmembrane-current recordings.

Main difference from postproc_tm_currents_extracellular_def.py
----------------------------------------------------------------
This version is the one that explicitly rebuilds HNN template cells and uses
NEURON section objects for morphology-dependent quantities. In particular, it
uses NEURON sections to obtain:

- segment membrane area through ``seg.area()``;
- the number of segments in a section through ``nrn_sec.nseg``;
- section length through ``nrn_sec.L``.

The companion file instead relies more directly on the cell object already stored
inside ``net.cell_types[cell_type]["cell_object"]`` and on HNN section metadata
for quantities such as ``sec.nseg``, ``sec.L``, and an approximate cylindrical
area calculation.

Important: this file does *not* read live NEURON recording vectors or pointer
objects for the currents. The current time series still come from the HNN
recording containers:

- ``net.cell_response.transmembrane_currents`` for intrinsic/mechanism currents;
- ``net.cell_response.isec`` for synaptic currents.

So, the role of NEURON objects here is morphology/geometry support for the
post-processing reconstruction, not live current extraction.

What the module reconstructs
----------------------------
This module is designed for exploratory work on top of the transmembrane-current
recording branch discussed in HNN PR #1212. It reconstructs approximate LFP/CSD
contributions from:

1. intrinsic / transmembrane current channels recorded per segment;
2. synaptic currents recorded at section midpoints, using a midpoint-segment
   approximation.

Assumptions / limitations
-------------------------
- This code assumes the PR branch stores recordings under
  ``net.cell_response.transmembrane_currents`` and ``net.cell_response.isec``.
- Intrinsic/mechanism currents are assumed to be recorded as current densities
  (mA / cm^2) except for ``agg_i_mem``, which is assumed to already be in nA.
- Synaptic currents are approximated as being located on the midpoint segment of
  the section. For even ``nseg``, the current is split equally over the two
  central segments by default.
- The sign convention for capacitive current can differ depending on exactly how
  you want to compare it to aggregate membrane current. A configurable sign flip
  is provided, but should be validated against the branch behavior.

This is a standalone analysis helper, not production HNN code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt

from hnn_core.cells_default import pyramidal
from hnn_core.extracellular import calculate_csd2d
from hnn_core.network_builder import load_custom_mechanisms

from numpy.linalg import norm

# Channels exposed by the transmembrane-current branch.
# These names correspond to the Jones 2009 intrinsic/mechanism currents
# currently exposed by the recording PR.
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
    """Metadata container for one extracellular current source.

    Each row of the current matrix has one matching ``SourceInfo`` object. For
    intrinsic/mechanism currents, one source corresponds to one recorded segment.
    For synaptic currents, one source corresponds to the section midpoint
    segment, or to one of the two central segments if ``midpoint_mode`` splits an
    even-``nseg`` section.

    ``area_um2`` is used to convert current densities to absolute currents.
    ``weight_fraction`` is used only for synaptic midpoint splitting.
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


def _default_template_builder(cell_type: str):
    """Build a template cell for geometry/area lookup.

    Currently supports HNN pyramidal cell types. Basket cells can be added later
    with a custom builder map if needed.
    """
    load_custom_mechanisms()
    if cell_type in {"L2_pyramidal", "L5_pyramidal"}:
        cell = pyramidal(cell_name=cell_type)
        cell.build(sec_name_apical="apical_trunk")
        return cell
    raise NotImplementedError(
        f"No default template builder for {cell_type!r}. "
        "Pass a custom template_builders mapping if you want to support it."
    )


def _get_template_cells(cell_types):
    """Build template cells for all requested cell types.

    This function intentionally rebuilds template cells rather than pulling the
    cell objects from ``net.cell_types``. That makes this version useful in a
    saved/postprocessed workflow where the full live network cell objects may not
    be available, but the standard HNN morphology can still be reconstructed.
    """
    return {
        cell_type: _default_template_builder(cell_type)
        for cell_type in cell_types
    }


def _get_gid_soma_pos(net, cell_type: str, gid: int) -> np.ndarray:
    """Return the 3D soma position for a gid.

    ``net.gid_ranges[cell_type]`` tells us where the gids for a cell type start.
    ``net.pos_dict[cell_type]`` stores positions relative to that start index.
    The soma position is used as the global offset for the template morphology.
    """
    start_gid = net.gid_ranges[cell_type][0]
    return np.asarray(net.pos_dict[cell_type][gid - start_gid], dtype=float)


def _segment_xs_for_section(nseg: int) -> np.ndarray:
    """Return normalized NEURON x locations for segment centers.

    For a section with ``nseg`` segments, NEURON places segment centers evenly
    between 0 and 1, excluding the endpoints. For example, ``nseg=5`` gives
    x locations ``[0.1, 0.3, 0.5, 0.7, 0.9]``.

    These x locations are used for segment area lookup and for assigning
    recorded currents to segment-level extracellular sources.
    """
    return np.asarray([(i - 0.5) / nseg for i in range(1, nseg + 1)], dtype=float)


def _get_nrn_section(template_cell, section: str):
    """Return the NEURON section object for a named HNN section.

    This is the main helper that makes this file the ``NRN-object`` version. It
    first looks for the section in ``template_cell._nrn_sections``. If that is
    not available, it falls back to the HNN section wrapper in
    ``template_cell.sections[section]`` and then uses its ``._sec`` attribute.

    The returned object is used for NEURON-native quantities such as
    ``nrn_sec.nseg``, ``nrn_sec.L``, and ``nrn_sec(x).area()``.
    """
    if hasattr(template_cell, "_nrn_sections") and section in template_cell._nrn_sections:
        return template_cell._nrn_sections[section]

    if hasattr(template_cell, "sections") and section in template_cell.sections:
        sec = template_cell.sections[section]
        if hasattr(sec, "_sec"):
            return sec._sec

    raise KeyError(
        f"NEURON section {section!r} not found. "
        f"_nrn_sections keys: {list(getattr(template_cell, '_nrn_sections', {}).keys())}; "
        f"sections keys: {list(getattr(template_cell, 'sections', {}).keys())}"
    )


def _segment_area_um2(template_cell, section: str, segment_x: float) -> float:
    """Return the membrane area of one segment in um^2.

    Unlike the postproc_tm_currents_extracellular_def.py, this version asks NEURON directly for segment
    area using ``nrn_sec(segment_x).area()``. 
    """
    nrn_sec = _get_nrn_section(template_cell, section)
    seg = nrn_sec(float(segment_x))
    return float(seg.area())

def _density_to_nA(current_density, area_um2: float) -> np.ndarray:
    """Convert current density mA/cm^2 -> nA using segment area in um^2."""
    return np.asarray(current_density, dtype=float) * area_um2 * 1e-2


def _ensure_2d_timeseries(matrix_like: list[np.ndarray]) -> np.ndarray:
    """Convert a list of 1D time series to a 2D source-by-time matrix.

    Each input array is assumed to have shape ``(n_times,)``. The output has
    shape ``(n_sources, n_times)``. If there are no sources, return an empty
    array with shape ``(0, 0)``.
    """
    if len(matrix_like) == 0:
        return np.empty((0, 0), dtype=float)
    return np.vstack([np.asarray(x, dtype=float)[None, :] for x in matrix_like])


def _pick_midpoint_segments(nseg: int, mode: str = "split_even"):
    """Choose which segment(s) represent a section-midpoint synapse.

    Synaptic currents in ``isec`` are organized by section/receptor rather than
    by exact segment. This helper maps each section-level synaptic current to a
    segment-level source for the extracellular reconstruction.

    For odd ``nseg``, the single central segment is used. For even ``nseg``, the
    default is to split the current equally over the two central segments, but
    ``nearest_lower`` and ``nearest_upper`` are available for debugging.

    Returns
    -------
    list of tuple
        ``[(segment_index, weight_fraction), ...]``.
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
    """Return recorded intrinsic/mechanism current data for one gid/channel.

    This reads from ``net.cell_response.transmembrane_currents``. It does not
    read from live NEURON pointer vectors.
    """
    return net.cell_response.transmembrane_currents[channel][trial_idx][gid]


def _synaptic_data_for_gid(net, trial_idx: int, gid: int):
    """Return recorded synaptic current data for one gid.

    This reads from ``net.cell_response.isec``. The data are later mapped onto
    midpoint segment sources using the template morphology.
    """
    return net.cell_response.isec[trial_idx][gid]


def collect_intrinsic_sources(
    net,
    trial_idx: int = 0,
    cell_types: Sequence[str] = ("L2_pyramidal", "L5_pyramidal"),
    channels: Sequence[str] | None = None,
    cap_current_sign: float = -1.0,
):
    """Collect intrinsic/mechanism current sources as post hoc source vectors.

    More specifically, this function creates:

    1. a list of ``SourceInfo`` objects, one for each segment-level current
       source that will be used in the extracellular reconstruction;
    2. a current matrix with one row per source and one column per time point.

    Current values are read from
    ``net.cell_response.transmembrane_currents[channel][trial_idx][gid]``.
    NEURON objects are used only to obtain the segment area needed to convert
    density currents from ``mA/cm^2`` to ``nA``.

    ``agg_i_mem`` is assumed to already be an absolute current in nA. All other
    supported intrinsic channels are treated as current densities.

    Returns
    -------
    sources : list[SourceInfo]
        Metadata for every segment-level source.
    current_matrix_nA : ndarray, shape (n_sources, n_times)
        Absolute current time series for all sources.
    """
    channels = tuple(channels or sorted(SUPPORTED_INTRINSIC_CHANNELS))
    bad = sorted(set(channels) - SUPPORTED_INTRINSIC_CHANNELS)
    if bad:
        raise ValueError(f"Unsupported intrinsic channels: {bad}")

    template_cells = _get_template_cells(cell_types)
    #template_cells = _get_template_cells(cell_types, template_builders)
    #template_cells = net.cell_types[cell_type]["cell_object"]
    sources: list[SourceInfo] = []
    currents_nA = []

    for cell_type in cell_types:
        #template_cell = net.cell_types[cell_type]["cell_object"]
        #template_cell = template_cells[cell_type]
        #template_cell = _make_template_cell(cell_type)
        template_cell = template_cells[cell_type]

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
    """Collect synaptic current sources using a midpoint-segment approximation.

    Synaptic currents are read from
    ``net.cell_response.isec[trial_idx][gid]``. In the current HNN recording
    structure, these currents are organized by section and synapse/receptor name,
    not by exact segment. This function therefore assigns each recorded synaptic
    current to the section midpoint.

    For even ``nseg``, the current is split over the two central segments by
    default. This creates a segment-level proxy that can be multiplied by the
    same transfer-resistance matrix used for intrinsic segment currents.

    NEURON objects are used here to get ``nseg`` and segment area for the rebuilt
    template morphology.
    """
    
    sources: list[SourceInfo] = []
    currents_nA = []

    template_cells = _get_template_cells(cell_types)

    for cell_type in cell_types:
        #template_cell = net.cell_types[cell_type]["cell_object"]
        template_cell = template_cells[cell_type]
        for gid in net.gid_ranges[cell_type]:
            syn_data = _synaptic_data_for_gid(net, trial_idx, gid)
            for section, syn_dict in syn_data.items():
                nrn_sec = _get_nrn_section(template_cell, section)
                nseg = int(nrn_sec.nseg)
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
        # Point-source approximation: each segment is treated as a point source
        # at its center.
        dis = norm(np.tile(electrode_pos, (nseg, 1)) - seg_ctr, axis=1)
        dis = np.maximum(dis, min_distance)
        phi = 1.0 / dis

    elif method == "lsa":
        # Line-source approximation: each segment is treated as a short line
        # source along the section.
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


def _get_global_section_geometry(net, template_cells, src):
    """Return global section endpoints and NEURON geometry for one source.

    The section endpoints come from the HNN template morphology
    ``sections[section]._end_pts`` and are shifted by the gid-specific soma
    position from ``net.pos_dict``. The number of segments and section length are
    taken from the corresponding NEURON section object.
    """
    soma_pos = _get_gid_soma_pos(net, src.cell_type, src.gid)

    hnn_sec = template_cells[src.cell_type].sections[src.section]
    nrn_sec = _get_nrn_section(template_cells[src.cell_type], src.section)

    end_pts = np.asarray(hnn_sec._end_pts, dtype=float)
    sec_start = end_pts[0] + soma_pos
    sec_end = end_pts[1] + soma_pos

    return sec_start, sec_end, int(nrn_sec.nseg), float(nrn_sec.L)


def build_transfer_resistance_matrix_for_sources(
    net,
    sources: Sequence[SourceInfo],
    array_name: str = "probe1",
):
    """Build the electrode-by-source transfer-resistance matrix.

    The output ``T`` has shape ``(n_contacts, n_sources)``. Each column contains
    the transfer resistance between one source segment and all electrode contacts.

    This is the post hoc analogue of HNN's on-the-fly extracellular calculation:
    current sources are first organized into a source-by-time current matrix, and
    then the extracellular potential is reconstructed with ``lfp = T @ I_nA``.

    To avoid recomputing the same section/electrode geometry repeatedly, transfer
    values are cached by ``(cell_type, gid, section)``. The cached matrix has
    shape ``(n_contacts, nseg)``; for each source, the function selects the
    column corresponding to ``src.segment_index``.
    """
    array = net.rec_arrays[array_name]
    cell_types = sorted({src.cell_type for src in sources})

    template_cells = _get_template_cells(cell_types)

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
    """Return reconstructed LFP in uV, shape ``(n_contacts, n_times)``.

    This multiplies the transfer-resistance matrix ``T`` with the current matrix
    ``I_nA``:

    ``lfp = T @ I_nA``

    This mirrors the logic of HNN's recorded array calculation, where gathered
    membrane currents are multiplied by a transfer-resistance matrix. Here the
    same idea is applied after the simulation to selected current sources.
    """
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
    """High-level wrapper for intrinsic-current LFP reconstruction.

    This function goes from ``net`` and a current/source selection to the
    reconstructed LFP. It returns the LFP, the source metadata, the transfer
    matrix, and the current matrix so that the same objects can be reused for
    filtering, plotting, or CSD reconstruction.
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
    """High-level wrapper for synaptic-current LFP reconstruction.

    This first builds segment-level proxy sources from ``isec`` using the
    midpoint-segment approximation, then builds the transfer matrix, then returns
    ``lfp = T @ I_nA`` together with the intermediate source/current objects.
    """
    sources, I_nA = collect_synaptic_sources(
        net,
        trial_idx=trial_idx,
        cell_types=cell_types,
        midpoint_mode=midpoint_mode,
    )
    T = build_transfer_resistance_matrix_for_sources(net, sources, array_name=array_name)
    lfp = reconstruct_lfp_from_sources(T, I_nA)
    return lfp, sources, T, I_nA


def reconstruct_csd(net, lfp: np.ndarray, array_name: str = "probe1") -> np.ndarray:
    """Compute laminar CSD from a reconstructed LFP array.

    The inter-contact spacing is inferred from the z coordinates of
    ``net.rec_arrays[array_name].positions``. This assumes a regularly spaced
    laminar probe.
    """
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
    """Filter an existing source/current collection without rerunning collection.

    This is useful after a broad reconstruction. For example, you can collect all
    intrinsic sources once and then isolate one channel, one cell type, one
    section, or a subset of gids before recomputing ``lfp = T_filtered @ I``.

    Note: in this version, ``syn_names`` requires exact matches to
    ``src.syn_name``.
    """
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
    if syn_names is not None: # THIS SHOULD STILL BE UPDATED ACCORDING TO THE COMPANION FILE. THIS IS JUST A PLACEHOLDER FOR NOW.
        syn_names = set(syn_names)
        keep &= np.array([src.syn_name in syn_names for src in sources])

    filt_sources = [src for src, k in zip(sources, keep) if k]
    filt_I = current_matrix_nA[keep]
    return filt_sources, filt_I


def summarize_sources(sources: Sequence[SourceInfo]):
    """Count sources grouped by source kind, cell type, and label.

    This answers questions such as: how many intrinsic vs synaptic sources were
    included, how many came from each cell type, and how many correspond to a
    particular current label such as ``ina_hh2`` or ``isec``.
    """
    summary = {}
    for src in sources:
        key = (src.kind, src.cell_type, src.label)
        summary[key] = summary.get(key, 0) + 1
    return summary


import matplotlib.pyplot as plt
from hnn_core.viz import plot_laminar_lfp, plot_laminar_csd

def plot_lfp_and_csd(times, lfp, csd, contact_positions=None,
                     titles=("LFP", "CSD")):
    """Plot reconstructed LFP and CSD using HNN's laminar plotting helpers.

    ``contact_positions`` can be passed as the recording-array positions so that
    the z coordinate of each contact is used as the depth label.
    """
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

