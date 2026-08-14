import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_dipole_lfp_csd as tme
#import tm_currents_utils_forLFP as tme_lfp
from hnn_core.extracellular import calculate_csd2d
import Plotting_tools as plott
import matplotlib.pyplot as plt

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model

l5_component_channels = [
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

# load from here:
#with open('runscripts/data/sim_results_dt0025.pkl', 'rb') as f:
#with open('runscripts/data/sim_results_dt000625.pkl', 'rb') as f: #from -625 up
with open('runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc.pkl', 'rb') as f: #from -550 up
    results = pickle.load(f)
# to run simulation: use Undestand_CSD_contributions_v3.py

dpl = results['dpl']
net = results['net']
times = results['times']
depths = results['depths']

contact_labels = np.asarray(depths, dtype=int)
lfp_hnn = net.rec_arrays["probe1"].voltages[0]


#lfp = net.rec_arrays["probe1"].voltages[0]  # HNN's LFP; trial 0
contact_labels = np.asarray(depths, dtype=int)


step = 4
tme.downsample_currents(net, step=step, include_isec=True, include_vsec=True, include_ca=True)
times = times[::step]
lfp_hnn = lfp_hnn[:,::step]

times_ = times[1:]
lfp_hnn_ = lfp_hnn[:, 1:]

delta = np.median(np.diff(depths))
csd_from_lfp = calculate_csd2d(lfp_hnn, delta=delta)
csd_from_lfp_ = csd_from_lfp[:, 1:]

# plot of LFP and CSD (from hnn output) to have a visual reference
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8),
                        gridspec_kw={'height_ratios': [3, 3]})

# LFP panel
plott.plot_laminar_lfp_AC(
    times_, lfp_hnn_, contact_labels,
    ax=axs[0], scale=5.0, voltage_scalebar=50, show=False)
# CSD panel
vmax = np.abs(csd_from_lfp_).max()
plott.plot_laminar_csd_AC(
    times_, csd_from_lfp_, contact_labels,
    ax=axs[1], vmin=-vmax, vmax=vmax, sink="red", show=False)
axs[1].set_yticklabels([f"{d:g}" for d in contact_labels])  # true HNN z-coordinates
axs[1].set_ylabel("z (µm)")

fig.tight_layout()
plt.show()


# lfp from agg_i_mem
sources_agg, I_agg = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
    channels=["agg_i_mem"],
)

T_agg = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_agg,
    array_name="probe1",
)

lfp_agg_i_mem = tme.reconstruct_lfp_from_sources(
    T_agg,
    I_agg,
)

contact_labels = np.asarray(depths, dtype=int)

lfp_agg_i_mem_ = lfp_agg_i_mem[:, 1:]

'''
check: are lfp_hnn and lfp_agg_i_mem identical? Yes, they are.
fig, ax = plt.subplots(figsize=(6, 8))
# not tme anymore please
tme.plot_stacked_traces(ax, times_, lfp_hnn_, depths=contact_labels, color="k", scale=1.0)
tme.plot_stacked_traces(ax, times_, lfp_agg_i_mem_, depths=contact_labels, color="crimson", scale=1.0)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth (µm)")
ax.legend(handles=[Line2D([0], [0], color="k", label="lfp_hnn"),
                    Line2D([0], [0], color="crimson", label="lfp_agg_i_mem")])
plt.show()
# -> great! Still identical
'''

# CSD from sources:
# agg_i_mem CSD
B_agg, V_bin, z_center = tme.build_binning_matrix_for_sources(
    net,
    sources_agg,
    array_name="probe1"
)

csd_from_sources = tme.compute_csd_from_sources(
    B_agg,
    I_agg,
    V_bin
)

csd_from_sources_ = csd_from_sources[:, 1:]


#sigma = 0.3  # S/m (HNN default)
sigma = 0.3#1 # CHECK THIS, I think in HNN's CSD sigma is actually dropped.
# To compare csd_from_sources (μA/mm³) with HNN's output (μV/μm²):
csd_in_hnn_units = csd_from_sources / (sigma * 1e3)   # → μV/μm²
csd_in_hnn_units_ = csd_in_hnn_units[:, 1:]   

'''
vmax = max(np.abs(csd_from_lfp_).max(), np.abs(csd_in_hnn_units_).max())
vmax_from_lfp = np.abs(csd_from_lfp_).max()
vmax_from_hnn_units = np.abs(csd_in_hnn_units_).max()
vmax_from_sources = np.abs(csd_from_sources_).max()

fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

plott.plot_laminar_csd_AC(
    times_, csd_from_lfp_, contact_labels,
    ax=axes[0], vmin=-vmax_from_lfp, vmax=vmax_from_lfp,
    overlay_csd_traces=False, unit_csd="µV/µm²", sink="red", show=False)
axes[0].set_title("2nd derivative of LFP\n(calculate_csd2d)")

plott.plot_laminar_csd_AC(
    times_, csd_in_hnn_units_, contact_labels,
    ax=axes[1], vmin=-vmax_from_hnn_units, vmax=vmax_from_hnn_units,
    overlay_csd_traces=False, unit_csd="µV/µm²", sink="red", show=False)
axes[1].set_title("compute_csd_from_sources in hnn units")

plott.plot_laminar_csd_AC(
    times_, csd_from_sources_, contact_labels,
    ax=axes[2], vmin=-vmax_from_sources, vmax=vmax_from_sources,
    overlay_csd_traces=False, unit_csd="µA/mm³", sink="red", show=False)
axes[2].set_title("compute_csd_from_sources\n(agg_i_mem)")

for ax in axes:
    ax.set_yticklabels([f"{d:g}" for d in contact_labels])
    ax.set_ylabel("z (µm)")

plt.show()

# provisional zoom in

fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

plott.plot_laminar_csd_AC(
    times_, csd_from_lfp_, contact_labels,
    ax=axes[0], vmin=-vmax_from_lfp, vmax=vmax_from_lfp,
    overlay_csd_traces=False, unit_csd="µV/µm²", sink="red", show=False)
axes[0].set_title("2nd derivative of LFP\n(calculate_csd2d)")

plott.plot_laminar_csd_AC(
    times_, csd_in_hnn_units_, contact_labels,
    ax=axes[1], vmin=-vmax_from_hnn_units, vmax=vmax_from_hnn_units,
    overlay_csd_traces=False, unit_csd="µV/µm²", sink="red", show=False)
axes[1].set_title("compute_csd_from_sources in hnn units")

plott.plot_laminar_csd_AC(
    times_, csd_from_sources_, contact_labels,
    ax=axes[2], vmin=-vmax_from_sources, vmax=vmax_from_sources,
    overlay_csd_traces=False, unit_csd="µA/mm³", sink="red", show=False)
axes[2].set_title("compute_csd_from_sources\n(agg_i_mem)")

xlim = (15, 50)     # ms
ylim = (950, 1550)  # µm

for ax in axes:
    ax.set_yticklabels([f"{d:g}" for d in contact_labels])
    ax.set_ylabel("z (µm)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

plt.show()


plott.plot_lfp_morph_csd(
    times_,
    lfp_hnn_,#lfp_agg_i_mem_,
    csd_from_lfp_, #  csd_from_lfp = calculate_csd2d(lfp_hnn, delta=delta)
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    #vmin=-60,
    #vmax=60,
    figsize=(18, 6),
    overlay_csd_traces=True,
    unit_csd="µV/µm²",
    overlay_raster_on_csd=True,
    sink="red")
fig.suptitle("LFP/CSD from whole-network LFP (2nd derivative)")

plott.plot_lfp_morph_csd(
    times_,
    lfp_agg_i_mem_,
    csd_from_sources_, #  csd_from_lfp = calculate_csd2d(lfp_hnn, delta=delta)
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-60,
    vmax=60,
    figsize=(18, 6),
    overlay_csd_traces=True,
    unit_csd="µV/µm²",
    overlay_raster_on_csd=True,
    sink="red")
fig.suptitle("LFP/CSD from whole-network sources")
'''


###################
# CONTRIBUTIONS TO CSD FROM DIFFERENT SOURCES
###################
#
# From synaptic currents only
#
sources_syn, I_syn = tme.collect_synaptic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
)

# LFP from synaptic currents only
T_syn = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_syn,
    array_name="probe1",
)

lfp_syn = tme.reconstruct_lfp_from_sources(
    T_syn,
    I_syn,
)

lfp_syn_ = lfp_syn[:, 1:]

csd_syn_from_lfp = calculate_csd2d(lfp_syn, delta=delta)
csd_syn_from_lfp_ = csd_syn_from_lfp[:, 1:]

# CSD from sources (synaptic currents only)
B_syn, V_bin_syn, z_center_syn = tme.build_binning_matrix_for_sources(
    net,
    sources_syn,
    array_name="probe1"
)

csd_syn = tme.compute_csd_from_sources(
    B_syn,
    I_syn,
    V_bin_syn
)

csd_syn_ = csd_syn[:, 1:]

# --- CSD as 2nd derivative of the synaptic-current LFP --- This is not valid as not closed system
vmax_syn_from_lfp = np.max(np.abs(csd_syn_from_lfp_))
scale_csd_syn_from_lfp = 0.5 * np.diff(contact_labels)[0] / vmax_syn_from_lfp

fig1 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_,
    csd_syn_from_lfp_,
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-vmax_syn_from_lfp,
    vmax=vmax_syn_from_lfp,
    figsize=(18, 6),
    overlay_csd_traces=True,
    scale_csd_traces=scale_csd_syn_from_lfp,
    unit_csd="µV/µm²",
    overlay_raster_on_csd=True,
    sink="red")
fig1.suptitle("LFP/CSD from synaptic currents (2nd derivative of LFP)")

# --- CSD directly from synaptic current sources ---
vmax_syn = np.max(np.abs(csd_syn_))
scale_csd_syn = 0.5 * np.diff(contact_labels)[0] / vmax_syn

fig2 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_,
    csd_syn_,
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-60,#vmax_syn,
    vmax=60,#vmax_syn,
    figsize=(18, 6),
    overlay_csd_traces=True,
    scale_csd_traces=scale_csd_syn,
    unit_csd="µA/mm³",
    overlay_raster_on_csd=True,
    sink="red")
fig2.suptitle("LFP/CSD from synaptic currents (compute_csd_from_sources)")

#
# From capacitive currents only
#
sources_cap, I_cap = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal", "L5_pyramidal"],
    channels=["agg_i_cap"],
)

T_cap = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_cap,
    array_name="probe1",
)

lfp_cap = tme.reconstruct_lfp_from_sources(
    T_cap,
    I_cap,
)

csd_cap_from_lfp = calculate_csd2d(lfp_cap, delta=delta)
csd_cap_from_lfp_ = csd_cap_from_lfp[:, 1:]

# CSD from capacitive sources only (from sources, not from LFP)
B_cap, V_bin_cap, _ = tme.build_binning_matrix_for_sources(
    net, 
    sources_cap,
    array_name="probe1")

csd_cap = tme.compute_csd_from_sources(B_cap, I_cap, V_bin_cap)

T_cap = tme.build_transfer_resistance_matrix_for_sources(net, sources_cap, array_name="probe1")
lfp_cap = tme.reconstruct_lfp_from_sources(T_cap, I_cap)

lfp_cap_, csd_cap_ = lfp_cap[:, 1:], csd_cap[:, 1:]

'''
vmax1 = np.max(np.abs(csd_cap_from_lfp_))

fig1 = plott.plot_lfp_morph_csd(
    times_, lfp_cap_, csd_cap_from_lfp_, contact_labels, net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0, voltage_scalebar=200,
    vmin=-vmax1, vmax=vmax1, figsize=(18, 6),
    overlay_csd_traces=True, unit_csd="µV/µm²", sink="red",
    overlay_raster_on_csd=True,
)
fig1.suptitle("LFP/CSD from capacitive currents (2nd derivative of LFP)")

# --- CSD directly from capacitive current sources ---
vmax2 = np.max(np.abs(csd_cap_))

fig2 = plott.plot_lfp_morph_csd(
    times_, lfp_cap_, csd_cap_, contact_labels, net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0, voltage_scalebar=200,
    vmin=-10, vmax=10, figsize=(18, 6),
    overlay_csd_traces=True, unit_csd="µA/mm³", sink="red",
    overlay_raster_on_csd=True,
)
fig2.suptitle("LFP/CSD from capacitive currents (compute_csd_from_sources)")
'''


#
# From ionic currents only
#

sources_ionic, I_ionic = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
    channels=["ina_hh2", "ik_hh2", "ik_kca", "ik_km",
              "ica_ca", "ica_cat", "il_hh2", "i_ar"]
)

T_ionic = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_ionic,
    array_name="probe1",
)

lfp_ionic = tme.reconstruct_lfp_from_sources(
    T_ionic,
    I_ionic,
)

csd_ionic_from_lfp = calculate_csd2d(lfp_ionic, delta=delta)
csd_ionic_from_lfp_ = csd_ionic_from_lfp[:, 1:]

B_ionic, V_bin_ionic, _ = tme.build_binning_matrix_for_sources(net, sources_ionic, array_name="probe1")
csd_ionic = tme.compute_csd_from_sources(B_ionic, I_ionic, V_bin_ionic)

T_ionic = tme.build_transfer_resistance_matrix_for_sources(net, sources_ionic, array_name="probe1")
lfp_ionic = tme.reconstruct_lfp_from_sources(T_ionic, I_ionic)

lfp_ionic_, csd_ionic_ = lfp_ionic[:, 1:], csd_ionic[:, 1:]

'''
vmax1 = np.max(np.abs(csd_ionic_from_lfp_))

fig1 = plott.plot_lfp_morph_csd(
    times_, lfp_ionic_, csd_ionic_from_lfp_, contact_labels, net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0, voltage_scalebar=200,
    vmin=-vmax1, vmax=vmax1, figsize=(18, 6),
    overlay_csd_traces=True, unit_csd="µV/µm²", sink="red",
    overlay_raster_on_csd=True,
)
fig1.suptitle("LFP/CSD from ionic currents (2nd derivative of LFP)")

# --- CSD directly from ionic current sources ---
vmax2 = np.max(np.abs(csd_ionic_))

fig2 = plott.plot_lfp_morph_csd(
    times_, lfp_ionic_, csd_ionic_, contact_labels, net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0, voltage_scalebar=200,
    vmin=-vmax2, vmax=vmax2, figsize=(18, 6),
    overlay_csd_traces=True, unit_csd="µA/mm³", sink="red",
    overlay_raster_on_csd=True,
)
fig2.suptitle("LFP/CSD from ionic currents (compute_csd_from_sources)")
'''

'''
# asseses the difference between the two LFPs (from agg_i_mem and from components)
contact_number = 8  # choose one electrode/contact
plt.figure()
plt.plot(times[1:], lfp_agg_i_mem[contact_number,1:], label="agg_i_mem")
#plt.plot(times[1:], lfp_cap[contact_number,1:], "--", label="cap")  
#plt.plot(times[1:], lfp_ionic[contact_number,1:], "--", label="ionic")  
#plt.plot(times[1:], lfp_syn[contact_number,1:], ":", label="syn")  
plt.plot(times[1:], lfp_cap[contact_number,1:] + lfp_ionic[contact_number,1:] + lfp_syn[contact_number,1:], "--", label="cap + ionic + syn")
#plt.plot(times[1:], lfp_hnn[contact_number,1:], label="lfp_hnn") # to check with the ground truth
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("LFP (µV)")
plt.title('LFP at contact 10 (z=375 µm)')


I_agg_sum_seg = np.sum(I_agg, axis=0)
I_cap_ionic_syn_sum_seg = np.sum(I_cap, axis=0) + np.sum(I_ionic, axis=0) + np.sum(I_syn, axis=0)

plt.figure()
plt.plot(times[1:], I_agg_sum_seg[1:], label="agg_i_mem")
plt.plot(times[1:], I_cap_ionic_syn_sum_seg[1:], label="cap + ionic + syn")
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Current (nA)")
plt.title('Total membrane current summed over segments')
# Comment: For Kirchhoff's law, this should be zero.
'''




###############################
# GABA B currents only
###############################
sources_syn_gabab, I_syn_gabab = tme.filter_sources(sources_syn, I_syn, syn_names=["gabab"])

'''
#when I_syn_gabab is negative? (thus originating sinks)
# Boolean mask: which sources have at least one negative time point
goes_negative = np.any(I_syn_gabab < 0, axis=1)  # shape (n_sources,)
print(np.any(goes_negative))   # True or False
print(goes_negative.sum())     # how many sources go negative

# Extract those sources and their currents
sources_neg = [src for src, flag in zip(sources_syn_gabab, goes_negative) if flag]
I_neg = I_syn_gabab[goes_negative]  # shape (n_neg_sources, n_times)

# Inspect them
for src, i_trace in zip(sources_neg, I_neg):
    print(f"gid={src.gid}  cell={src.cell_type}  section={src.section}  "
          f"seg_x={src.segment_x:.2f}  syn={src.syn_name}  "
          f"min={i_trace.min():.4f} nA  fraction_negative={np.mean(i_trace < 0):.3f}")
'''
# all the currents in I_syn_gabab are positive, thus they all originate sources (current flowing out of the cell). This is consistent with GABA_B being inhibitory and thus hyperpolarizing (positive current out of the cell).

# LFP from GABA_B synaptic currents only
T_syn_gabab = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_syn_gabab,
    array_name="probe1",
)

lfp_syn_gabab = tme.reconstruct_lfp_from_sources(
    T_syn_gabab,
    I_syn_gabab,
)

csd_syn_gabab_from_lfp = calculate_csd2d(lfp_syn_gabab, delta=delta)
csd_syn_gabab_from_lfp_ = csd_syn_gabab_from_lfp[:, 1:]
  

B_syn_gabab, V_bin_syn_gabab, z_center_syn_gabab = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_gabab,
    array_name="probe1"
)

csd_syn_gabab = tme.compute_csd_from_sources(
    B_syn_gabab,
    I_syn_gabab,
    V_bin_syn_gabab
)

lfp_syn_gabab_ = lfp_syn_gabab[:, 1:]
csd_syn_gabab_ = csd_syn_gabab[:, 1:]

'''
vmax1 = np.max(np.abs(csd_syn_gabab_from_lfp_))

fig1 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_gabab_,
    csd_syn_gabab_from_lfp_,
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-vmax1,
    vmax=vmax1,
    figsize=(18, 6),
    overlay_csd_traces=True,
    unit_csd="µV/µm²",
    overlay_raster_on_csd=False,
    sink="red")
fig1.suptitle("LFP/CSD from GABA_B currents (2nd derivative of LFP)")
## THERE ARE ARTIFACTUAL SINKS! Explanation is in the second derivative of the LFP to compute the CSD. 

# Look below: 

lfp_syn_gabab_ = lfp_syn_gabab[:, 1:]

timepoint = 4000
lfp_syn_gabab_t = lfp_syn_gabab_[:, timepoint]

delta = np.median(np.diff(contact_labels))

# 1st derivative: central difference (interior points only, edges undefined)
d1 = np.full_like(lfp_syn_gabab_t, np.nan)
d1[1:-1] = (lfp_syn_gabab_t[2:] - lfp_syn_gabab_t[:-2]) / (contact_labels[2:] - contact_labels[:-2])

# 2nd derivative: 3-point central difference, same as in calculate_csd2d
# (y[i-1] - 2*y[i] + y[i+1]) / delta**2, but without the minus sign
d2 = np.full_like(lfp_syn_gabab_t, np.nan)
d2[1:-1] = np.diff(lfp_syn_gabab_t, n=2) / delta**2

csd_t = calculate_csd2d(lfp_syn_gabab_t[:, None], delta=delta)[:, 0]

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 12), constrained_layout=True)

axes[0].plot(contact_labels, lfp_syn_gabab_t, '*')
axes[0].set_ylabel("LFP")

axes[1].plot(contact_labels, d1, '*')
axes[1].set_ylabel("1st derivative")

axes[2].plot(contact_labels, d2, '*')
axes[2].set_ylabel("2nd derivative")

axes[3].plot(contact_labels, csd_t, '*')
axes[3].set_ylabel("CSD (calculate_csd2d)")
axes[3].set_xlabel("z (µm)")

plt.show()


# --- CSD directly from GABA_B current sources ---
vmax2 = np.max(np.abs(csd_syn_gabab_))

fig2 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_gabab_,
    csd_syn_gabab_,
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-10,#vmax2,
    vmax=10,#vmax2,
    figsize=(18, 6),
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    overlay_raster_on_csd=True,
    sink="red")
fig2.suptitle("LFP/CSD from GABA_B currents (compute_csd_from_sources)")
'''


#### GABA_A only
sources_syn_gabaa, I_syn_gabaa = tme.filter_sources(sources_syn, I_syn, syn_names=["gabaa"])

B_syn_gabaa, V_bin_syn_gabaa, z_center_syn_gabaa = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_gabaa,
    array_name="probe1"
)

csd_syn_gabaa = tme.compute_csd_from_sources(
    B_syn_gabaa,
    I_syn_gabaa,
    V_bin_syn_gabaa
)

# LFP from GABA_A synaptic currents only
T_syn_gabaa = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_syn_gabaa,
    array_name="probe1",
)

lfp_syn_gabaa = tme.reconstruct_lfp_from_sources(
    T_syn_gabaa,
    I_syn_gabaa,
)
  
lfp_syn_gabaa_ = lfp_syn_gabaa[:, 1:]
csd_syn_gabaa_ = csd_syn_gabaa[:, 1:]

csd_syn_gabaa_from_lfp = calculate_csd2d(lfp_syn_gabaa, delta=delta)
csd_syn_gabaa_from_lfp_ = csd_syn_gabaa_from_lfp[:, 1:]

'''
# --- CSD directly from GABA_A current sources ---
vmax2 = np.max(np.abs(csd_syn_gabaa_))

fig2 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_gabaa_,
    csd_syn_gabaa_,
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-10,#vmax2,
    vmax=10,#vmax2,
    figsize=(18, 6),
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    overlay_raster_on_csd=True,
    sink="red")
fig2.suptitle("LFP/CSD from GABA_A currents (compute_csd_from_sources)")
'''

## AMPA only
sources_syn_ampa, I_syn_ampa = tme.filter_sources(sources_syn, I_syn, syn_names=["ampa"])

B_syn_ampa, V_bin_syn_ampa, z_center_syn_ampa = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_ampa,
    array_name="probe1"
)

csd_syn_ampa = tme.compute_csd_from_sources(
    B_syn_ampa,
    I_syn_ampa,
    V_bin_syn_ampa
)

# LFP from AMPA synaptic currents only
T_syn_ampa = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_syn_ampa,
    array_name="probe1",
)

lfp_syn_ampa = tme.reconstruct_lfp_from_sources(
    T_syn_ampa,
    I_syn_ampa,
)
  
lfp_syn_ampa_ = lfp_syn_ampa[:, 1:]
csd_syn_ampa_ = csd_syn_ampa[:, 1:]

sources_syn_ampa, I_syn_ampa = tme.filter_sources(sources_syn, I_syn, syn_names=["ampa"])

B_syn_ampa, V_bin_syn_ampa, z_center_syn_ampa = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_ampa,
    array_name="probe1"
)

csd_syn_ampa = tme.compute_csd_from_sources(
    B_syn_ampa,
    I_syn_ampa,
    V_bin_syn_ampa
)

# LFP from AMPA synaptic currents only
T_syn_ampa = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_syn_ampa,
    array_name="probe1",
)

lfp_syn_ampa = tme.reconstruct_lfp_from_sources(
    T_syn_ampa,
    I_syn_ampa,
)

lfp_syn_ampa_ = lfp_syn_ampa[:, 1:]
csd_syn_ampa_ = csd_syn_ampa[:, 1:]

'''
vmax2 = np.max(np.abs(csd_syn_ampa_))

fig2 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_ampa_,
    csd_syn_ampa_,
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-vmax2,
    vmax=vmax2,
    figsize=(18, 6),
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    overlay_raster_on_csd=True,
    sink="red")
fig2.suptitle("LFP/CSD from AMPA currents (compute_csd_from_sources)")
'''

## NMDA only
sources_syn_nmda, I_syn_nmda = tme.filter_sources(sources_syn, I_syn, syn_names=["nmda"])

B_syn_nmda, V_bin_syn_nmda, z_center_syn_nmda = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_nmda,
    array_name="probe1"
)

csd_syn_nmda = tme.compute_csd_from_sources(
    B_syn_nmda,
    I_syn_nmda,
    V_bin_syn_nmda
)

# LFP from NMDA synaptic currents only
T_syn_nmda = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_syn_nmda,
    array_name="probe1",
)

lfp_syn_nmda = tme.reconstruct_lfp_from_sources(
    T_syn_nmda,
    I_syn_nmda,
)

lfp_syn_nmda_ = lfp_syn_nmda[:, 1:]
csd_syn_nmda_ = csd_syn_nmda[:, 1:]

# --- CSD directly from NMDA current sources ---
vmax2 = np.max(np.abs(csd_syn_nmda_))

fig2 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_nmda_,
    csd_syn_nmda_,
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-vmax2,
    vmax=vmax2,
    figsize=(18, 6),
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    overlay_raster_on_csd=True,
    sink="red")
fig2.suptitle("LFP/CSD from NMDA currents (compute_csd_from_sources)")


## CHECK: does the sum of the CSD of AMPA, NMDA, GABAA and GABAB currents equal the CSD from all synaptic currents?
total_csd_syn_ = csd_syn_ampa_ + csd_syn_nmda_ + csd_syn_gabaa_ + csd_syn_gabab_
#csd_syn_
csd_syn_residual_ = csd_syn_ - total_csd_syn_
vmax_check = np.max(np.abs(csd_syn_residual_)) # IT'S OKAY!


###########################
# THIS PART IS ALL ABOUT CALCULATING AND PLOTTING CSD FOR A SINGLE CELL
###########################

# 1) find L5 pyramidal cells that spiked in trial 0
l5_gids = set(net.gid_ranges["L5_pyramidal"])
spike_gids_trial0 = set(np.asarray(net.cell_response.spike_gids[0]).tolist())
spiking_l5_gids = sorted(l5_gids & spike_gids_trial0)

print(f"{len(spiking_l5_gids)} spiking L5 pyramidal cells (of {len(l5_gids)})")

# 2) pick one spiking cell
example_gid = 226#207 #spiking_l5_gids[0]
print(f"Selected gid = {example_gid}")

# 3) filter the capacitive sources for this one cell
sources_cap_cell, I_cap_cell = tme.filter_sources(
    sources_cap, I_cap, gid_subset=[example_gid]
)

# 4) bin -> CSD for this single cell
B_cap_cell, V_bin_cap_cell, _ = tme.build_binning_matrix_for_sources(
    net, sources_cap_cell, array_name="probe1"
)
csd_cap_cell = tme.compute_csd_from_sources(B_cap_cell, I_cap_cell, V_bin_cap_cell)
csd_cap_cell_ = csd_cap_cell[:, 1:]  # drop first sample, matches times_


# 5) plot morphology; CSD (top row) + membrane potential (below CSD only)
fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[3, 1])
ax_morph = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[0, 1])
ax_vm = fig.add_subplot(gs[1, 1])

plott.plot_cell_morphology_for_lfp_csd(
    net,
    contact_positions=contact_labels,
    cell_types=('L5_pyramidal',),
    gid=example_gid,
    ax=ax_morph,
    show=False,
)

plott.plot_laminar_csd_AC(
    times_,
    csd_cap_cell_,
    contact_labels,
    ax=ax,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
    show=False,
)

spike_times_all = np.asarray(net.cell_response.spike_times[0])
spike_gids_all = np.asarray(net.cell_response.spike_gids[0])
cell_spike_times = spike_times_all[spike_gids_all == example_gid]

for t in cell_spike_times:
    ax.axvline(t, color='k', linestyle='--', alpha=0.7, lw=1)

ax.set_title(f"Capacitive CSD — spiking L5 pyramidal cell (gid={example_gid}), "
             f"{len(cell_spike_times)} spikes")

# --- membrane potential panel ---
vsoma = np.asarray(net.cell_response.vsec[0][example_gid]["soma"])
vsoma_ = vsoma[1:]  # drop first sample, matches times_

ax_vm.plot(times_, vsoma_, color='k', lw=1)
for t in cell_spike_times:
    ax_vm.axvline(t, color='k', linestyle='--', alpha=0.7, lw=1)
ax_vm.set_xlabel("Time (ms)")
ax_vm.set_ylabel("Vm (mV)")
ax_vm.set_title("Somatic membrane potential")

plt.show()

# --- total, capacitive, ionic, synaptic CSD for this single cell ---
sources_ionic_cell, I_ionic_cell = tme.filter_sources(sources_ionic, I_ionic, gid_subset=[example_gid])
sources_syn_cell, I_syn_cell = tme.filter_sources(sources_syn, I_syn, gid_subset=[example_gid])

def csd_for_cell(sources, I, array_name="probe1"):
    B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sources, array_name=array_name)
    csd = tme.compute_csd_from_sources(B, I, V_bin)
    return csd[:, 1:]  # drop first sample, matches times_

sources_agg, I_agg = tme.collect_intrinsic_sources(
    net, trial_idx=0, cell_types=["L5_pyramidal"], channels=["agg_i_mem"],
)
sources_agg_cell, I_agg_cell = tme.filter_sources(sources_agg, I_agg, gid_subset=[example_gid])

csd_agg_cell_ = csd_for_cell(sources_agg_cell, I_agg_cell)
csd_ionic_cell_ = csd_for_cell(sources_ionic_cell, I_ionic_cell)
csd_syn_cell_ = csd_for_cell(sources_syn_cell, I_syn_cell)
# csd_cap_cell_ was already computed earlier

# slab (electrode-bin) boundaries used by build_binning_matrix_for_sources
z_edges = tme._z_edges_from_array(net, "probe1")

fig = plt.figure(figsize=(18, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 5, width_ratios=[1, 4, 4, 4, 4])
ax_morph = fig.add_subplot(gs[0, 0])
ax_total = fig.add_subplot(gs[0, 1])
ax_cap = fig.add_subplot(gs[0, 2])
ax_ionic = fig.add_subplot(gs[0, 3])
ax_syn = fig.add_subplot(gs[0, 4])

plott.plot_cell_morphology_for_lfp_csd(
    net, contact_positions=contact_labels,
    cell_types=('L5_pyramidal',), gid=example_gid,
    ax=ax_morph, show=False,
)

panels = [
    (ax_total, csd_agg_cell_, "Total"),
    (ax_cap, csd_cap_cell_, "Capacitive"),
    (ax_ionic, csd_ionic_cell_, "Ionic"),
    (ax_syn, csd_syn_cell_, "Synaptic"),
]

for i, (ax, csd_, title) in enumerate(panels):
    vmax = 1
    plott.plot_laminar_csd_AC(
        times_, csd_, contact_labels,
        ax=ax, vmin=-vmax, vmax=vmax,
        overlay_csd_traces=False, unit_csd="µA/mm³", sink="red", show=False,
        colorbar=(i == len(panels) - 1),
    )
    ax.set_title(title)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelleft=False)

for ax in (ax_morph, ax_total, ax_cap, ax_ionic, ax_syn):
    for z in z_edges:
        ax.axhline(z, color='red', alpha=0.2, lw=1)

fig.suptitle(f"gid={example_gid}", fontsize=11)
plt.show()



# CSD as traces:
def plot_csd_traces(ax, times_, csd_, contact_labels, title, scale_mult=-2.0):
    scale = scale_mult * np.diff(contact_labels)[0] / np.max(np.abs(csd_))
    tme.plot_stacked_traces(ax, times_, csd_, depths=contact_labels, color='k', scale=scale)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)

z_edges = tme._z_edges_from_array(net, "probe1")

fig = plt.figure(figsize=(20, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 5, width_ratios=[1, 4, 4, 4, 4])
ax_morph = fig.add_subplot(gs[0, 0])
ax_total = fig.add_subplot(gs[0, 1])
ax_cap = fig.add_subplot(gs[0, 2])
ax_ionic = fig.add_subplot(gs[0, 3])
ax_syn = fig.add_subplot(gs[0, 4])

plott.plot_cell_morphology_for_lfp_csd(
    net, contact_positions=contact_labels,
    cell_types=('L5_pyramidal',), gid=example_gid,
    ax=ax_morph, show=False,
)

plot_csd_traces(ax_total, times_, csd_agg_cell_, contact_labels, f"Total (agg_i_mem) CSD — gid={example_gid}")
plot_csd_traces(ax_cap, times_, csd_cap_cell_, contact_labels, f"Capacitive CSD — gid={example_gid}")
plot_csd_traces(ax_ionic, times_, csd_ionic_cell_, contact_labels, f"Ionic CSD — gid={example_gid}")
plot_csd_traces(ax_syn, times_, csd_syn_cell_, contact_labels, f"Synaptic CSD — gid={example_gid}")

for ax in (ax_morph, ax_total, ax_cap, ax_ionic, ax_syn):
    for z in z_edges:
        ax.axhline(z, color='red', alpha=0.2, lw=1)

fig.supxlabel(
    "CSD sign convention: positive = source (current leaving the cell), negative = sink (current entering the cell)",
    fontsize=10,
)

plt.show()

# Decide when to zoom in on the traces.
# ZOOMS in

sources_syn_exc_cell, I_syn_exc_cell = tme.filter_sources(sources_syn_cell, I_syn_cell, syn_names=["ampa", "nmda"])
sources_syn_inh_cell, I_syn_inh_cell = tme.filter_sources(sources_syn_cell, I_syn_cell, syn_names=["gabaa", "gabab"])

csd_syn_exc_cell_ = csd_for_cell(sources_syn_exc_cell, I_syn_exc_cell)
csd_syn_inh_cell_ = csd_for_cell(sources_syn_inh_cell, I_syn_inh_cell)


def make_csd_traces_figure(xlim, scale_mult=2.0):
    mask = (times_ >= xlim[0]) & (times_ <= xlim[1])

    # ONE shared scale, computed once from capacitive, ionic, excitatory, and inhibitory combined
    all_csd = [csd_cap_cell_, csd_ionic_cell_, csd_syn_exc_cell_, csd_syn_inh_cell_]
    vmax = max(np.max(np.abs(csd_[:, mask])) for csd_ in all_csd)
    scale = scale_mult * np.diff(contact_labels)[0] / vmax

    def plot_csd_traces(ax, csd_list, title):
        """csd_list: list of (csd_array, color) tuples, all sharing the same `scale`."""
        for csd_, color in csd_list:
            tme.plot_stacked_traces(ax, times_, csd_, depths=contact_labels, color=color, scale=-scale)
        ax.set_xlabel("Time (ms)")
        ax.set_title(title)
        ax.set_xlim(*xlim)

    z_edges = tme._z_edges_from_array(net, "probe1")

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 4, 4, 4])
    ax_morph = fig.add_subplot(gs[0, 0])
    ax_cap = fig.add_subplot(gs[0, 1])
    ax_ionic = fig.add_subplot(gs[0, 2])
    ax_syn = fig.add_subplot(gs[0, 3])

    plott.plot_cell_morphology_for_lfp_csd(
        net, contact_positions=contact_labels,
        cell_types=('L5_pyramidal',), gid=example_gid,
        ax=ax_morph, show=False,
    )

    plot_csd_traces(ax_cap, [(csd_cap_cell_, 'k')], f"Capacitive CSD — gid={example_gid}")
    plot_csd_traces(ax_ionic, [(csd_ionic_cell_, 'k')], f"Ionic CSD — gid={example_gid}")
    plot_csd_traces(ax_syn, [(csd_syn_exc_cell_, 'tab:red'), (csd_syn_inh_cell_, 'tab:blue')],
                     f"Synaptic CSD — gid={example_gid}")

    for ax in (ax_morph, ax_cap, ax_ionic, ax_syn):
        for z in z_edges:
            ax.axhline(z, color='red', alpha=0.2, lw=1)

    fig.legend(
        handles=[Line2D([0], [0], color='tab:red', label='Excitatory (AMPA+NMDA)'),
                 Line2D([0], [0], color='tab:blue', label='Inhibitory (GABA_A+GABA_B)')],
        loc='lower right', fontsize=9,
    )
    fig.supxlabel(
        "CSD sign convention: positive = source (current leaving the cell), negative = sink (current entering the cell)",
        fontsize=10,
    )
    fig.suptitle(f"Time window: {xlim[0]}-{xlim[1]} ms")

    plt.show()

# these are good for cell 226
make_csd_traces_figure((25, 50))
make_csd_traces_figure((60, 85))
make_csd_traces_figure((130, 155))



# plot of membrane potential and synaptic currents
def get_syn_current_sum(gid, section, keyword):
    """Sum all synaptic current traces at this section matching a (case-insensitive) keyword."""
    syn_dict = net.cell_response.isec[0][gid].get(section, {})
    total = np.zeros_like(net.cell_response.times)
    for k, v in syn_dict.items():
        if keyword.lower() in k.lower():
            total = total + np.asarray(v)
    return total

def zero_crossings(trace, times, threshold=0.0, direction='rising'):
    above = trace >= threshold
    if direction == 'rising':
        idx = np.where(above[1:] & ~above[:-1])[0]
    elif direction == 'falling':
        idx = np.where(~above[1:] & above[:-1])[0]
    else:
        raise ValueError("direction must be 'rising' or 'falling'")
    return times[idx + 1]

sections_to_plot = ['basal_2', 'apical_oblique', 'soma']

def make_syn_vm_figure(xlim=None):
    fig, axes = plt.subplots(len(sections_to_plot), 1, figsize=(9, 7),
                              sharex=True, constrained_layout=True)

    for i, sec in enumerate(sections_to_plot):
        ax_vm = axes[i]
        vm = np.asarray(net.cell_response.vsec[0][example_gid][sec])[1:]
        I_exc = (get_syn_current_sum(example_gid, sec, 'ampa')
                 + get_syn_current_sum(example_gid, sec, 'nmda'))[1:]
        I_inh = (get_syn_current_sum(example_gid, sec, 'gabaa')
                 + get_syn_current_sum(example_gid, sec, 'gabab'))[1:]

        ax_vm.axhline(0, color='gray', lw=0.5, alpha=0.5)
        for t_cross in zero_crossings(vm, times_, direction='rising'):
            ax_vm.axvline(t_cross, color='green', lw=0.8, linestyle='--', alpha=0.7, zorder=0)
        for t_cross in zero_crossings(vm, times_, direction='falling'):
            ax_vm.axvline(t_cross, color='purple', lw=0.8, linestyle=':', alpha=0.7, zorder=0)

        ax_vm.plot(times_, vm, color='k', lw=1)
        ax_vm.set_ylabel('$V_m$ (mV)')
        ax_vm.set_title(sec, fontsize=10)
        if xlim is not None:
            ax_vm.set_xlim(*xlim)

        ax_syn = ax_vm.twinx()
        ax_syn.plot(times_, I_exc, color='tab:red', lw=1)
        ax_syn.plot(times_, I_inh, color='tab:blue', lw=1)
        ax_syn.set_ylabel('$I_{syn}$ (nA)')

    axes[-1].set_xlabel('Time (ms)')
    fig.legend(
        handles=[Line2D([0], [0], color='k', label='$V_m$'),
                 Line2D([0], [0], color='tab:red', label='Excitatory (AMPA+NMDA)'),
                 Line2D([0], [0], color='tab:blue', label='Inhibitory (GABA_A+GABA_B)'),
                 Line2D([0], [0], color='green', lw=0.8, linestyle='--', label='$V_m=0$ rising'),
                 Line2D([0], [0], color='purple', lw=0.8, linestyle=':', label='$V_m=0$ falling')],
        loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, ncol=2,
    )
    fig.suptitle(f"$V_m$ and synaptic current — gid={example_gid}")
    plt.show()

make_syn_vm_figure()

# plot of membrane potential and synaptic conductance
def get_syn_current_sum(gid, section, keyword):
    """Sum all synaptic current traces at this section matching a (case-insensitive) keyword."""
    syn_dict = net.cell_response.isec[0][gid].get(section, {})
    total = np.zeros_like(net.cell_response.times)
    for k, v in syn_dict.items():
        if keyword.lower() in k.lower():
            total = total + np.asarray(v)
    return total

def synaptic_conductance(I_syn, Vm, E_rev, eps=1.0):
    """g = I / (V - E), guarding near-zero driving force (|V-E| < eps mV)."""
    driving_force = Vm - E_rev
    g = np.full_like(I_syn, np.nan)
    valid = np.abs(driving_force) >= eps
    g[valid] = I_syn[valid] / driving_force[valid]
    return g

def zero_crossings(trace, times, threshold=0.0, direction='rising'):
    above = trace >= threshold
    if direction == 'rising':
        idx = np.where(above[1:] & ~above[:-1])[0]
    elif direction == 'falling':
        idx = np.where(~above[1:] & above[:-1])[0]
    else:
        raise ValueError("direction must be 'rising' or 'falling'")
    return times[idx + 1]

sections_to_plot = ['basal_2', 'apical_oblique', 'soma']

def make_syn_vm_figure(xlim=None):
    fig, axes = plt.subplots(len(sections_to_plot), 1, figsize=(9, 7),
                              sharex=True, constrained_layout=True)

    for i, sec in enumerate(sections_to_plot):
        ax_vm = axes[i]
        vm_full = np.asarray(net.cell_response.vsec[0][example_gid][sec])
        vm = vm_full[1:]

        I_exc = get_syn_current_sum(example_gid, sec, 'ampa') + get_syn_current_sum(example_gid, sec, 'nmda')
        I_inh = get_syn_current_sum(example_gid, sec, 'gabaa') + get_syn_current_sum(example_gid, sec, 'gabab')

        g_exc = synaptic_conductance(I_exc, vm_full, E_rev=0.0)[1:]
        g_inh = synaptic_conductance(I_inh, vm_full, E_rev=-80.0)[1:]

        ax_vm.axhline(0, color='gray', lw=0.5, alpha=0.5)
        for t_cross in zero_crossings(vm, times_, direction='rising'):
            ax_vm.axvline(t_cross, color='green', lw=0.8, linestyle='--', alpha=0.7, zorder=0)
        for t_cross in zero_crossings(vm, times_, direction='falling'):
            ax_vm.axvline(t_cross, color='purple', lw=0.8, linestyle=':', alpha=0.7, zorder=0)

        ax_vm.plot(times_, vm, color='k', lw=1)
        ax_vm.set_ylabel('$V_m$ (mV)')
        ax_vm.set_title(sec, fontsize=10)
        if xlim is not None:
            ax_vm.set_xlim(*xlim)

        ax_g = ax_vm.twinx()
        ax_g.plot(times_, g_exc, color='tab:red', lw=1)
        ax_g.plot(times_, g_inh, color='tab:blue', lw=1)
        ax_g.set_ylabel('$g_{syn}$ (µS)')

    axes[-1].set_xlabel('Time (ms)')
    fig.legend(
        handles=[Line2D([0], [0], color='k', label='$V_m$'),
                 Line2D([0], [0], color='tab:red', label='$g_{exc}$ (AMPA+NMDA)'),
                 Line2D([0], [0], color='tab:blue', label='$g_{inh}$ (GABA_A+GABA_B)'),
                 Line2D([0], [0], color='green', lw=0.8, linestyle='--', label='$V_m=0$ rising'),
                 Line2D([0], [0], color='purple', lw=0.8, linestyle=':', label='$V_m=0$ falling')],
        loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, ncol=2,
    )
    fig.suptitle(f"$V_m$ and synaptic conductance — gid={example_gid}")
    plt.show()

make_syn_vm_figure()
#make_syn_vm_figure(xlim=(60, 75))



# for plotting calcium concentration traces:
sections = list(net.cell_response.vsec[0][example_gid].keys())

cell_type = 'L5_pyramidal'
template = net.cell_types[cell_type]['cell_object']
start_gid = net.gid_ranges[cell_type][0]
soma_pos = np.asarray(net.pos_dict[cell_type][example_gid - start_gid], dtype=float)

section_depths = np.array([
    np.mean(np.asarray(template.sections[sec]._end_pts, dtype=float)[:, 2]) + soma_pos[2]
    for sec in sections
])

Vm_matrix = np.array(
    [net.cell_response.vsec[0][example_gid][sec] for sec in sections]
)[:, 1:]
# for calcium
Ca_matrix = np.array(
    [net.cell_response.ca[0][example_gid][sec] for sec in sections]
)[:, 1:]

def make_ca_traces_figure(xlim=None, fixed_height=60.0, baseline_samps=50):
    Ca_baseline = Ca_matrix - np.mean(Ca_matrix[:, :baseline_samps], axis=1, keepdims=True)
    row_ptp = np.ptp(Ca_baseline, axis=1, keepdims=True)
    row_ptp[row_ptp == 0] = 1.0  # guard against a flat/no-signal compartment
    Ca_normalized = Ca_baseline / row_ptp * fixed_height

    z_edges = tme._z_edges_from_array(net, "probe1")

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4])
    ax_morph = fig.add_subplot(gs[0, 0])
    ax_ca = fig.add_subplot(gs[0, 1])

    plott.plot_cell_morphology_for_lfp_csd(
        net, contact_positions=contact_labels,
        cell_types=('L5_pyramidal',), gid=example_gid,
        ax=ax_morph, show=False,
    )

    tme.plot_stacked_traces(
        ax_ca, times_, Ca_normalized, depths=section_depths,
        color='darkgreen', scale=-1.0, labels=sections,
    )
    ax_ca.set_yticks(section_depths)
    ax_ca.set_yticklabels(sections)
    ax_ca.set_xlabel("Time (ms)")
    ax_ca.set_title(f"[Ca²⁺] by compartment — gid={example_gid}")
    if xlim is not None:
        ax_ca.set_xlim(*xlim)

    for ax in (ax_morph, ax_ca):
        for z in z_edges:
            ax.axhline(z, color='red', alpha=0.2, lw=1)

    plt.show()

make_ca_traces_figure()
make_ca_traces_figure(xlim=(50, 75))
make_ca_traces_figure(xlim=(135, 160))

'''
def make_vm_ca_traces_figure(xlim=None, fixed_height=60.0, baseline_samps=50):
    Vm_baseline = Vm_matrix - np.mean(Vm_matrix[:, :baseline_samps], axis=1, keepdims=True)
    Vm_normalized = Vm_baseline / np.ptp(Vm_baseline, axis=1, keepdims=True) * fixed_height

    Ca_baseline = Ca_matrix - np.mean(Ca_matrix[:, :baseline_samps], axis=1, keepdims=True)
    row_ptp_ca = np.ptp(Ca_baseline, axis=1, keepdims=True)
    row_ptp_ca[row_ptp_ca == 0] = 1.0
    Ca_normalized = Ca_baseline / row_ptp_ca * fixed_height

    z_edges = tme._z_edges_from_array(net, "probe1")

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4])
    ax_morph = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])

    plott.plot_cell_morphology_for_lfp_csd(
        net, contact_positions=contact_labels,
        cell_types=('L5_pyramidal',), gid=example_gid,
        ax=ax_morph, show=False,
    )

    tme.plot_stacked_traces(
        ax_trace, times_, Vm_normalized, depths=section_depths,
        color='k', scale=-1.0, labels=sections,
    )
    tme.plot_stacked_traces(
        ax_trace, times_, Ca_normalized, depths=section_depths,
        color='darkgreen', scale=-1.0, labels=sections,
    )

    ax_trace.set_yticks(section_depths)
    ax_trace.set_yticklabels(sections)
    ax_trace.set_xlabel("Time (ms)")
    ax_trace.set_title(f"Vm & [Ca²⁺] by compartment — gid={example_gid}")
    if xlim is not None:
        ax_trace.set_xlim(*xlim)

    for ax in (ax_morph, ax_trace):
        for z in z_edges:
            ax.axhline(z, color='red', alpha=0.2, lw=1)

    legend_elements = [
        Line2D([0], [0], color='k', lw=1.5, label='Vm'),
        Line2D([0], [0], color='darkgreen', lw=1.5, label='[Ca²⁺]'),
    ]
    ax_trace.legend(handles=legend_elements, loc='lower right')

    plt.show()

make_vm_ca_traces_figure()
#make_vm_ca_traces_figure(xlim=(50, 75))
#make_vm_ca_traces_figure(xlim=(135, 160))
'''

order = np.argsort(-section_depths)  # descending depth: most superficial (tuft) first
sections_ordered = [sections[i] for i in order]
Vm_ordered = Vm_matrix[order]
Ca_ordered = Ca_matrix[order]
vm_ylim = (np.min(Vm_matrix) - 5, np.max(Vm_matrix) + 5)
ca_pad = 0.05 * (np.max(Ca_matrix) - np.min(Ca_matrix))
ca_ylim = (np.min(Ca_matrix) - ca_pad, np.max(Ca_matrix) + ca_pad)

def make_vm_ca_panels_figure(xlim=None):
    n = len(sections_ordered)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.6 * n), sharex=True,
                              constrained_layout=True)
    for i, sec in enumerate(sections_ordered):
        ax_vm = axes[i]
        ax_ca = ax_vm.twinx()
        ax_vm.axhline(0, color='gray', lw=0.7, alpha=0.6, zorder=0)
        for t_cross in zero_crossings(Vm_ordered[i], times_):
            ax_vm.axvline(t_cross, color='gray', lw=0.7, linestyle='--',
                          alpha=0.6, zorder=0)
        ax_vm.plot(times_, Vm_ordered[i], color='indigo', lw=1)
        ax_ca.plot(times_, Ca_ordered[i], color='darkorange', lw=1)
        ax_vm.set_ylim(*vm_ylim)
        ax_ca.set_ylim(*ca_ylim)
        ax_vm.set_ylabel('$V_m$ (mV)', color='indigo', fontsize=8)
        ax_ca.set_ylabel('[Ca$^{2+}$] (mM)', color='darkorange', fontsize=8)
        ax_vm.tick_params(axis='y', labelcolor='indigo', labelsize=7)
        ax_ca.tick_params(axis='y', labelcolor='darkorange', labelsize=7)
        ax_vm.text(-0.14, 0.5, sec, transform=ax_vm.transAxes,
                   ha='right', va='center', fontsize=9, fontweight='bold')
        if xlim is not None:
            ax_vm.set_xlim(*xlim)
    axes[-1].set_xlabel('Time (ms)')
    legend_elements = [
        Line2D([0], [0], color='indigo', lw=1.5, label='$V_m$'),
        Line2D([0], [0], color='darkorange', lw=1.5, label='[Ca$^{2+}$]'),
        Line2D([0], [0], color='gray', lw=0.7, linestyle='--', label='$V_m=0$ crossing'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=9)
    fig.suptitle(f'Membrane potential & [Ca²⁺] by compartment — gid={example_gid}',
                 fontsize=11)
    plt.show()

make_vm_ca_panels_figure()
#make_vm_ca_panels_figure(xlim=(55, 80))
#make_vm_ca_panels_figure(xlim=(140, 165))


# The following functions are used to extract the middle-segment density of ionic currents and synaptic currents for apical tuft and soma
def mid_segment_density(name, gid, section):
    """Current density from the middle segment of the section (native NEURON units: mA/cm^2)."""
    seg_dict = net.cell_response.transmembrane_currents[name][0][gid][section]
    values = list(seg_dict.values())
    mid_idx = len(values) // 2
    return np.asarray(values[mid_idx])

def get_syn_current(gid, section, keyword):
    """Synaptic point-process current (absolute, nA)."""
    syn_dict = net.cell_response.isec[0][gid].get(section, {})
    for k, v in syn_dict.items():
        if keyword.lower() in k.lower():
            return np.asarray(v)
    return np.zeros_like(net.cell_response.times)

compartments = {'soma': 'black', 'apical_tuft': 'darkorange'}

Vm     = {c: np.asarray(net.cell_response.vsec[0][example_gid][c])[1:] for c in compartments}
I_Na   = {c: mid_segment_density('ina_hh2', example_gid, c)[1:] for c in compartments}
I_Ca   = {c: mid_segment_density('ica_ca', example_gid, c)[1:] for c in compartments}
I_CaT  = {c: mid_segment_density('ica_cat', example_gid, c)[1:] for c in compartments}
I_K    = {c: (mid_segment_density('ik_hh2', example_gid, c) + mid_segment_density('ik_kca', example_gid, c)
              + mid_segment_density('ik_km', example_gid, c))[1:] for c in compartments}
I_AMPA = {c: get_syn_current(example_gid, c, 'ampa')[1:] for c in compartments}
I_NMDA = {c: get_syn_current(example_gid, c, 'nmda')[1:] for c in compartments}
I_GABAA = {c: get_syn_current(example_gid, c, 'gabaa')[1:] for c in compartments}
I_GABAB = {c: get_syn_current(example_gid, c, 'gabab')[1:] for c in compartments}

panels = [
    ('$V_m$ (mV)', Vm),
    ('$I_{Na}$ (mA/cm²)', I_Na),
    ('$I_{Ca}$ (mA/cm²)', I_Ca),
    ('$I_{CaT}$ (mA/cm²)', I_CaT),
    ('$I_K$ (mA/cm²)', I_K),
    ('$I_{AMPA}$ (nA)', I_AMPA),
    ('$I_{NMDA}$ (nA)', I_NMDA),
    ('$I_{GABA_A}$ (nA)', I_GABAA),
    ('$I_{GABA_B}$ (nA)', I_GABAB),
]

def make_figure(xlim):
    fig, axes = plt.subplots(len(panels), 1, figsize=(4.5, 8), sharex=True, constrained_layout=True)
    for i, (ylabel, data) in enumerate(panels):
        ax = axes[i]
        ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
        for c, color in compartments.items():
            ax.plot(times_, data[c], color=color, lw=1.3, label=c)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(*xlim)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle(f'Soma vs. apical_tuft — gid={example_gid}', fontsize=12)
    plt.show()

make_figure((25, 50))
make_figure((60, 85))
make_figure((130, 155))