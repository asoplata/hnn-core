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
with open('runscripts/data/sim_results_dt000625_withmembranepot_newelect2.pkl', 'rb') as f: #from -550 up
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

tme.convert_currents_to_numpy(net, include_isec=True)

'''
# one example trace
channel = "agg_i_mem"
gid = list(net.gid_ranges["L5_pyramidal"])[0]
trial_data = net.cell_response.transmembrane_currents[channel][0]  # trial 0
section = list(trial_data[gid].keys())[0]
segment = list(trial_data[gid][section].keys())[0]

example = trial_data[gid][section][segment]
print(len(example))        # length of that trace
'''

step = 4
tme.downsample_currents(net, step=step, include_isec=True, include_vsec=True)
times = times[::step]
lfp_hnn = lfp_hnn[:,::step]

times_ = times[1:]
lfp_hnn_ = lfp_hnn[:, 1:]

delta = np.median(np.diff(depths))
csd_from_lfp = calculate_csd2d(lfp_hnn, delta=delta)
csd_from_lfp_ = csd_from_lfp[:, 1:]

'''
# one example trace
channel = "agg_i_mem"
gid = list(net.gid_ranges["L5_pyramidal"])[0]
trial_data = net.cell_response.transmembrane_currents[channel][0]  # trial 0
section = list(trial_data[gid].keys())[0]
segment = list(trial_data[gid][section].keys())[0]

example = trial_data[gid][section][segment]
print(len(example))        # length of that trace
'''
# plot of LFP and CSD (from hnn output) to have a visual reference
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8),
                        gridspec_kw={'height_ratios': [3, 3]})

# LFP panel
plott.plot_laminar_lfp_AC(
    times_, lfp_hnn_, contact_labels,
    ax=axs[0], scale=5.0, voltage_scalebar=50, show=False)

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
sigma = 1 # CHECK THIS, I think in HNN's CSD sigma is actually dropped.
# To compare csd_from_sources (μA/mm³) with HNN's output (μV/μm²):
csd_in_hnn_units = csd_from_sources / (sigma * 1e3)   # → μV/μm²
csd_in_hnn_units_ = csd_in_hnn_units[:, 1:]   

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
    unit_csd="µA/mm³",
    overlay_raster_on_csd=True,
    sink="red")
fig.suptitle("LFP/CSD from whole-network LFP (2nd derivative)")



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

# --- CSD as 2nd derivative of the synaptic-current LFP ---
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
# dipole from gabab only
dpl_gabab = tme.reconstruct_dipole_from_sources(net, sources_syn_gabab, I_syn_gabab)
dpl_gabab_ = dpl_gabab[1:]

fig, ax = plt.subplots()
ax.plot(times_, dpl_agg_i_mem_, 'k', label='Total (agg_i_mem)', lw=1.5)
ax.plot(times_, dpl_gabab_, label='GABA_B', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()
#fig.suptitle("Dipole from synaptic currents")
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


vmax1 = np.max(np.abs(csd_syn_gabaa_from_lfp_))

fig1 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_gabaa_,
    csd_syn_gabaa_from_lfp_,
    contact_labels,
    net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-0.01,#vmax1,
    vmax=0.01,#vmax1,
    figsize=(18, 6),
    overlay_csd_traces=True,
    unit_csd="µV/µm²",
    overlay_raster_on_csd=True,
    sink="red")
fig1.suptitle("LFP/CSD from GABA_A currents (2nd derivative of LFP)")

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

# --- CSD as 2nd derivative of the AMPA-current LFP ---
csd_syn_ampa_from_lfp_ = calculate_csd2d(lfp_syn_ampa_, delta=delta)
vmax1 = np.max(np.abs(csd_syn_ampa_from_lfp_))

fig1 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_ampa_,
    csd_syn_ampa_from_lfp_,
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
    overlay_raster_on_csd=True,
    sink="red")
fig1.suptitle("LFP/CSD from AMPA currents (2nd derivative of LFP)")

# --- CSD directly from AMPA current sources ---
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

# --- CSD as 2nd derivative of the NMDA-current LFP ---
csd_syn_nmda_from_lfp_ = calculate_csd2d(lfp_syn_nmda_, delta=delta)
vmax1 = np.max(np.abs(csd_syn_nmda_from_lfp_))

fig1 = plott.plot_lfp_morph_csd(
    times_,
    lfp_syn_nmda_,
    csd_syn_nmda_from_lfp_,
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
    overlay_raster_on_csd=True,
    sink="red")
fig1.suptitle("LFP/CSD from NMDA currents (2nd derivative of LFP)")

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



#################
# FIGURE panel for meetings:
#############
#directly from sources, not from LFP
fig_all = plott.make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_from_sources_, csd_cap_, csd_ionic_, csd_syn_,
    csd_syn_gabab_, csd_syn_gabaa_, csd_syn_ampa_, csd_syn_nmda_,
    vmax_row0=60, vmax_row1=10,
    suptitle="Whole network (all pyramidal cells) - from sources",
)
plt.show()

# from lfp
vmax_row0 = max(
    np.max(np.abs(csd_from_lfp_)),
    np.max(np.abs(csd_cap_from_lfp_)),
    np.max(np.abs(csd_ionic_from_lfp_)),
    np.max(np.abs(csd_syn_from_lfp_)),
)
vmax_row1 = max(
    np.max(np.abs(csd_syn_gabab_from_lfp_)),
    np.max(np.abs(csd_syn_gabaa_from_lfp_)),
    np.max(np.abs(csd_syn_ampa_from_lfp_)),
    np.max(np.abs(csd_syn_nmda_from_lfp_)),
)

fig_all_from_lfp = plott.make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_from_lfp_, csd_cap_from_lfp_, csd_ionic_from_lfp_, csd_syn_from_lfp_,
    csd_syn_gabab_from_lfp_, csd_syn_gabaa_from_lfp_, csd_syn_ampa_from_lfp_, csd_syn_nmda_from_lfp_,
    vmax_row0=0.06, vmax_row1=0.01,
    suptitle="Whole network (all pyramidal cells) - from LFP (2nd derivative)",
)
plt.show()


'''
for name, drive in net.external_drives.items():
    print(name, drive)

fig, ax = plt.subplots()
ax.plot(times, np.sum(I_syn_gabab, axis=0), label='GABA-B')
ax.plot(times, np.sum(I_syn_gabaa, axis=0), label='GABA-A')
ax.plot(times, np.sum(I_syn_ampa,  axis=0), label='AMPA')
ax.plot(times, np.sum(I_syn_nmda,  axis=0), label='NMDA')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Current (nA)')
ax.legend()
plt.show()
'''


###########################
# CAPACITIVE CSD FOR A SINGLE SPIKING L5 PYRAMIDAL CELL
###########################

# 1) find L5 pyramidal cells that spiked in trial 0
l5_gids = set(net.gid_ranges["L5_pyramidal"])
spike_gids_trial0 = set(np.asarray(net.cell_response.spike_gids[0]).tolist())
spiking_l5_gids = sorted(l5_gids & spike_gids_trial0)

print(f"{len(spiking_l5_gids)} spiking L5 pyramidal cells (of {len(l5_gids)})")

# 2) pick one -- e.g. the first spiking L5 pyramidal cell
example_gid = 207#spiking_l5_gids[0]
print(f"Selected gid = {example_gid}")

# 3) filter the capacitive sources down to this one cell
sources_cap_cell, I_cap_cell = tme.filter_sources(
    sources_cap, I_cap, gid_subset=[example_gid]
)

# 4) bin -> CSD for this single cell
B_cap_cell, V_bin_cap_cell, _ = tme.build_binning_matrix_for_sources(
    net, sources_cap_cell, array_name="probe1"
)
csd_cap_cell = tme.compute_csd_from_sources(B_cap_cell, I_cap_cell, V_bin_cap_cell)
csd_cap_cell_ = csd_cap_cell[:, 1:]  # drop first sample, matches times_

# 5) plot morphology | CSD (top row) + membrane potential (below CSD only)
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


#another version:
# Capacitive current restricted to this cell's basal dendrites
sources_cap_basal, I_cap_basal = tme.filter_sources(
    sources_cap_cell, I_cap_cell, sections={"basal_1", "basal_2", "basal_3"}
)
I_cap_basal_total = I_cap_basal.sum(axis=0)[1:]  # sum over segments, drop first sample

# Synaptic current for this same cell, restricted to the same basal sections
sources_syn_cell, I_syn_cell = tme.filter_sources(
    sources_syn, I_syn, gid_subset=[example_gid]
)
sources_syn_basal, I_syn_basal = tme.filter_sources(
    sources_syn_cell, I_syn_cell, sections={"basal_1", "basal_2", "basal_3"}
)
I_syn_basal_total = I_syn_basal.sum(axis=0)[1:]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(times_, I_cap_basal_total, label="Capacitive current (basal)")
ax.plot(times_, I_syn_basal_total, label="Synaptic current (basal)")
for t in cell_spike_times:
    ax.axvline(t, color='k', linestyle='--', alpha=0.5, lw=1)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Current (nA)")
ax.legend()
ax.set_title(f"Basal dendrite currents — gid={example_gid}")
plt.show()



# --- capacitive, ionic, synaptic CSD for this single cell ---
sources_ionic_cell, I_ionic_cell = tme.filter_sources(sources_ionic, I_ionic, gid_subset=[example_gid])
sources_syn_cell, I_syn_cell = tme.filter_sources(sources_syn, I_syn, gid_subset=[example_gid])

def csd_for_cell(sources, I, array_name="probe1"):
    B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sources, array_name=array_name)
    csd = tme.compute_csd_from_sources(B, I, V_bin)
    return csd[:, 1:]  # drop first sample, matches times_

csd_ionic_cell_ = csd_for_cell(sources_ionic_cell, I_ionic_cell)
csd_syn_cell_ = csd_for_cell(sources_syn_cell, I_syn_cell)
# csd_cap_cell_ was already computed earlier

# slab (electrode-bin) boundaries used by build_binning_matrix_for_sources
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

panels = [
    (ax_cap, csd_cap_cell_, "Capacitive"),
    (ax_ionic, csd_ionic_cell_, "Ionic"),
    (ax_syn, csd_syn_cell_, "Synaptic"),
]

for ax, csd_, title in panels:
    vmax = np.max(np.abs(csd_))
    plott.plot_laminar_csd_AC(
        times_, csd_, contact_labels,
        ax=ax, vmin=-vmax, vmax=vmax,
        overlay_csd_traces=False, unit_csd="µA/mm³", sink="red", show=False,
    )
    ax.set_title(f"{title} CSD — gid={example_gid}")

# faint red lines marking the slab (electrode-bin) boundaries, on every panel
for ax in (ax_morph, ax_cap, ax_ionic, ax_syn):
    for z in z_edges:
        ax.axhline(z, color='red', alpha=0.2, lw=1)

plt.show()



template = net.cell_types["L5_pyramidal"]["cell_object"]

def section_mean_z(section):
    pts = np.asarray(template.sections[section]._end_pts, dtype=float)
    return pts[:, 2].mean()

sections = sorted(template.sections.keys(), key=section_mean_z, reverse=True)
n_sections = len(sections)

ncols = 4
nrows = int(np.ceil(n_sections / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
axes = np.atleast_1d(axes).flatten()

for i, section in enumerate(sections):
    ax = axes[i]
    vm_section = np.asarray(net.cell_response.vsec[0][example_gid][section])[1:]

    _, I_syn_sec_exc = tme.filter_sources(
        sources_syn_cell, I_syn_cell, sections={section}, syn_names=["ampa", "nmda"]
    )
    _, I_syn_sec_inh = tme.filter_sources(
        sources_syn_cell, I_syn_cell, sections={section}, syn_names=["gabaa", "gabab"]
    )
    I_exc_total = I_syn_sec_exc.sum(axis=0)[1:]
    I_inh_total = I_syn_sec_inh.sum(axis=0)[1:]

    ax.plot(times_, vm_section, color='k', lw=1)
    for t in cell_spike_times:
        ax.axvline(t, color='k', linestyle='--', alpha=0.4, lw=0.8)
    ax.set_title(section, fontsize=10)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Vm (mV)")

    ax2 = ax.twinx()
    ax2.plot(times_, I_exc_total, color='tab:red', lw=1)
    ax2.plot(times_, I_inh_total, color='tab:blue', lw=1)
    ax2.set_ylabel("I (nA)")

# hide any unused grid cells
for j in range(n_sections, len(axes)):
    axes[j].axis('off')

fig.legend(
    handles=[Line2D([0], [0], color='k', label='Vm'),
             Line2D([0], [0], color='tab:red', label='Excitatory (AMPA+NMDA)'),
             Line2D([0], [0], color='tab:blue', label='Inhibitory (GABA_A+GABA_B)')],
    loc='lower right', fontsize=9,
)
fig.suptitle(f"Vm and synaptic current by type — gid={example_gid}, all compartments")
plt.show()





# THIS IS INTERESTING
template = net.cell_types["L5_pyramidal"]["cell_object"]

def section_mean_z(section):
    pts = np.asarray(template.sections[section]._end_pts, dtype=float)
    return pts[:, 2].mean()

sections = sorted(template.sections.keys(), key=section_mean_z, reverse=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(sections)))
highlight = {"apical_tuft", "soma"}

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

for section, color in zip(sections, colors):
    v = np.asarray(net.cell_response.vsec[0][example_gid][section])[1:]
    lw = 2.5 if section in highlight else 1.0
    zorder = 3 if section in highlight else 1
    ax.plot(times_, v, color=color, lw=lw, label=section, zorder=zorder)

for t in cell_spike_times:
    ax.axvline(t, color='k', linestyle='--', alpha=0.4, lw=0.8, zorder=2)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Vm (mV)")
ax.set_title(f"Membrane potential across all compartments — gid={example_gid}")
ax.legend(loc='upper right', fontsize=8, ncol=2)
plt.show()

#combined with the csd
# --- capacitive, ionic, synaptic CSD for this single cell ---
sources_cap_cell, I_cap_cell = tme.filter_sources(sources_cap, I_cap, gid_subset=[example_gid])
sources_ionic_cell, I_ionic_cell = tme.filter_sources(sources_ionic, I_ionic, gid_subset=[example_gid])
sources_syn_cell, I_syn_cell = tme.filter_sources(sources_syn, I_syn, gid_subset=[example_gid])

def csd_for_cell(sources, I, array_name="probe1"):
    B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sources, array_name=array_name)
    csd = tme.compute_csd_from_sources(B, I, V_bin)
    return csd[:, 1:]  # drop first sample, matches times_

csd_cap_cell_ = csd_for_cell(sources_cap_cell, I_cap_cell)
csd_ionic_cell_ = csd_for_cell(sources_ionic_cell, I_ionic_cell)
csd_syn_cell_ = csd_for_cell(sources_syn_cell, I_syn_cell)

# slab (electrode-bin) boundaries used by build_binning_matrix_for_sources
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

panels = [
    (ax_cap, csd_cap_cell_, "Capacitive"),
    (ax_ionic, csd_ionic_cell_, "Ionic"),
    (ax_syn, csd_syn_cell_, "Synaptic"),
]

for ax, csd_, title in panels:
    vmax = np.max(np.abs(csd_))
    plott.plot_laminar_csd_AC(
        times_, csd_, contact_labels,
        ax=ax, vmin=-vmax, vmax=vmax,
        overlay_csd_traces=False, unit_csd="µA/mm³", sink="red", show=False,
    )
    ax.set_title(f"{title} CSD — gid={example_gid}")

# faint red lines marking the slab (electrode-bin) boundaries, on every panel
for ax in (ax_morph, ax_cap, ax_ionic, ax_syn):
    for z in z_edges:
        ax.axhline(z, color='red', alpha=0.2, lw=1)

plt.show()


# CSD as traces:
def plot_csd_traces(ax, times_, csd_, contact_labels, title, scale_mult=2.0):
    scale = scale_mult * np.diff(contact_labels)[0] / np.max(np.abs(csd_))
    tme.plot_stacked_traces(ax, times_, csd_, depths=contact_labels, color='k', scale=scale)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)

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

plot_csd_traces(ax_cap, times_, csd_cap_cell_, contact_labels, f"Capacitive CSD — gid={example_gid}")
plot_csd_traces(ax_ionic, times_, csd_ionic_cell_, contact_labels, f"Ionic CSD — gid={example_gid}")
plot_csd_traces(ax_syn, times_, csd_syn_cell_, contact_labels, f"Synaptic CSD — gid={example_gid}")

for ax in (ax_morph, ax_cap, ax_ionic, ax_syn):
    for z in z_edges:
        ax.axhline(z, color='red', alpha=0.2, lw=1)

fig.supxlabel(
    "CSD sign convention: positive = source (current leaving the cell), negative = sink (current entering the cell)",
    fontsize=10,
)

plt.show()


# ZOOMS in
def make_csd_traces_figure(xlim, scale_mult=2.0):
    mask = (times_ >= xlim[0]) & (times_ <= xlim[1])

    def plot_csd_traces(ax, csd_, title):
        vmax = np.max(np.abs(csd_[:, mask]))
        scale = scale_mult * np.diff(contact_labels)[0] / vmax
        tme.plot_stacked_traces(ax, times_, csd_, depths=contact_labels, color='k', scale=scale)
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

    plot_csd_traces(ax_cap, csd_cap_cell_, f"Capacitive CSD — gid={example_gid}")
    plot_csd_traces(ax_ionic, csd_ionic_cell_, f"Ionic CSD — gid={example_gid}")
    plot_csd_traces(ax_syn, csd_syn_cell_, f"Synaptic CSD — gid={example_gid}")

    for ax in (ax_morph, ax_cap, ax_ionic, ax_syn):
        for z in z_edges:
            ax.axhline(z, color='red', alpha=0.2, lw=1)

    fig.supxlabel(
        "CSD sign convention: positive = source (current leaving the cell), negative = sink (current entering the cell)",
        fontsize=10,
    )
    fig.suptitle(f"Time window: {xlim[0]}-{xlim[1]} ms")

    plt.show()


make_csd_traces_figure((50, 75))
make_csd_traces_figure((135, 160))

def make_vm_traces_figure(xlim=None, fixed_height=60.0, baseline_samps=50):
    Vm_baseline = Vm_matrix - np.mean(Vm_matrix[:, :baseline_samps], axis=1, keepdims=True)
    row_ptp = np.ptp(Vm_baseline, axis=1, keepdims=True)
    Vm_normalized = Vm_baseline / row_ptp * fixed_height  # every row now has the same peak-to-peak height

    z_edges = tme._z_edges_from_array(net, "probe1")

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4])
    ax_morph = fig.add_subplot(gs[0, 0])
    ax_vm = fig.add_subplot(gs[0, 1])

    plott.plot_cell_morphology_for_lfp_csd(
        net, contact_positions=contact_labels,
        cell_types=('L5_pyramidal',), gid=example_gid,
        ax=ax_morph, show=False,
    )

    tme.plot_stacked_traces(
        ax_vm, times_, Vm_normalized, depths=section_depths,
        color='k', scale=1.0, labels=sections,
    )
    ax_vm.set_yticks(section_depths)
    ax_vm.set_yticklabels(sections)
    ax_vm.set_xlabel("Time (ms)")
    ax_vm.set_title(f"Membrane potential by compartment — gid={example_gid}")
    if xlim is not None:
        ax_vm.set_xlim(*xlim)

    for ax in (ax_morph, ax_vm):
        for z in z_edges:
            ax.axhline(z, color='red', alpha=0.2, lw=1)

    plt.show()

make_vm_traces_figure()
make_vm_traces_figure(xlim=(50, 75))
make_vm_traces_figure(xlim=(135, 160))
# UP TO HERE!!



##################
# TO plot the contributions to the CSD from specific ionic currents
##################


def compute_lfp_csd_for_sources(net, sources, I, array_name="probe1"):
    B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sources, array_name=array_name)
    csd = tme.compute_csd_from_sources(B, I, V_bin)
    T = tme.build_transfer_resistance_matrix_for_sources(net, sources, array_name=array_name)
    lfp = tme.reconstruct_lfp_from_sources(T, I)
    return lfp[:, 1:], csd[:, 1:]

# for checking the contribution from specific ionic currents (ana capacitive)
sources_intr, I_intr = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal", "L5_pyramidal"],
    channels=["agg_i_cap", "ina_hh2", "ik_hh2", "ik_kca", "ik_km",
              "ica_ca", "ica_cat", "il_hh2", "i_ar"],
)
sources_agg_i_cap, I_agg_i_cap = tme.filter_sources(sources_intr, I_intr, labels=["agg_i_cap"])
sources_ina, I_ina = tme.filter_sources(sources_intr, I_intr, labels=["ina_hh2"])
sources_ik,  I_ik  = tme.filter_sources(sources_intr, I_intr, labels=["ik_hh2"])
sources_ik_kca, I_ik_kca = tme.filter_sources(sources_intr, I_intr, labels=["ik_kca"])
sources_ik_km, I_ik_km = tme.filter_sources(sources_intr, I_intr, labels=["ik_km"])
sources_ica_ca, I_ica_ca = tme.filter_sources(sources_intr, I_intr, labels=["ica_ca"])
sources_ica_cat, I_ica_cat = tme.filter_sources(sources_intr, I_intr, labels=["ica_cat"])
sources_il, I_il = tme.filter_sources(sources_intr, I_intr, labels=["il_hh2"])
sources_i_ar, I_i_ar = tme.filter_sources(sources_intr, I_intr, labels=["i_ar"])    


lfp_agg_i_cap_, csd_agg_i_cap_ = compute_lfp_csd_for_sources(net, sources_agg_i_cap, I_agg_i_cap)
lfp_ina_,       csd_ina_       = compute_lfp_csd_for_sources(net, sources_ina,       I_ina)
lfp_ik_,        csd_ik_        = compute_lfp_csd_for_sources(net, sources_ik,        I_ik)
lfp_ik_kca_,    csd_ik_kca_    = compute_lfp_csd_for_sources(net, sources_ik_kca,    I_ik_kca)
lfp_ik_km_,     csd_ik_km_     = compute_lfp_csd_for_sources(net, sources_ik_km,     I_ik_km)
lfp_ica_ca_,    csd_ica_ca_    = compute_lfp_csd_for_sources(net, sources_ica_ca,    I_ica_ca)
lfp_ica_cat_,   csd_ica_cat_   = compute_lfp_csd_for_sources(net, sources_ica_cat,   I_ica_cat)
lfp_il_,        csd_il_        = compute_lfp_csd_for_sources(net, sources_il,        I_il)
lfp_i_ar_,      csd_i_ar_      = compute_lfp_csd_for_sources(net, sources_i_ar,      I_i_ar)

'''
fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_agg_i_cap_, 
    csd_agg_i_cap_, 
    contact_labels, 
    net, 
    ext_inputs=net.cell_response, 
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']}, 
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-10,
    vmax=10,
    figsize=(18, 6),
    overlay_csd_traces=False,
    unit_csd="µA/mm³")
fig.suptitle("LFP/CSD from capacitive currents")
'''


intrinsic_panels = [
    (lfp_agg_i_cap_, csd_agg_i_cap_, "Capacitive (agg_i_cap)"),
    (lfp_ina_,       csd_ina_,       "Na (ina_hh2)"),
    (lfp_ik_,        csd_ik_,        "K (ik_hh2)"),
    (lfp_ik_kca_,    csd_ik_kca_,    "KCa (ik_kca)"),
    (lfp_ik_km_,     csd_ik_km_,     "KM (ik_km)"),
    (lfp_ica_ca_,    csd_ica_ca_,    "Ca (ica_ca)"),
    (lfp_ica_cat_,   csd_ica_cat_,   "CaT (ica_cat)"),
    (lfp_il_,        csd_il_,        "Leak (il_hh2)"),
    (lfp_i_ar_,      csd_i_ar_,      "Ih (i_ar)"),
]

for lfp_, csd_, title in intrinsic_panels:
    fig = plott.plot_lfp_morph_csd(
        times_, lfp_, csd_, contact_labels, net,
        ext_inputs=net.cell_response,
        spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
        scale_lfp=3.0,
        voltage_scalebar=200,
        vmin=-10,
        vmax=10,
        figsize=(18, 6),
        overlay_csd_traces=False,
        unit_csd="µA/mm³",
        overlay_raster_on_csd=True,
        sink="r"
    )
    fig.suptitle(f"LFP/CSD from {title}")

####
# check: is the sum of all the ionic and capacitive matching what we got before?
####
csd_cap_ion_ = csd_agg_i_cap_ + csd_ina_ + csd_ik_ + csd_ik_kca_ + csd_ik_km_ + csd_ica_ca_ + csd_ica_cat_ + csd_il_ + csd_i_ar_
resid = csd_intr_ - csd_cap_ion_
print(np.min(resid))
print(np.max(resid))


##############################
# Dividing spiking from non-spiking pyramidal cells and assess contributions to the CSD
##############################

pyr_gids = set(net.gid_ranges["L2_pyramidal"]) | set(net.gid_ranges["L5_pyramidal"])
spike_gids_trial0 = set(np.asarray(net.cell_response.spike_gids[0]).tolist())

spiking_gids = pyr_gids & spike_gids_trial0
non_spiking_gids = pyr_gids - spiking_gids

print(f"{len(spiking_gids)} spiking / {len(non_spiking_gids)} non-spiking (of {len(pyr_gids)} total)")
#182 spiking / 18 non-spiking (of 200 total)

def csd_for_subset(sources, I, gid_subset, array_name="probe1"):
    sub_sources, sub_I = tme.filter_sources(sources, I, gid_subset=gid_subset)
    B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sub_sources, array_name=array_name)
    return tme.compute_csd_from_sources(B, sub_I, V_bin)[:, 1:]

source_pairs = {
    "csd_from_sources_": (sources_agg,        I_agg),
    "csd_cap_":          (sources_cap_,       I_cap_),
    "csd_ionic_":        (sources_ionic_,     I_ionic_),
    "csd_syn_":          (sources_syn,        I_syn),
    "csd_syn_gabab_":    (sources_syn_gabab,  I_syn_gabab),
    "csd_syn_gabaa_":    (sources_syn_gabaa,  I_syn_gabaa),
    "csd_syn_ampa_":     (sources_syn_ampa,   I_syn_ampa),
    "csd_syn_nmda_":     (sources_syn_nmda,   I_syn_nmda),
}

def csd_dict_for_subset(gid_subset):
    return {
        name: csd_for_subset(sources, I, gid_subset)
        for name, (sources, I) in source_pairs.items()
    }

csd_spiking = csd_dict_for_subset(spiking_gids)
csd_nonspiking = csd_dict_for_subset(non_spiking_gids) 
# there's only one non-spiking L5 pyramidal cell (gid = 269)


resid = csd_from_sources_ - (csd_spiking['csd_from_sources_'] + csd_nonspiking['csd_from_sources_'])
print(np.max(resid))
print(np.min(resid))


fig_spiking = plott.make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_from_sources_=csd_spiking["csd_from_sources_"],
    csd_cap_=csd_spiking["csd_cap_"],
    csd_ionic_=csd_spiking["csd_ionic_"],
    csd_syn_=csd_spiking["csd_syn_"],
    csd_syn_gabab_=csd_spiking["csd_syn_gabab_"],
    csd_syn_gabaa_=csd_spiking["csd_syn_gabaa_"],
    csd_syn_ampa_=csd_spiking["csd_syn_ampa_"],
    csd_syn_nmda_=csd_spiking["csd_syn_nmda_"],
    vmax_row0=60, vmax_row1=10,
    suptitle="Spiking pyramidal cells (182/200)",
    cell_response=net.cell_response,
    overlay_raster=False,
)

fig_nonspiking = plott.make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_from_sources_=csd_nonspiking["csd_from_sources_"],
    csd_cap_=csd_nonspiking["csd_cap_"],
    csd_ionic_=csd_nonspiking["csd_ionic_"],
    csd_syn_=csd_nonspiking["csd_syn_"],
    csd_syn_gabab_=csd_nonspiking["csd_syn_gabab_"],
    csd_syn_gabaa_=csd_nonspiking["csd_syn_gabaa_"],
    csd_syn_ampa_=csd_nonspiking["csd_syn_ampa_"],
    csd_syn_nmda_=csd_nonspiking["csd_syn_nmda_"],
    vmax_row0=20, vmax_row1=5,
    suptitle="Non-spiking pyramidal cells (18/200)",
)




#################
# WITHOUT APICAL OBLIQUE
#################
sources_agg, I_agg = tme.collect_intrinsic_sources(
    net, trial_idx=0,
    cell_types=["L2_pyramidal", "L5_pyramidal"],
    channels=["agg_i_mem"],
)

keep_sections = {src.section for src in sources_agg} - {"apical_oblique"}

sources_agg_no_oblique, I_agg_no_oblique = tme.filter_sources(
    sources_agg, I_agg, sections=keep_sections
)
lfp_agg_no_oblique_, csd_agg_no_oblique_ = compute_lfp_csd_for_sources(net, sources_agg_no_oblique, I_agg_no_oblique)


fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_agg_no_oblique_, 
    csd_agg_no_oblique_, 
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
    unit_csd="µA/mm³",
    overlay_raster_on_csd=True,
    sink="r")
fig.suptitle("LFP/CSD from agg_i_mem, excluding apical obliques")





sources_syn_no_oblique, I_syn_no_oblique = tme.filter_sources(
    sources_syn, I_syn, sections=keep_sections
)
lfp_syn_no_oblique_, csd_syn_no_oblique_ = compute_lfp_csd_for_sources(net, sources_syn_no_oblique, I_syn_no_oblique)

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_syn_no_oblique_, 
    csd_syn_no_oblique_, 
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
    unit_csd="µA/mm³",
    sink="r")
fig.suptitle("LFP/CSD syn, excluding apical obliques")




sources_ion_no_oblique, I_ion_no_oblique = tme.filter_sources(
    sources_ionic, I_ionic, sections=keep_sections
)
lfp_ion_no_oblique_, csd_ion_no_oblique_ = compute_lfp_csd_for_sources(net, sources_ion_no_oblique, I_ion_no_oblique)

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_ion_no_oblique_, 
    csd_ion_no_oblique_, 
    contact_labels, 
    net, 
    ext_inputs=net.cell_response, 
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']}, 
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-60,
    vmax=60,
    figsize=(18, 6),
    overlay_csd_traces=False,
    unit_csd="µA/mm³",
    sink="r")
fig.suptitle("LFP/CSD ionic, excluding apical obliques")


sources_cap_no_oblique, I_cap_no_oblique = tme.filter_sources(
    sources_cap, I_cap, sections=keep_sections
)
lfp_cap_no_oblique_, csd_cap_no_oblique_ = compute_lfp_csd_for_sources(net, sources_cap_no_oblique, I_cap_no_oblique)

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_cap_no_oblique_, 
    csd_cap_no_oblique_, 
    contact_labels, 
    net, 
    ext_inputs=net.cell_response, 
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']}, 
    scale_lfp=3.0,
    voltage_scalebar=200,
    vmin=-60,
    vmax=60,
    figsize=(18, 6),
    overlay_csd_traces=False,
    unit_csd="µA/mm³",
    sink="r")
fig.suptitle("LFP/CSD ionic, excluding apical obliques")



sources_ampa_no_oblique, I_ampa_no_oblique = tme.filter_sources(
    sources_syn_ampa, I_syn_ampa, sections=keep_sections
)
lfp_ampa_no_oblique_, csd_ampa_no_oblique_ = compute_lfp_csd_for_sources(net, sources_ampa_no_oblique, I_ampa_no_oblique)

sources_nmda_no_oblique, I_nmda_no_oblique = tme.filter_sources(
    sources_syn_nmda, I_syn_nmda, sections=keep_sections
)
lfp_nmda_no_oblique_, csd_nmda_no_oblique_ = compute_lfp_csd_for_sources(net, sources_nmda_no_oblique, I_nmda_no_oblique)

sources_gabaa_no_oblique, I_gabaa_no_oblique = tme.filter_sources(
    sources_syn_gabaa, I_syn_gabaa, sections=keep_sections
)
lfp_gabaa_no_oblique_, csd_gabaa_no_oblique_ = compute_lfp_csd_for_sources(net, sources_gabaa_no_oblique, I_gabaa_no_oblique)

sources_gabab_no_oblique, I_gabab_no_oblique = tme.filter_sources(
    sources_syn_gabab, I_syn_gabab, sections=keep_sections
)
lfp_gabab_no_oblique_, csd_gabab_no_oblique_ = compute_lfp_csd_for_sources(net, sources_gabab_no_oblique, I_gabab_no_oblique)

csd_reconstructed_no_oblique_ = csd_syn_no_oblique_ + csd_cap_no_oblique_ + csd_ion_no_oblique_
csd_residual_no_oblique_ = csd_agg_no_oblique_ - csd_reconstructed_no_oblique_
## this is basically the same error we get from the reconstruction taking into account all the sections


'''
fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_cap_no_oblique_,  #### WRONG IN THIS CONTEXT
    csd_ion_no_oblique_,
    #csd_residual_no_oblique_, 
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
    unit_csd="µA/mm³",
    sink="r")
fig.suptitle("ionic")
'''
'''
fig, axes = plt.subplots(1, 6, constrained_layout=True, figsize=(28, 4))

plott.plot_cell_morphology_for_lfp_csd(
    net,
    contact_positions=contact_labels,
    cell_types=('L2_pyramidal', 'L5_pyramidal'),
    ax=axes[0],
    show=False,
)
'''
####

#The following generates two figures to directly compare the full network
# to the one with no apical obliques included in the lfp/csd calculation. 


# ---- Full network ----
fig, axes = plt.subplots(1, 6, constrained_layout=True, figsize=(28, 4),
                         gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.25]})

vmax_row0 = 60
titles_row0 = ["Total (agg_i_mem)", "Capacitive", "Ionic", "Synaptic"]
data_row0 = [csd_from_sources_, csd_cap_, csd_ionic_, csd_syn_]

axes[0].set_title("Morphology")
plott.plot_cell_morphology_for_lfp_csd(
    net, contact_positions=contact_labels,
    cell_types=('L2_pyramidal', 'L5_pyramidal'),
    ax=axes[0], show=False,
)

for ax, data, title in zip(axes[1:5], data_row0, titles_row0):
    plott.plot_laminar_csd_AC(
        times_, data, contact_labels,
        ax=ax, vmin=-vmax_row0, vmax=vmax_row0,
        overlay_csd_traces=True,
        unit_csd="µA/mm³",
        sink="red",
        colorbar=False,
        show=False,
    )
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

axes[5].axis('off')
cax = axes[5].inset_axes([0.35, 0.15, 0.06, 0.25])
cbar = fig.colorbar(axes[1].collections[-1], cax=cax)
cbar.set_label("CSD (µA/mm³)")

fig.suptitle('full network')
plt.show()


# ---- No apical obliques ----
fig, axes = plt.subplots(1, 6, constrained_layout=True, figsize=(28, 4),
                                                  gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.25]})

titles_row0 = ["Total (agg_i_mem)", "Capacitive", "Ionic", "Synaptic"]
data_row0 = [csd_agg_no_oblique_, csd_cap_no_oblique_, csd_ion_no_oblique_, csd_syn_no_oblique_]

axes[0].set_title("Morphology")
plott.plot_cell_morphology_for_lfp_csd(
    net, contact_positions=contact_labels,
    cell_types=('L2_pyramidal', 'L5_pyramidal'),
    ax=axes[0], show=False,
)

for ax, data, title in zip(axes[1:5], data_row0, titles_row0):
    plott.plot_laminar_csd_AC(
        times_, data, contact_labels,
        ax=ax, vmin=-vmax_row0, vmax=vmax_row0,
        overlay_csd_traces=True,
        unit_csd="µA/mm³",
        sink="red",
        colorbar=False,
        show=False,
    )
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

axes[5].axis('off')
cax = axes[5].inset_axes([0.35, 0.15, 0.06, 0.25])
cbar = fig.colorbar(axes[1].collections[-1], cax=cax)
cbar.set_label("CSD (µA/mm³)")

fig.suptitle('no apical obliques')
plt.show()

####



fig_all = plott.make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_agg_no_oblique_, csd_cap_no_oblique_, csd_ion_no_oblique_, csd_syn_no_oblique_,
    csd_gabab_no_oblique_, csd_gabaa_no_oblique_, csd_ampa_no_oblique_, csd_nmda_no_oblique_,
    vmax_row0=60, vmax_row1=10,
    suptitle="no apical oblique",
)
plt.show()



csd_agg_from_apical_obliques = csd_from_sources_ - csd_agg_no_oblique_


csd_agg_from_apical_obliques = csd_from_sources_ - csd_agg_no_oblique_

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_agg_from_apical_obliques,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("CSD contribution from apical obliques (Total − no-oblique)")
plt.show()

csd_ion_from_apical_obliques = csd_ionic_ - csd_ion_no_oblique_

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_ion_from_apical_obliques,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Ionic CSD contribution from apical obliques")
plt.show()

csd_cap_from_apical_obliques = csd_cap_ - csd_cap_no_oblique_

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_cap_from_apical_obliques,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Capacitive CSD contribution from apical obliques")
plt.show()


csd_syn_from_apical_obliques = csd_syn_ - csd_syn_no_oblique_

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_syn_from_apical_obliques,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Synaptic CSD contribution from apical obliques")
plt.show()




# check if results match with keeping apical_oblique only
keep_sections = {"apical_oblique"}

sources_agg_only_oblique, I_agg_only_oblique = tme.filter_sources(
    sources_agg, I_agg, sections=keep_sections
)
lfp_agg_only_oblique_, csd_agg_only_oblique_ = compute_lfp_csd_for_sources(net, sources_agg_only_oblique, I_agg_only_oblique)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_agg_only_oblique_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Synaptic CSD contribution from apical obliques (ground-truth)")
plt.show()


# Look at contributions from other sections
for cell_type in ("L2_pyramidal", "L5_pyramidal"):
    sections = net.cell_types[cell_type]["cell_object"].sections
    print(cell_type, list(sections.keys()))

#apical_trunk
keep_sections = {"apical_trunk"}

sources_agg_only_trunk, I_agg_only_trunk = tme.filter_sources(
    sources_agg, I_agg, sections=keep_sections
)
lfp_agg_only_trunk_, csd_agg_only_trunk_ = compute_lfp_csd_for_sources(net, sources_agg_only_trunk, I_agg_only_trunk)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_agg_only_trunk_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("CSD contribution from apical trunk")
plt.show()

# synaptic should be zero - and it is!
sources_syn_only_trunk, I_syn_only_trunk = tme.filter_sources(
    sources_syn, I_syn, sections=keep_sections
)
lfp_syn_only_trunk_, csd_syn_only_trunk_ = compute_lfp_csd_for_sources(net, sources_syn_only_trunk, I_syn_only_trunk)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_syn_only_trunk_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Synaptic CSD contribution from apical trunk")
plt.show()

# look at ionic currents from apical trunk
sources_ionic_only_trunk, I_ionic_only_trunk = tme.filter_sources(
    sources_ionic, I_ionic, sections=keep_sections
)
lfp_ionic_only_trunk_, csd_ionic_only_trunk_ = compute_lfp_csd_for_sources(net, sources_ionic_only_trunk, I_ionic_only_trunk)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_ionic_only_trunk_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Ionic CSD contribution from apical trunk")
plt.show()


sources_cap_only_trunk, I_cap_only_trunk = tme.filter_sources(
    sources_cap, I_cap, sections=keep_sections
)
lfp_cap_only_trunk_, csd_cap_only_trunk_ = compute_lfp_csd_for_sources(net, sources_cap_only_trunk, I_cap_only_trunk)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_cap_only_trunk_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Capacitive CSD contribution from apical trunk")
plt.show()


# apical_1
#apical_1
keep_sections = {"apical_1"}

sources_agg_only_apical1, I_agg_only_apical1 = tme.filter_sources(
    sources_agg, I_agg, sections=keep_sections
)
lfp_agg_only_apical1_, csd_agg_only_apical1_ = compute_lfp_csd_for_sources(net, sources_agg_only_apical1, I_agg_only_apical1)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_agg_only_apical1_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("CSD contribution from apical_1")
plt.show()

# synaptic should be zero - apical_1 isn't a proximal or distal drive target
sources_syn_only_apical1, I_syn_only_apical1 = tme.filter_sources(
    sources_syn, I_syn, sections=keep_sections
)
lfp_syn_only_apical1_, csd_syn_only_apical1_ = compute_lfp_csd_for_sources(net, sources_syn_only_apical1, I_syn_only_apical1)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_syn_only_apical1_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Synaptic CSD contribution from apical_1")
plt.show()

# look at ionic currents from apical_1
sources_ionic_only_apical1, I_ionic_only_apical1 = tme.filter_sources(
    sources_ionic, I_ionic, sections=keep_sections
)
lfp_ionic_only_apical1_, csd_ionic_only_apical1_ = compute_lfp_csd_for_sources(net, sources_ionic_only_apical1, I_ionic_only_apical1)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_ionic_only_apical1_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Ionic CSD contribution from apical_1")
plt.show()


sources_cap_only_apical1, I_cap_only_apical1 = tme.filter_sources(
    sources_cap, I_cap, sections=keep_sections
)
lfp_cap_only_apical1_, csd_cap_only_apical1_ = compute_lfp_csd_for_sources(net, sources_cap_only_apical1, I_cap_only_apical1)

fig = plott.plot_laminar_csd_AC(
    times_,
    csd_cap_only_apical1_,
    contact_labels,
    overlay_csd_traces=True,
    unit_csd="µA/mm³",
    sink="red",
)
fig.suptitle("Capacitive CSD contribution from apical_1")
plt.show()






######
