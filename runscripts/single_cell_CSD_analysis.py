import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_dipole_lfp_csd as tme
#import tm_currents_utils_forLFP as tme_lfp
from hnn_core.extracellular import calculate_csd2d
import Plotting_tools as plott
import matplotlib.pyplot as plt
import os

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
'''
# load from here:
#with open('runscripts/data/sim_results_dt0025.pkl', 'rb') as f:
#with open('runscripts/data/sim_results_dt000625.pkl', 'rb') as f: #from -625 up
#with open('runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc.pkl', 'rb') as f: #from -550 up
#with open('runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_norecconn.pkl', 'rb') as f: #from -550 up
#with open('runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_norecconn_calcium_blocked.pkl', 'rb') as f: #from -550 up
#with open('runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_calcium_blocked.pkl', 'rb') as f: #from -550 up    

#with open('runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_noreccon_allpassive.pkl', 'rb') as f: #from -550 up
with open('runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_noreccon_allpassive_excsoma.pkl', 'rb') as f: #from -550 up
    results = pickle.load(f)
# to run simulation: use Undestand_CSD_contributions_v3.py
'''

pkl_path = 'runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc.pkl'
#pkl_path = 'runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_norecconn_all_passive_excsoma.pkl'
#pkl_path = 'runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_norecconn_all_passive.pkl'
#pkl_path = 'runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_norecconn.pkl'
#pkl_path = 'runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_noreccon_allpassive_newprobe.pkl'
#pkl_path = 'runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_noreccon_allpassive_newprobe2.pkl'


with open(pkl_path, 'rb') as f:
    results = pickle.load(f)

run_tag = os.path.splitext(os.path.basename(pkl_path))[0].replace(
    "sim_results_dt000625_withmembranepot_newelect_calc_", ""
)

FIGURES_DIR = "/Users/annacattani/Documents/HNN/hnn-core/runscripts/Figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

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

'''
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
'''

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
'''
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
'''
# --- CSD directly from synaptic current sources ---
vmax_syn = np.max(np.abs(csd_syn_))
scale_csd_syn = 0.5 * np.diff(contact_labels)[0] / vmax_syn
'''
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
'''
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

#csd_ionic_from_lfp = calculate_csd2d(lfp_ionic, delta=delta)
#csd_ionic_from_lfp_ = csd_ionic_from_lfp[:, 1:]

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

#csd_syn_gabab_from_lfp = calculate_csd2d(lfp_syn_gabab, delta=delta)
#csd_syn_gabab_from_lfp_ = csd_syn_gabab_from_lfp[:, 1:]
  

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

#csd_syn_gabaa_from_lfp = calculate_csd2d(lfp_syn_gabaa, delta=delta)
#csd_syn_gabaa_from_lfp_ = csd_syn_gabaa_from_lfp[:, 1:]

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
'''
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
'''

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

vmax_cap = np.percentile(np.abs(csd_cap_cell_), 99)

plott.plot_laminar_csd_AC(
    times_,
    csd_cap_cell_,
    contact_labels,
    ax=ax,
    vmin=-vmax_cap,
    vmax=vmax_cap,
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

fig.savefig(os.path.join(FIGURES_DIR, f"capacitive_csd_vm_{run_tag}.png"), dpi=150)
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



I_agg_cell_sum_seg = np.sum(I_agg_cell, axis=0)
plt.figure()
plt.plot(times_, I_agg_cell_sum_seg[1:], label="agg_i_mem")
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Current (nA)")
plt.title(f'Total membrane current summed over segments for gid={example_gid}')

#I_cap_ionic_syn_sum_seg = np.sum(I_cap, axis=0) + np.sum(I_ionic, axis=0) + np.sum(I_syn, axis=0)

I_cap_ionic_syn_cell_sum_seg = np.sum(I_cap_cell, axis=0) + np.sum(I_ionic_cell, axis=0) + np.sum(I_syn_cell, axis=0)
plt.figure()
plt.plot(times_, I_cap_ionic_syn_cell_sum_seg[1:], label="cap + ionic + syn")
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Current (nA)")
#plt.title(f'Total membrane current summed over segments for gid={example_gid}')





csd_agg_cell_ = csd_for_cell(sources_agg_cell, I_agg_cell)
csd_ionic_cell_ = csd_for_cell(sources_ionic_cell, I_ionic_cell)
csd_syn_cell_ = csd_for_cell(sources_syn_cell, I_syn_cell)
# csd_cap_cell_ was already computed earlier

# --- LFP, zoomed into the windows around this cell's external-drive inputs ---
# (same per-gid windows used elsewhere in this script for the CSD/calcium trace figures)
if example_gid == 226:
    lfp_windows = [(25, 50), (60, 85), (130, 155)]
elif example_gid == 207:
    lfp_windows = [(60, 85), (135, 160)]
else:
    lfp_windows = [(times_[0], times_[-1])]  # fall back to the full trace

T_agg_cell = tme.build_transfer_resistance_matrix_for_sources(
    net, sources_agg_cell, array_name="probe1",
)
lfp_agg_cell = tme.reconstruct_lfp_from_sources(T_agg_cell, I_agg_cell)
lfp_agg_cell_ = lfp_agg_cell[:, 1:]  # drop first sample, matches times_



# slab (electrode-bin) boundaries used by build_binning_matrix_for_sources
z_edges = tme._z_edges_from_array(net, "probe1")

fig = plt.figure(figsize=(18, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 5, width_ratios=[1, 4, 4, 4, 4], height_ratios=[2.5, 2])
ax_morph = fig.add_subplot(gs[0, 0])
ax_total = fig.add_subplot(gs[0, 1])
ax_cap = fig.add_subplot(gs[0, 2])
ax_ionic = fig.add_subplot(gs[0, 3])
ax_syn = fig.add_subplot(gs[0, 4])

gs_lfp = gs[1, :].subgridspec(1, len(lfp_windows))
axes_lfp = [fig.add_subplot(gs_lfp[0, j]) for j in range(len(lfp_windows))]
for i, (ax_lfp, (t0, t1)) in enumerate(zip(axes_lfp, lfp_windows)):
    plott.plot_laminar_lfp_AC(
        times_, lfp_agg_cell_, contact_labels,
        ax=ax_lfp, scale=10.0, voltage_scalebar=None, show=False,
    )
    ax_r_lfp = ax_lfp.figure.axes[-1]  # the twinx "HNN z" axis the function just created

    # thin out the depth tick labels -- one per contact is unreadable in a short panel
    tick_step = 4
    for a in (ax_lfp, ax_r_lfp):
        yt = a.get_yticks()
        ytl = [t.get_text() for t in a.get_yticklabels()]
        a.set_yticks(yt[::tick_step])
        a.set_yticklabels(ytl[::tick_step])

    ax_lfp.set_xlim(t0, t1)
    ax_lfp.set_title(f"LFP, t = {t0}-{t1} ms", fontsize=9)
    if i > 0:
        ax_lfp.set_ylabel('')
        ax_lfp.tick_params(axis='y', labelleft=False)
        ax_r_lfp.set_ylabel('')
        ax_r_lfp.set_yticklabels([])

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

vmax_panels = np.percentile(
    np.abs(np.concatenate([csd_agg_cell_, csd_cap_cell_, csd_ionic_cell_, csd_syn_cell_])),
    99,
)

for i, (ax, csd_, title) in enumerate(panels):
    vmax = vmax_panels
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
fig.savefig(os.path.join(FIGURES_DIR, f"csd_panels_total_cap_ionic_syn_{run_tag}.png"), dpi=150)
plt.show()

# --- Progressive cell-addition CSD figure ---
cell_type = 'L5_pyramidal'
l5_gids_all = sorted(net.gid_ranges[cell_type])
ordered_gids = [226] + [g for g in l5_gids_all if g != 226]
n_list = [1, 2, 5, 10, 20, 40, 70, len(ordered_gids)]
sources_agg_all, I_agg_all = tme.collect_intrinsic_sources(
    net, trial_idx=0, cell_types=[cell_type], channels=["agg_i_mem"],
)
def csd_for_gids(gids):
    sources_g, I_g = tme.filter_sources(sources_agg_all, I_agg_all, gid_subset=gids)
    B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sources_g, array_name="probe1")
    csd = tme.compute_csd_from_sources(B, I_g, V_bin)
    return csd[:, 1:]  # drop first sample, matches times_
fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True, sharex=True, sharey=True)
axes = axes.ravel()
for ax, n in zip(axes, n_list):
    gids_subset = ordered_gids[:n]
    csd_n = csd_for_gids(gids_subset)
    vmax_n = np.percentile(np.abs(csd_n), 99)  # per-panel scaling -- shows shape, not raw magnitude
    plott.plot_laminar_csd_AC(
        times_, csd_n, contact_labels,
        ax=ax, vmin=-vmax_n, vmax=vmax_n,
        overlay_csd_traces=False, unit_csd="µA/mm³", sink="red", show=False,
        colorbar=True,
    )
    ax.set_title(f"N = {n} cell{'s' if n > 1 else ''}")
fig.suptitle("CSD as cells are progressively added (each panel independently scaled)", fontsize=13)
fig.savefig(os.path.join(FIGURES_DIR, f"csd_progressive_ncells_{run_tag}.png"), dpi=150)
plt.show()


# --- Standalone figure: this cell's LFP as seen by the close probe (probe2) ---
array_name = "probe2"

# contact z-positions for probe2 (same z-grid as probe1 if you reused `depths` for z)
contact_pos2 = np.array(net.rec_arrays[array_name].positions)
depths2 = contact_pos2[:, 2]
contact_labels2 = np.asarray(depths2, dtype=int)

# forward-reconstruct this cell's LFP contribution onto probe2
T_agg_cell2 = tme.build_transfer_resistance_matrix_for_sources(
    net, sources_agg_cell, array_name=array_name,
)
lfp_agg_cell2 = tme.reconstruct_lfp_from_sources(T_agg_cell2, I_agg_cell)
lfp_agg_cell2_ = lfp_agg_cell2[:, 1:]  # drop first sample, matches times_

# --- locate the external-drive synapses on this cell, for the morphology panel ---
gid = example_gid
cell_type = 'L5_pyramidal'
template = net.cell_types[cell_type]['cell_object']
sect_loc = template.sect_loc  # e.g. {'proximal': [...], 'distal': [...]}

soma_pos = np.asarray(net.pos_dict[cell_type][gid - net.gid_ranges[cell_type][0]], dtype=float)
x_off = soma_pos[0]  # matches center_x=True inside _draw_cell_morphology

drive_names = set(net.external_drives.keys())
syn_sections = set()
for conn in net.connectivity:
    if conn['target_type'] != cell_type:
        continue
    if conn['src_type'] not in drive_names:
        continue
    targets = set()
    for tgt_list in conn['gid_pairs'].values():
        targets.update(tgt_list)
    if gid not in targets:
        continue
    loc = conn['loc']  # 'proximal' or 'distal'
    syn_sections.update(sect_loc[loc])

syn_xz = []
for name in syn_sections:
    sec = template.sections[name]
    pts = np.asarray(sec._end_pts, dtype=float) + soma_pos
    x_mid = 0.5 * (pts[0, 0] + pts[1, 0]) - x_off
    z_mid = 0.5 * (pts[0, 2] + pts[1, 2])
    syn_xz.append((x_mid, z_mid))
syn_xz = np.array(syn_xz)

# --- figure ---
'''
fig2 = plt.figure(figsize=(5 * len(lfp_windows) + 3, 6), constrained_layout=True)
gs2 = fig2.add_gridspec(1, len(lfp_windows) + 1, width_ratios=[1] + [4] * len(lfp_windows))

ax_morph2 = fig2.add_subplot(gs2[0, 0])
axes_lfp2 = [fig2.add_subplot(gs2[0, j + 1]) for j in range(len(lfp_windows))]

plott.plot_cell_morphology_for_lfp_csd(
    net, contact_positions=contact_labels2,
    cell_types=('L5_pyramidal',), gid=example_gid,
    ax=ax_morph2, show=False,
)

# hide the secondary "HNN depth" axis on the right, keep every number on the left
ax_hnn2 = ax_morph2.child_axes[-1]
ax_hnn2.set_ylabel('')
ax_hnn2.set_yticklabels([])
ax_hnn2.tick_params(axis='y', length=0)

ax_morph2.scatter(
    syn_xz[:, 0], syn_xz[:, 1],
    marker='x', s=80, color='red', linewidths=2, zorder=5,
    label='external-drive synapse',
)
ax_morph2.legend(loc='upper right', fontsize=7)

for i, (ax_lfp, (t0, t1)) in enumerate(zip(axes_lfp2, lfp_windows)):
    plott.plot_laminar_lfp_AC(
        times_, lfp_agg_cell2_, contact_labels2,
        ax=ax_lfp, scale=10.0, voltage_scalebar=None, show=False,
    )
    ax_r_lfp = ax_lfp.figure.axes[-1]  # twinx "HNN z" axis

    # thin out the left-axis depth ticks -- one per contact is unreadable in a short panel
    tick_step = 1
    yt = ax_lfp.get_yticks()
    ytl = [t.get_text() for t in ax_lfp.get_yticklabels()]
    ax_lfp.set_yticks(yt[::tick_step])
    ax_lfp.set_yticklabels(ytl[::tick_step])

    # hide the right-hand "HNN z" axis entirely
    ax_r_lfp.set_ylabel('')
    ax_r_lfp.set_yticklabels([])
    ax_r_lfp.tick_params(axis='y', length=0)

    ax_lfp.set_xlim(t0, t1)
    ax_lfp.set_title(f"LFP (probe2), t = {t0}-{t1} ms", fontsize=9)

fig2.suptitle(f"gid={example_gid}: LFP at close probe (probe2)", fontsize=11)
fig2.savefig(os.path.join(FIGURES_DIR, f"lfp_probe2_{run_tag}.png"), dpi=150)
plt.show()
'''

# another version:
# --- Standalone figure: this cell's LFP as seen by probe3 ---
array_name = "probe3"

contact_pos3 = np.array(net.rec_arrays[array_name].positions)
depths3 = contact_pos3[:, 2]
contact_labels3 = np.asarray(depths3, dtype=int)

T_agg_cell3 = tme.build_transfer_resistance_matrix_for_sources(
    net, sources_agg_cell, array_name=array_name,
)
lfp_agg_cell3 = tme.reconstruct_lfp_from_sources(T_agg_cell3, I_agg_cell)
lfp_agg_cell3_ = lfp_agg_cell3[:, 1:]

# --- synapse locations (probe-independent) ---
gid = example_gid
cell_type = 'L5_pyramidal'
template = net.cell_types[cell_type]['cell_object']
sect_loc = template.sect_loc

soma_pos = np.asarray(net.pos_dict[cell_type][gid - net.gid_ranges[cell_type][0]], dtype=float)
x_off = soma_pos[0]

drive_names = set(net.external_drives.keys())
syn_sections = set()
for conn in net.connectivity:
    if conn['target_type'] != cell_type:
        continue
    if conn['src_type'] not in drive_names:
        continue
    targets = set()
    for tgt_list in conn['gid_pairs'].values():
        targets.update(tgt_list)
    if gid not in targets:
        continue
    loc = conn['loc']
    syn_sections.update(sect_loc[loc])

syn_xz = []
for name in syn_sections:
    sec = template.sections[name]
    pts = np.asarray(sec._end_pts, dtype=float) + soma_pos
    x_mid = 0.5 * (pts[0, 0] + pts[1, 0]) - x_off
    z_mid = 0.5 * (pts[0, 2] + pts[1, 2])
    syn_xz.append((x_mid, z_mid))
syn_xz = np.array(syn_xz)

# --- figure ---
fig3 = plt.figure(figsize=(5 * len(lfp_windows) + 3, 6), constrained_layout=True)
gs3 = fig3.add_gridspec(1, len(lfp_windows) + 1, width_ratios=[1] + [4] * len(lfp_windows))

ax_morph3 = fig3.add_subplot(gs3[0, 0])
axes_lfp3 = [fig3.add_subplot(gs3[0, j + 1]) for j in range(len(lfp_windows))]

plott.plot_cell_morphology_for_lfp_csd(
    net, contact_positions=contact_labels3,
    cell_types=('L5_pyramidal',), gid=example_gid,
    ax=ax_morph3, show=False,
)

ax_morph3.scatter(
    syn_xz[:, 0], syn_xz[:, 1],
    marker='x', s=80, color='red', linewidths=2, zorder=5,
    label='external-drive synapse',
)
ax_morph3.legend(loc='upper right', fontsize=7)

for i, (ax_lfp, (t0, t1)) in enumerate(zip(axes_lfp3, lfp_windows)):
    plott.plot_laminar_lfp_AC(
        times_, lfp_agg_cell3_, contact_labels3,
        ax=ax_lfp, scale=10.0, voltage_scalebar=None, show=False,
    )
    ax_r_lfp = ax_lfp.figure.axes[-1]

    tick_step = 1
    yt = ax_lfp.get_yticks()
    ytl = [t.get_text() for t in ax_lfp.get_yticklabels()]
    ax_lfp.set_yticks(yt[::tick_step])
    ax_lfp.set_yticklabels(ytl[::tick_step])

    ax_r_lfp.set_ylabel('')
    ax_r_lfp.set_yticklabels([])
    ax_r_lfp.tick_params(axis='y', length=0)

    ax_lfp.set_xlim(t0, t1)
    ax_lfp.set_title(f"LFP (probe3), t = {t0}-{t1} ms", fontsize=9)

fig3.suptitle(f"gid={example_gid}: LFP at probe3", fontsize=11)
fig3.savefig(os.path.join(FIGURES_DIR, f"lfp_probe3_{run_tag}.png"), dpi=150)
plt.show()

def get_segment_positions(net, sources, center_on_soma=True):
    """(x, y, z) midpoint of each source's segment.

    If center_on_soma=True (default), x is centered on each source's own soma
    position — matching the coordinate convention _draw_cell_morphology uses —
    so the scatter overlays correctly on top of the morphology drawing.
    """
    template_cells = {
        ct: net.cell_types[ct]["cell_object"]
        for ct in sorted({src.cell_type for src in sources})
    }
    positions = np.zeros((len(sources), 3))
    for i, src in enumerate(sources):
        sec_start, sec_end, _, _ = tme._get_global_section_geometry(net, template_cells, src)
        positions[i] = sec_start + src.segment_x * (sec_end - sec_start)

    if center_on_soma:
        for i, src in enumerate(sources):
            start_gid = net.gid_ranges[src.cell_type][0]
            soma_pos = np.asarray(net.pos_dict[src.cell_type][src.gid - start_gid])
            positions[i, 0] -= soma_pos[0]

    return positions


seg_pos = get_segment_positions(net, sources_agg_cell)  # recompute with the fix


# PARENTHESIS MOVIE OPENS HERE (we want to understand if sinks/sources distribute as in Einevoll's paper)
# this should be used in a network with no recurrent connectivity
import matplotlib.animation as animation
from matplotlib.colors import SymLogNorm

# auto-scale to the actual current range in this passive simulation
vmax = np.percentile(np.abs(I_agg_cell), 99.5)
vmin = -vmax
linthresh = vmax / 50
norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, base=10)

s_min, s_max = 15, 200
size_denom = np.max(np.abs(I_agg_cell))

def sizes_for(I_vals):
    return s_min + (s_max - s_min) * (np.abs(I_vals) / size_denom)

# somatic membrane potential, for the new panel below the morphology
soma_vm = np.array(net.cell_response.vsec[0][example_gid]['soma'])

fig, (ax, ax_vm) = plt.subplots(
    2, 1, figsize=(5, 9), gridspec_kw={'height_ratios': [5, 1]},
    constrained_layout=True,
)

plott._draw_cell_morphology(
    ax, net, cell_type="L5_pyramidal", gid=example_gid, color_by_region=False,
)

sc = ax.scatter(
    seg_pos[:, 0], seg_pos[:, 2],
    c=I_agg_cell[:, 1], cmap="jet_r", norm=norm,
    s=sizes_for(I_agg_cell[:, 1]), edgecolor='k', linewidth=0.3, zorder=5,
)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Transmembrane current (nA)")
ax.set_xlabel("x (µm)")
ax.set_ylabel("z (µm)")
title = ax.set_title(f"agg_i_mem spatial distribution, gid={example_gid}\nt = {times[1]:.2f} ms")

# --- bottom-left box: running sum of sources / sinks / total ---
annot = ax.text(
    0.02, 0.02, "", transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
    family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
)

def _annot_text(I_t):
    I_pos_sum = I_t[I_t > 0].sum()
    I_neg_sum = I_t[I_t < 0].sum()
    return (
        f"Σ sources (+): {I_pos_sum:.4f} nA\n"
        f"Σ sinks (−):   {I_neg_sum:.4f} nA\n"
        f"Σ total:       {I_pos_sum + I_neg_sum:.2e} nA"
    )

annot.set_text(_annot_text(I_agg_cell[:, 1]))  # initialize for the first frame

# --- bottom panel: somatic Vm trace with a moving marker for the current frame ---
ax_vm.plot(times, soma_vm, color='indigo', lw=1)
ax_vm.set_xlabel("Time (ms)")
ax_vm.set_ylabel("$V_m$ soma (mV)")
ax_vm.set_xlim(times[0], times[-1])
vline = ax_vm.axvline(times[1], color='k', lw=1.2)

frame_step = 4
frame_idx = np.arange(1, len(times), frame_step)  # start at 1, skip the first time step

def update(i):
    t_idx = frame_idx[i]
    I_t = I_agg_cell[:, t_idx]
    sc.set_array(I_t)
    sc.set_sizes(sizes_for(I_t))
    title.set_text(f"agg_i_mem spatial distribution, gid={example_gid}\nt = {times[t_idx]:.2f} ms")
    annot.set_text(_annot_text(I_t))
    vline.set_xdata([times[t_idx], times[t_idx]])
    return sc, title, annot, vline

ani = animation.FuncAnimation(fig, update, frames=len(frame_idx), interval=50, blit=False)

FIGURES_DIR = "/Users/annacattani/Documents/HNN/hnn-core/runscripts/Figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

save_path = os.path.join(FIGURES_DIR, f"agg_i_mem_movie_{run_tag}.mp4")
#save_path = os.path.join(FIGURES_DIR, "agg_i_mem_movie_all_passive.mp4")
print(f"Saving to: {save_path}")

try:
    ani.save(save_path, writer="ffmpeg", fps=20)
    print("ani.save() completed without raising an exception.")
except Exception as e:
    print(f"ani.save() raised: {type(e).__name__}: {e}")

print(f"File exists at save_path? {os.path.exists(save_path)}")
if os.path.exists(save_path):
    print(f"File size: {os.path.getsize(save_path)} bytes")

def render_at_time(t_query):
    t_idx = int(np.argmin(np.abs(times - t_query)))
    I_t = I_agg_cell[:, t_idx]
    sc.set_array(I_t)
    sc.set_sizes(sizes_for(I_t))
    title.set_text(f"agg_i_mem spatial distribution, gid={example_gid}\nt = {times[t_idx]:.2f} ms")
    annot.set_text(_annot_text(I_t))
    vline.set_xdata([times[t_idx], times[t_idx]])
    return t_idx, times[t_idx]

snapshot_times = [25.33, 25.73, 30.13, 32.23, 40, 50.62, 67.42, 72.33, 138.12, 140.92, 147]

for t_query in snapshot_times:
    t_idx, t_actual = render_at_time(t_query)
    fname = f"agg_i_mem_movie_{run_tag}_t{t_actual:.2f}ms.png"
    #fname = f"agg_i_mem_movie_all_passive_t{t_actual:.2f}ms.png"
    save_path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}  (requested t={t_query}, actual t={t_actual:.2f} ms)")

# PARENTHESIS MOVIE CLOSES HERE



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


def make_csd_traces_figure(xlim, scale_mult=2.0, syn_scale_mult=2.5):
    mask = (times_ >= xlim[0]) & (times_ <= xlim[1])

    # ONE shared scale, computed once from capacitive, ionic, excitatory, and inhibitory combined
    all_csd = [csd_cap_cell_, csd_ionic_cell_, csd_syn_exc_cell_, csd_syn_inh_cell_]
    vmax = max(np.max(np.abs(csd_[:, mask])) for csd_ in all_csd)
    scale = scale_mult * np.diff(contact_labels)[0] / vmax

    def plot_csd_traces(ax, csd_list, title, panel_scale_mult=1.0):
        """csd_list: list of (csd_array, color) tuples, all sharing the same `scale`,
        optionally boosted per-panel via panel_scale_mult."""
        for csd_, color in csd_list:
            tme.plot_stacked_traces(ax, times_, csd_, depths=contact_labels, color=color,
                                     scale=-scale * panel_scale_mult)
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
                     f"Synaptic CSD — gid={example_gid}", panel_scale_mult=syn_scale_mult)

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
'''
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
'''
# these are good for cell 226
if example_gid == 226:
    make_csd_traces_figure((25, 50), scale_mult=2.0, syn_scale_mult=5)
    make_csd_traces_figure((60, 85), scale_mult=2.0, syn_scale_mult=5)
    make_csd_traces_figure((130, 155), scale_mult=2.0, syn_scale_mult=5)
elif example_gid == 207:
    make_csd_traces_figure((60, 85))
    make_csd_traces_figure((135, 160))
    make_csd_traces_figure((60,85), scale_mult=2.0, syn_scale_mult=5)
    make_csd_traces_figure((135, 160), scale_mult=2.0, syn_scale_mult=5)



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

sections_to_plot = ['apical_tuft', 'apical_oblique', 'basal_2', 'soma']

def make_syn_vm_figure(xlim=None):
    fig, axes = plt.subplots(len(sections_to_plot), 1, figsize=(10, 8),
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

sections_to_plot = ['apical_tuft', 'apical_oblique', 'basal_2', 'soma']

def make_syn_vm_figure(xlim=None):
    fig, axes = plt.subplots(len(sections_to_plot), 1, figsize=(10, 8),
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

# these are good for cell 226
if example_gid == 226:
    make_ca_traces_figure(xlim=(25, 50))
    make_ca_traces_figure(xlim=(60, 85))
    make_ca_traces_figure(xlim=(130, 155))
elif example_gid == 207:
    make_ca_traces_figure((60, 85))
    make_ca_traces_figure((135, 160))

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

Vm      = {c: np.asarray(net.cell_response.vsec[0][example_gid][c])[1:] for c in compartments}
I_Na    = {c: mid_segment_density('ina_hh2', example_gid, c)[1:] for c in compartments}
I_K     = {c: mid_segment_density('ik_hh2', example_gid, c)[1:] for c in compartments}
I_KM    = {c: mid_segment_density('ik_km', example_gid, c)[1:] for c in compartments}
I_KCa   = {c: mid_segment_density('ik_kca', example_gid, c)[1:] for c in compartments}
I_Ca    = {c: mid_segment_density('ica_ca', example_gid, c)[1:] for c in compartments}
I_CaT   = {c: mid_segment_density('ica_cat', example_gid, c)[1:] for c in compartments}
I_Ih    = {c: mid_segment_density('i_ar', example_gid, c)[1:] for c in compartments}
I_Leak  = {c: mid_segment_density('il_hh2', example_gid, c)[1:] for c in compartments}
I_AMPA  = {c: get_syn_current(example_gid, c, 'ampa')[1:] for c in compartments}
I_NMDA  = {c: get_syn_current(example_gid, c, 'nmda')[1:] for c in compartments}
I_GABAA = {c: get_syn_current(example_gid, c, 'gabaa')[1:] for c in compartments}
I_GABAB = {c: get_syn_current(example_gid, c, 'gabab')[1:] for c in compartments}

panels = [
    ('$V_m$ (mV)', Vm),
    ('$I_{Na}$ (mA/cm²)', I_Na),
    ('$I_{K}$ (mA/cm²)', I_K),
    ('$I_{KM}$ (mA/cm²)', I_KM),
    ('$I_{KCa}$ (mA/cm²)', I_KCa),
    ('$I_{Ca}$ (mA/cm²)', I_Ca),
    ('$I_{CaT}$ (mA/cm²)', I_CaT),
    ('$I_{h}$ (mA/cm²)', I_Ih),
    ('$I_{leak}$ (mA/cm²)', I_Leak),
    ('$I_{AMPA}$ (nA)', I_AMPA),
    ('$I_{NMDA}$ (nA)', I_NMDA),
    ('$I_{GABA_A}$ (nA)', I_GABAA),
    ('$I_{GABA_B}$ (nA)', I_GABAB),
]

FIGURES_DIR = "/Users/annacattani/Documents/HNN/hnn-core/runscripts/Figures"

def make_figure(gid, xlims, save_path=None):
    """One column per time window in `xlims`, each column stacking all panels."""
    nrows = len(panels)
    ncols = len(xlims)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 0.9 * nrows + 1.2),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for col, xlim in enumerate(xlims):
        for row, (ylabel, data) in enumerate(panels):
            ax = axes[row, col]
            ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
            for c, color in compartments.items():
                ax.plot(times_, data[c], color=color, lw=1.3, label=c)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            ax.yaxis.get_offset_text().set_fontsize(7)
            ax.set_xlim(*xlim)

            is_last_row = (row == nrows - 1)
            ax.tick_params(axis='x', labelbottom=is_last_row)
            if is_last_row:
                ax.set_xlabel('Time (ms)', fontsize=10)

        axes[0, col].set_title(f"{xlim[0]}\u2013{xlim[1]} ms", fontsize=11)

    axes[0, 0].legend(fontsize=7, loc='upper right')
    fig.align_ylabels(axes[:, 0])
    fig.suptitle(f'Soma vs. apical_tuft — gid={gid}', fontsize=13)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

if example_gid == 226:
    make_figure(example_gid, [(25, 50), (60, 85), (135, 160)],
                save_path=f"{FIGURES_DIR}/intrinsic_currents_gid{example_gid}.png")
elif example_gid == 207:
    make_figure(example_gid, [(60, 85), (140, 165)],
                save_path=f"{FIGURES_DIR}/intrinsic_currents_gid{example_gid}.png")