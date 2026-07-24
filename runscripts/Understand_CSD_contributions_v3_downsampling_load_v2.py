#dpl_gabaa = tme.reconstruct_dipole_from_sources(net, sources_syn_GABAA, I_syn_GABAA)import matplotlib.pyplot as plt
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
with open('runscripts/data/sim_results_dt000625.pkl', 'rb') as f:
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
tme.downsample_currents(net, step=step, include_isec=True)
times = times[::step]
lfp_hnn = lfp_hnn[:,::step]

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


#######################
# The following calculates the LFP from agg_i_mem and from components and asseses the difference between the two
#######################

#######
# From synaptic currents only
#######

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

#######
# From capacitive currents only
#######

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

#######
# From ionic currents only
#######

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

#######
# From aggregated membrane current
#######

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



########################################
# From here we start focusing on the CSD
########################################

#####################
# Total CSD from agg_i_mem
#####################

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
times_ = times[1:]
lfp_agg_i_mem_ = lfp_agg_i_mem[:, 1:]

'''
plott.plot_lfp_morph_csd(
    times_, 
    lfp_agg_i_mem_, 
    csd_from_sources_, 
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
    sink="red")
'''
'''
# agg_i_mem dipole
dpl_agg_i_mem = tme.reconstruct_dipole_from_sources(net, sources_agg, I_agg)
dpl_agg_i_mem_ = dpl_agg_i_mem[1:]

fig, ax = plt.subplots()
ax.plot(dpl.times, dpl.data['agg'], 'k', label='Total', lw=1.5)
ax.plot(times_, dpl_agg_i_mem_, label='dipole from agg_i_mem sources', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()
'''

###########################
# FROM SYNAPTIC CURRENTS
###########################

sources_syn, I_syn = tme.collect_synaptic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
)

# CSD from synaptic currents only
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
csd_syn_ = csd_syn[:, 1:]

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_syn_, 
    csd_syn_, 
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
    overlay_raster_on_csd=True,
    sink="red")
fig.suptitle("LFP/CSD from synaptic currents")

'''
# dipole from synaptic currents only
dpl_syn = tme.reconstruct_dipole_from_sources(net, sources_syn, I_syn)
dpl_syn_ = dpl_syn[1:]

fig, ax = plt.subplots()
ax.plot(times_, dpl_agg_i_mem_, label='dipole from agg_i_mem sources', lw=1.5)
ax.plot(times_, dpl_syn_, label='dipole from synaptic currents', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()
#fig.suptitle("Dipole from synaptic currents")
'''
'''
# sanity check [SHOULD BE CHECKED MUCH BETTER!]
sources_intr, I_intr = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
    channels=sorted(l5_component_channels),  # excludes agg_i_mem
)
dpl_intr = tme.reconstruct_dipole_from_sources(net, sources_intr, I_intr)
dpl_intr_ = dpl_intr[1:]

fig, ax = plt.subplots()
ax.plot(times_, dpl_agg_i_mem_, label='dipole from agg_i_mem sources', lw=1.5)
ax.plot(times_, dpl_intr_, label='dipole from intrisinc currents', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()

dpl_tot_ = dpl_intr_ + dpl_syn_

fig, ax = plt.subplots()
ax.plot(times_, dpl_agg_i_mem_, label='Total (agg_i_mem)', lw=1.5)
ax.plot(times_, dpl_tot_, label='Total (reconstructed)', lw=1.5)
ax.plot(times_, dpl_intr_, label='Intrinsic (capacitive + ionic)', lw=1.5)
ax.plot(times_, dpl_syn_, label='Synaptic', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()
fig.suptitle("Dipole")
# there's something off in the previous plot
'''
## The whole validation of dipole is going to be done by Dylan


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
  
lfp_syn_gabab_ = lfp_syn_gabab[:, 1:]
csd_syn_gabab_ = csd_syn_gabab[:, 1:]

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_syn_gabab_, 
    csd_syn_gabab_, 
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
    sink="red")
fig.suptitle("LFP/CSD from GABA_B currents")

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

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_syn_gabaa_, 
    csd_syn_gabaa_, 
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
    sink="red")
fig.suptitle("LFP/CSD from GABA_A currents")

'''
# dipole from gabaa only
dpl_gabaa = tme.reconstruct_dipole_from_sources(net, sources_syn_gabaa, I_syn_gabaa)
dpl_gabaa_ = dpl_gabaa[1:]

fig, ax = plt.subplots()
ax.plot(times_, dpl_agg_i_mem_, 'k', label='Total (agg_i_mem)', lw=1.5)
ax.plot(times_, dpl_gabaa_, label='GABA_A', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()
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

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_syn_ampa_, 
    csd_syn_ampa_, 
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
    sink="red")
fig.suptitle("LFP/CSD from AMPA currents")

'''
# dipole from gabaa only
dpl_ampa = tme.reconstruct_dipole_from_sources(net, sources_syn_ampa, I_syn_ampa)
dpl_ampa_ = dpl_ampa[1:]

fig, ax = plt.subplots()
ax.plot(times_, dpl_agg_i_mem_, 'k', label='Total (agg_i_mem)', lw=1.5)
ax.plot(times_, dpl_ampa_, label='AMPA', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()
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

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_syn_nmda_, 
    csd_syn_nmda_, 
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
    sink="red")
fig.suptitle("LFP/CSD from NMDA currents")

'''
# dipole from nmda only
dpl_nmda = tme.reconstruct_dipole_from_sources(net, sources_syn_nmda, I_syn_nmda)
dpl_nmda_ = dpl_nmda[1:]

fig, ax = plt.subplots()
ax.plot(times_, dpl_agg_i_mem_, 'k', label='Total (agg_i_mem)', lw=1.5)
ax.plot(times_, dpl_nmda_, label='NMDA', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()
'''

#net.cell_response.plot_spikes_raster(show=False)

## CHECK: does the sum of the CSD of AMPA, NMDA, GABAA and GABAB currents equal the CSD from all synaptic currents?
total_csd_syn_ = csd_syn_ampa_ + csd_syn_nmda_ + csd_syn_gabaa_ + csd_syn_gabab_
#csd_syn_
csd_syn_residual_ = csd_syn_ - total_csd_syn_
vmax_check = np.max(np.abs(csd_syn_residual_))


###########################
# FROM CAPACITIVE AND IONIC CURRENTS
###########################

sources_intr, I_intr = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
    channels=l5_component_channels
)

# CSD from instrinsic currents only
B_intr, V_bin_intr, z_center_intr = tme.build_binning_matrix_for_sources(
    net,
    sources_intr,
    array_name="probe1"
)

csd_intr = tme.compute_csd_from_sources(
    B_intr,
    I_intr,
    V_bin_intr
)

# LFP from intrinsic currents only
T_intr = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_intr,
    array_name="probe1",
)

lfp_intr = tme.reconstruct_lfp_from_sources(
    T_intr,
    I_intr,
)

lfp_intr_ = lfp_intr[:, 1:]
csd_intr_ = csd_intr[:, 1:]

fig = plott.plot_lfp_morph_csd(
    times_, 
    lfp_intr_, 
    csd_intr_, 
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
    sink="red")
fig.suptitle("LFP/CSD from capacitive and ionic currents")


###########################
# FROM CAPACITIVE CURRENTS only
###########################
sources_cap_, I_cap_ = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal", "L5_pyramidal"],
    channels=["agg_i_cap"],
)

B_cap, V_bin_cap, _ = tme.build_binning_matrix_for_sources(net, sources_cap_, array_name="probe1")
csd_cap = tme.compute_csd_from_sources(B_cap, I_cap_, V_bin_cap)

T_cap = tme.build_transfer_resistance_matrix_for_sources(net, sources_cap_, array_name="probe1")
lfp_cap = tme.reconstruct_lfp_from_sources(T_cap, I_cap_)

lfp_cap_, csd_cap_ = lfp_cap[:, 1:], csd_cap[:, 1:]

fig = plott.plot_lfp_morph_csd(
    times_, lfp_cap_, csd_cap_, contact_labels, net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0, voltage_scalebar=200,
    vmin=-10, vmax=10, figsize=(18, 6),
    overlay_csd_traces=True, unit_csd="µA/mm³", sink="red",
    overlay_raster_on_csd=True,
)
fig.suptitle("LFP/CSD from capacitive currents")


###########################
# FROM IONIC CURRENTS only
###########################
ionic_channels = [ch for ch in l5_component_channels if ch != "agg_i_cap"]

sources_ionic_, I_ionic_ = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal", "L5_pyramidal"],
    channels=ionic_channels,
)

B_ionic, V_bin_ionic, _ = tme.build_binning_matrix_for_sources(net, sources_ionic_, array_name="probe1")
csd_ionic = tme.compute_csd_from_sources(B_ionic, I_ionic_, V_bin_ionic)

T_ionic = tme.build_transfer_resistance_matrix_for_sources(net, sources_ionic_, array_name="probe1")
lfp_ionic = tme.reconstruct_lfp_from_sources(T_ionic, I_ionic_)

lfp_ionic_, csd_ionic_ = lfp_ionic[:, 1:], csd_ionic[:, 1:]

fig = plott.plot_lfp_morph_csd(
    times_, lfp_ionic_, csd_ionic_, contact_labels, net,
    ext_inputs=net.cell_response,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    scale_lfp=3.0, voltage_scalebar=200,
    vmin=-60, vmax=60, figsize=(18, 6),
    overlay_csd_traces=True, unit_csd="µA/mm³", sink="red",
    overlay_raster_on_csd=True
)
fig.suptitle("LFP/CSD from ionic currents")

#################
# FIGURE panel for meetings:
#############
fig_all = plott.make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_from_sources_, csd_cap_, csd_ionic_, csd_syn_,
    csd_syn_gabab_, csd_syn_gabaa_, csd_syn_ampa_, csd_syn_nmda_,
    vmax_row0=60, vmax_row1=10,
    suptitle="Whole network (all pyramidal cells)",
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
    overlay_csd_traces=False,
    unit_csd="µA/mm³",
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
    overlay_csd_traces=False,
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


'''

fig_all = plott.make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_agg_no_oblique_, csd_cap_no_oblique_, csd_ion_no_oblique_, csd_syn_no_oblique_,
    csd_gabab_no_oblique_, csd_gabaa_no_oblique_, csd_ampa_no_oblique_, csd_nmda_no_oblique_,
    vmax_row0=60, vmax_row1=10,
    suptitle="no apical oblique",
)
plt.show()

'''