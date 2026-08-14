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

net = jones_2009_model()
add_erp_drives_to_jones_model(net)
net.set_cell_positions(inplane_distance=30.)
#net.connectivity.clear() #<- clears everything, including drives
#net.clear_connectivity()  # this keeps drives, removes recurrent


# Laminar probe
#depths = np.arange(-625, 2150, 100) what I usually use
#depths = np.arange(-650, 2150, 100) #v1
depths = np.arange(-550, 2150, 100) #v2
#depths = np.arange(-650.5, 2100, 100)
electrode_pos = [(135, 135, z) for z in depths]
net.add_electrode_array('probe1', electrode_pos)

n_trials = 1

if "dpls" not in locals():
    with JoblibBackend(1):
        dpls = simulate_dipole(
            net,
            tstop=170.0,
            dt=0.00625,#0.0125, # 0.025
            n_trials=n_trials,
            record_agg_i_mem="all",   # aggregated total transmembrane current
            # record_agg_ina="all",
            # record_agg_ik="all",
            record_agg_i_cap="all",   # aggregated capacitive current
            record_ina_hh2="all",
            record_ik_hh2="all",
            record_ik_kca="all",
            record_ik_km="all",
            record_ica_ca="all",
            record_ica_cat="all",
            record_il_hh2="all",      # aggregated leak current
            record_i_ar="all",
            record_isec="all",
            record_vsec="all",
            record_ca="all"
        )

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


scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

dpl = dpls[0]
times = net.cell_response.times


def _to_numpy_nested(d):
    """Recursively convert every leaf list in a nested dict to a numpy array, in place."""
    for key, value in d.items():
        if isinstance(value, dict):
            _to_numpy_nested(value)
        else:
            d[key] = np.asarray(value, dtype=float)


def convert_currents_to_numpy(net, channels=None, include_isec=True,
                               include_vsec=True, include_ca=True):
    cr = net.cell_response
    cr._times = np.asarray(cr.times, dtype=float)

    channels = channels or list(cr.transmembrane_currents.keys())
    for channel in channels:
        for trial_data in getattr(cr, f"_{channel}"):
            _to_numpy_nested(trial_data)

    for flag, attr in [(include_isec, "_isec"), (include_vsec, "_vsec"), (include_ca, "_ca")]:
        if flag:
            for trial_data in getattr(cr, attr):
                _to_numpy_nested(trial_data)

convert_currents_to_numpy(net, include_isec=True)

results = {
    "net": net,
    "dpl": dpl,
    "dt": net._dt,
    "scaling_factor": scaling_factor,
    "depths": depths,
    "times": times,
}
#with open("runscripts/data/sim_results_dt0025.pkl", "wb") as f:
#with open("runscripts/data/sim_results_dt000625.pkl", "wb") as f:
with open("runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc.pkl", "wb") as f:
    pickle.dump(results, f)

print(results.keys())
'''
results = {
    "net": net,
    "dpl": dpl,
    "dt": net._dt,
    "scaling_factor": scaling_factor,
    "depths": depths,
    'times': times
}

with open("runscripts/data/sim_results_dt0025.pkl", "wb") as f:
    pickle.dump(results, f)
'''



# load from here:
#with open('runscripts/data/sim_results_dt0025.pkl', 'rb') as f:
with open('runscripts/data/sim_results_dt000625.pkl', 'rb') as f:
    results = pickle.load(f)

dpl = results['dpl']
net = results['net']
#rec_arrays = results['rec_arrays']
times = results['times']
depths = results['depths']


contact_labels = np.asarray(depths, dtype=int)
#lfp = net.rec_arrays["probe1"].voltages[0]  # HNN's LFP; trial 0
#

#lfp = lfp[::4]
#times = times[::4]

def downsample_currents(net, step=2, channels=None, include_isec=False,
                         include_vsec=False, include_ca=False):
    cell_response = net.cell_response

    if channels is None:
        channels = list(cell_response.transmembrane_currents.keys())

    # downsample the shared time vector
    cell_response._times = cell_response.times[::step]

    # transmembrane_currents: channel -> trial -> gid -> section -> segment -> values
    for channel in channels:
        channel_data = getattr(cell_response, f"_{channel}")
        for trial_data in channel_data:
            for gid, section_dict in trial_data.items():
                for section, segment_dict in section_dict.items():
                    for segment, values in segment_dict.items():
                        segment_dict[segment] = values[::step]

    ## vsec / ca: trial -> gid -> section -> values
    #for include_flag, attr_name in (
    #    (include_vsec, "_vsec"),
    #    (include_ca, "_ca"),
    #):
    #    if not include_flag:
    #        continue
    #    for trial_data in getattr(cell_response, attr_name):
    #        for gid, section_dict in trial_data.items():
    #            for section, values in section_dict.items():
    #                section_dict[section] = values[::step]

    # isec: trial -> gid -> section -> syn_name -> values
    if include_isec:
        for trial_data in cell_response._isec:
            for gid, section_dict in trial_data.items():
                for section, syn_dict in section_dict.items():
                    for syn_name, values in syn_dict.items():
                        syn_dict[syn_name] = values[::step]


# pick one example trace
channel = "agg_i_mem"
gid = list(net.gid_ranges["L5_pyramidal"])[0]
trial_data = net.cell_response.transmembrane_currents[channel][0]  # trial 0
section = list(trial_data[gid].keys())[0]
segment = list(trial_data[gid][section].keys())[0]

example = trial_data[gid][section][segment]
print(len(example))        # length of that trace

step = 4
downsample_currents(net, step=step, include_isec=True)

# pick one example trace
channel = "agg_i_mem"
gid = list(net.gid_ranges["L5_pyramidal"])[0]
trial_data = net.cell_response.transmembrane_currents[channel][0]  # trial 0
section = list(trial_data[gid].keys())[0]
segment = list(trial_data[gid][section].keys())[0]

example = trial_data[gid][section][segment]
print(len(example))        # length of that trace





#######
# From synaptic currents only
#######
'''
sources_syn, I_syn = tme.collect_synaptic_sources(
    net,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
)

sources_syn_soma, I_syn_soma = tme.filter_sources(
    sources_syn,
    I_syn,
    cell_types=["L5_pyramidal"],
    sections=["soma"],
)

# LFP from synaptic currents only
T_syn_soma = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_syn_soma,
    array_name="probe1",
)

lfp_syn_soma = tme.reconstruct_lfp_from_sources(
    T_syn_soma,
    I_syn_soma,
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

sources_cap_soma, I_cap_soma = tme.filter_sources(
    sources_cap,
    I_cap,
    cell_types=["L5_pyramidal"],
    sections=["soma"],
)

T_cap_soma = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_cap_soma,
    array_name="probe1",
)

lfp_cap_soma = tme.reconstruct_lfp_from_sources(
    T_cap_soma,
    I_cap_soma,
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

sources_ionic_soma, I_ionic_soma = tme.filter_sources(
    sources_ionic,
    I_ionic,
    cell_types=["L5_pyramidal"],
    sections=["soma"],
)

T_ionic_soma = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_ionic_soma,
    array_name="probe1",
)

lfp_ionic_soma = tme.reconstruct_lfp_from_sources(
    T_ionic_soma,
    I_ionic_soma,
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

sources_agg_soma, I_agg_soma = tme.filter_sources(
    sources_agg,
    I_agg,
    cell_types=["L5_pyramidal"],
    sections=["soma"],
)

T_agg_agg_soma = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_agg_soma,
    array_name="probe1",
)

lfp_agg_i_mem_soma = tme.reconstruct_lfp_from_sources(
    T_agg_agg_soma,
    I_agg_soma,
)



contact_number = 8  # choose one electrode/contact
plt.figure()
plt.plot(times[1:], lfp_agg_i_mem_soma[contact_number,1:], label="agg_i_mem")
plt.plot(times[1:], lfp_cap_soma[contact_number,1:], "--", label="cap")  
plt.plot(times[1:], lfp_ionic_soma[contact_number,1:], "--", label="ionic")  
plt.plot(times[1:], lfp_syn_soma[contact_number,1:], ":", label="syn")  

plt.plot(times[1:], lfp_cap_soma[contact_number,1:] + lfp_ionic_soma[contact_number,1:] + lfp_syn_soma[contact_number,1:], "--", label="cap + ionic + syn")
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("LFP (µV)")
plt.title('LFP at contact 10 (z=375 µm) - L5 pyramidal cell soma')
'''



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


sources_agg, I_agg = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
    channels=["agg_i_mem"],
)

#######
# From aggregated membrane current
#######
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

# UP TO HERE!!!!

# plot of all the contacts:
# ALL THE TRACES — agg_i_mem vs (cap + ionic + synaptic):
positions = np.asarray(net.rec_arrays["probe1"].positions)
z = positions[:, 2]
n_contacts = lfp_agg_soma.shape[0]

# single global scaling factor (applies to both signals)
gain = 0.005

traces_agg  = lfp_agg_soma * gain
traces_comp = (lfp_intr_soma + lfp_syn_soma) * gain
offset = 1.0   # fixed spacing between traces in visual units

# --- scale bar in µV ---
scale_uV = 200.0
bar_height_visual = scale_uV * gain

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(n_contacts):
    ax.plot(times[1:], traces_agg[c, 1:]  + c * offset,
            color="k", lw=0.8,
            label="agg_i_mem" if c == 0 else None)
    ax.plot(times[1:], traces_comp[c, 1:] + c * offset,
            color="tab:red", lw=0.8, alpha=0.8,
            label="cap + ionic + synaptic" if c == 0 else None)

ax.set_yticks([c * offset for c in range(n_contacts)])
ax.set_yticklabels([f"{int(z[c])} µm" for c in range(n_contacts)], fontsize=9)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth z")
ax.set_title("LFP per contact — agg_i_mem vs cap + ionic + synaptic")

# --- legend (only one entry per signal) ---
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# --- scale bar near the left edge, just below the first trace ---
x_bar = times[1] + (times[-1] - times[1]) * 0.015
y_bar_bottom = -1.0 * offset
ax.plot([x_bar, x_bar],
        [y_bar_bottom, y_bar_bottom + bar_height_visual],
        color="black", lw=4, solid_capstyle="butt")
ax.text(x_bar + (times[-1] - times[1]) * 0.01,
        y_bar_bottom + bar_height_visual / 2,
        f"{scale_uV:g} µV",
        color="black", ha="left", va="center",
        fontsize=11, fontweight="bold")

plt.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.05)



# FIN QUI 

'''
breakpoint()

import copy

window_len, scaling_factor = 30, 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

dpls_smooth = [copy.deepcopy(dpl).smooth(window_len) for dpl in dpls]
dpl_smooth = dpls_smooth[0]

'''



'''
fig, (ax_hist, ax_raster, ax) = plt.subplots(
    3, 1, sharex=True,
    gridspec_kw={'height_ratios': [1, 3, 3]},
    constrained_layout=True
)

plott.plot_spikes_hist(
    net.cell_response,
    ax=ax_hist,
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
    color={'Distal': 'green', 'Proximal': 'red'},
    show=False
)
ax_hist.set_xlabel('')


from hnn_core.viz import plot_spikes_raster
plot_spikes_raster(net.cell_response, ax=ax_raster, show=False, marker_size=5.0)
ax_raster.set_xlabel('')

ax.plot(times, dpl.data['agg'], label='', lw=1.5)
#ax.plot(times, dpl_smooth.data['agg'], label='smoothed', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
fig.suptitle("Dipole")
plt.show()
'''


#####################
# Total CSD from agg_i_mem
#####################

sources_agg, I_agg = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
    channels=["agg_i_mem"],
)

# agg_i_mem LFP
T_agg = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_agg,
    array_name="probe1",
)

lfp_agg_i_mem = tme.reconstruct_lfp_from_sources(
    T_agg,
    I_agg,
)

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

times_ = times[1:]
lfp_agg_i_mem_ = lfp_agg_i_mem[:, 1:]
#plott.plot_laminar_lfp_AC(times[1:], lfp_agg_i_mem_, contact_labels, scale=5.0)

csd_from_sources_ = csd_from_sources[:, 1:]
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
    overlay_raster_on_csd=True,)
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
    overlay_raster_on_csd=True)
fig.suptitle("LFP/CSD from synaptic currents")

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

# sanity check [SHOULD BE CHECKED MUCH BETTER!]
# errors build up because here we're considering all the compartments (not just the soma!) of both L2 and L5 pyramidal cells.
sources_intr, I_intr = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
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
#ax.plot(times_, dpl_tot_, label='Total (reconstructed)', lw=1.5)
#ax.plot(times_, dpl_intr_, label='Intrinsic (capacitive + ionic)', lw=1.5)
#ax.plot(times_, dpl_syn_, label='Synaptic', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Dipole (nAm)')
ax.legend()
plt.show()
fig.suptitle("Dipole")

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
    unit_csd="µA/mm³")
fig.suptitle("LFP/CSD from GABA_B currents")

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
    unit_csd="µA/mm³")
fig.suptitle("LFP/CSD from GABA_A currents")

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
    unit_csd="µA/mm³")
fig.suptitle("LFP/CSD from AMPA currents")

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
    unit_csd="µA/mm³")
fig.suptitle("LFP/CSD from NMDA currents")

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

net.cell_response.plot_spikes_raster(show=False)


###########################
# FROM INTRINSIC CURRENTS
###########################

sources_intr, I_intr = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
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

# LFP from synaptic currents only
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
    unit_csd="µA/mm³")
fig.suptitle("LFP/CSD from intrinsic currents")





# plot for meeting
titles = [
    ["Morphology", "Total (agg_i_mem)", "Synaptic", "Intrinsic"],
    ["GABA-B",     "GABA-A",            "AMPA",     "NMDA"],
]

fig, axes = plt.subplots(
    2, 4,
    constrained_layout=True,
    figsize=(20, 8)
)

# Morphology
plott.plot_cell_morphology_for_lfp_csd(
    net,
    contact_positions=contact_labels,
    cell_types=('L2_pyramidal', 'L5_pyramidal'),
    ax=axes[0, 0],
    show=False,
)

# CSD panels: (ax, data, vmin, vmax)
'''
csd_panels = [
    (axes[0, 1], csd_from_sources_, -60,  60),
    (axes[0, 2], csd_syn_,          -60,  60),
    (axes[0, 3], csd_intr_,         -60,  60),
    (axes[1, 0], csd_syn_gabab_,    -10,  10),
    (axes[1, 1], csd_syn_gabaa_,    -10,  10),
    (axes[1, 2], csd_syn_ampa_,     -10,  10),
    (axes[1, 3], csd_syn_nmda_,     -10,  10),
]'''
csd_panels = [
    (axes[0, 1], csd_from_sources_, -10,  10),
    (axes[0, 2], csd_syn_,          -10,  10),
    (axes[0, 3], csd_intr_,         -10,  10),
    #(axes[1, 0], csd_syn_gabab_,    -10,  10),
    #(axes[1, 1], csd_syn_gabaa_,    -10,  10),
    (axes[1, 2], csd_syn_ampa_,     -5,  5),
    (axes[1, 3], csd_syn_nmda_,     -5,  5),
]

for ax, data, vmin, vmax in csd_panels:
    plott.plot_laminar_csd_AC(
        times_, data, contact_labels,
        ax=ax, vmin=vmin, vmax=vmax,
        overlay_csd_traces=True,
        unit_csd="µA/mm³",
        show=False,
    )

# Titles
for row in range(2):
    for col in range(4):
        axes[row, col].set_title(titles[row][col])

for ax, _, _, _ in csd_panels:
    ax.figure.axes[-1].set_ylabel('')

for row in range(2):
    for col in range(4):
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel('')

for ax in fig.axes:
    ax.set_ylabel('')

plt.show()




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


def compute_lfp_csd_for_sources(net, sources, I, array_name="probe1"):
    B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sources, array_name=array_name)
    csd = tme.compute_csd_from_sources(B, I, V_bin)
    T = tme.build_transfer_resistance_matrix_for_sources(net, sources, array_name=array_name)
    lfp = tme.reconstruct_lfp_from_sources(T, I)
    return lfp[:, 1:], csd[:, 1:]

# for checking the contribution from specific intrinsic currents
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
    )
    fig.suptitle(f"LFP/CSD from {title}")








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

