import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_dipole_lfp_csd as tme
#import tm_currents_utils_forLFP as tme_lfp
from hnn_core.extracellular import calculate_csd2d
import Plotting_tools as plott


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


# Laminar probe
depths = np.arange(-625, 2150, 100)
electrode_pos = [(135, 135, z) for z in depths]
net.add_electrode_array('probe1', electrode_pos)

n_trials = 1

if "dpls" not in locals():
    with JoblibBackend(8):
        dpls = simulate_dipole(
            net,
            tstop=170.0,
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

contact_labels = np.asarray(depths, dtype=int)

plott.plot_lfp_morph_csd(
    times, 
    lfp_agg_i_mem, 
    csd_from_sources, 
    contact_labels, 
    net, 
    ext_inputs=net.cell_response, 
    spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']}, 
    scale_lfp=100.0,
    voltage_scalebar=50,
    vmin=-60,
    vmax=60,
    figsize=(18, 6)
    )



plott.plot_laminar_csd_AC(
    times,
    csd_from_sources,
    contact_labels=contact_labels,
    vmin=-100,
    vmax=100,
    interpolation=None,
    unit_csd="µA/mm³")
#plt.title("Total CSD")
    
# y-axis in units of μA/mm³, which is the unit of the CSD computed from the sources!!

# CSD from synaptic currents only
sources_syn, I_syn = tme.collect_synaptic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
)


B_syn, V_bin_syn, z_center_syn = tme.build_binning_matrix_for_sources(
    net,
    sources_syn,
    array_name="probe1"
)

csd_from_syn = tme.compute_csd_from_sources(
    B_syn,
    I_syn,
    V_bin_syn
)

plott.plot_laminar_csd_AC(
    times,
    csd_from_syn,
    contact_labels=contact_labels,
    vmin=-100,
    vmax=100,
    interpolation=None,
    unit_csd="µA/mm³")
#plt.title("CSD from I_syn")


# GABA B currents only
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

goes_negative = np.any(I_syn_gabab < 0, axis=1)  # shape (n_sources,)
print(np.any(goes_negative))   # True or False
print(goes_negative.sum())     # how many sources go negative

csd_from_syn_gabab = tme.compute_csd_from_sources(
    B_syn_gabab,
    I_syn_gabab,
    V_bin_syn_gabab
)

csd_goes_negative = np.any(csd_from_syn_gabab < 0, axis=1)
print(np.any(csd_goes_negative))   # True or False
print(csd_goes_negative.sum())   

plott.plot_laminar_csd_AC(
    times,
    csd_from_syn_gabab,
    contact_labels=contact_labels,
    vmin=-60,
    vmax=60,
    unit_csd="µA/mm³",
    interpolation=None
    )

#plt.title("CSD from GABA_B currents only")


# UP to here: GABA_B currents are all positive (sources), but the resulting CSD through plot_laminar_csd_AC displays sinks. I think this is due to interpolation (checking it).

# AMPA current only
sources_syn_ampa, I_syn_ampa = tme.filter_sources(sources_syn, I_syn, syn_names=["ampa"])

goes_positive = np.any(I_syn_ampa > 0, axis=1)  # shape (n_sources,)
print(np.any(goes_positive))   # True or False
print(goes_positive.sum())     # how many sources go positive

plt.figure()
for i, trace in enumerate(I_syn_ampa):
    plt.plot(times, trace, alpha=0.3, color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Current (nA)')
plt.title('ampa synaptic currents')
plt.show()

plt.figure()
plt.plot(times, I_syn_ampa[10,:], alpha=0.3, color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Current (nA)')
plt.title('ampa synaptic currents')
plt.show()

B_syn_ampa, V_bin_syn_ampa, z_center_syn_ampa = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_ampa,
    array_name="probe1"
)

csd_from_syn_ampa = tme.compute_csd_from_sources(
    B_syn_ampa,
    I_syn_ampa,
    V_bin_syn_ampa
)

goes_positive = np.any(csd_from_syn_ampa > 0, axis=1)  # shape (n_sources,)
print(np.any(goes_positive))   # True or False
print(goes_positive.sum())     # how many sources go positive


tme.plot_laminar_csd_AC(
    times,
    csd_from_syn_ampa,
    contact_labels=contact_labels,
    vmin=-30,
    vmax=30)


#NMDA only
sources_syn_NMDA, I_syn_NMDA = tme.filter_sources(sources_syn, I_syn, syn_names=["nmda"])


B_syn_NMDA, V_bin_syn_NMDA, z_center_syn_NMDA = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_NMDA,
    array_name="probe1"
)

csd_from_syn_NMDA = tme.compute_csd_from_sources(
    B_syn_NMDA,
    I_syn_NMDA,
    V_bin_syn_NMDA
)

tme.plot_laminar_csd_AC(
    times,
    csd_from_syn_NMDA,
    contact_labels=contact_labels,
    vmin=-30,
    vmax=30)



# GABA_A only
sources_syn_GABAA, I_syn_GABAA = tme.filter_sources(sources_syn, I_syn, syn_names=["gabaa"])

B_syn_GABAA, V_bin_syn_GABAA, z_center_syn_GABAA = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_GABAA,
    array_name="probe1"
)

csd_from_syn_GABAA = tme.compute_csd_from_sources(
    B_syn_GABAA,
    I_syn_GABAA,
    V_bin_syn_GABAA
)

tme.plot_laminar_csd_AC(
    times,
    csd_from_syn_GABAA,
    contact_labels=contact_labels,
    vmin=-30,
    vmax=30)


# GABA_B only
sources_syn_GABAB, I_syn_GABAB = tme.filter_sources(sources_syn, I_syn, syn_names=["gabab"])

B_syn_GABAB, V_bin_syn_GABAB, z_center_syn_GABAB = tme.build_binning_matrix_for_sources(
    net,
    sources_syn_GABAB,
    array_name="probe1"
)

csd_from_syn_GABAB = tme.compute_csd_from_sources(
    B_syn_GABAB,
    I_syn_GABAB,
    V_bin_syn_GABAB
)

tme.plot_laminar_csd_AC(
    times,
    csd_from_syn_GABAB,
    contact_labels=contact_labels,
    vmin=-30,
    vmax=30)
