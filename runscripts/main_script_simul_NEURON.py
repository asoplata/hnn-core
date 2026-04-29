# to DO:
# 1) update probe to avoid source artifacts (done in main_script_simul_save.py) 
# 2) update filter_sources (line 733) in postproc_tm_currents_extracellular_nrncompatible.py.

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_extracellular_nrncompatible as tme

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

n_trials = 1

# Laminar probe
depths = np.arange(-125, 2150, 100)
electrode_pos = [(135, 135, z) for z in depths]
net.add_electrode_array('probe1', electrode_pos)

if "dpls" not in locals():
    with JoblibBackend(8):
        dpls = simulate_dipole(
            net,
            tstop=250.0,
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

scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

dpl = dpls[0]

# ina_hh2 in L5 pyramidal neurons
lfp_na_l5, src_na_l5, T_na_l5, I_na_l5 = tme.reconstruct_intrinsic_lfp(
    net,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=["ina_hh2"],
    array_name="probe1",
)

csd_na_l5 = tme.reconstruct_csd(net, lfp_na_l5, array_name="probe1")
times = np.asarray(net.rec_arrays["probe1"].times)
contact_positions = net.rec_arrays["probe1"].positions

contact_labels = np.asarray(depths, dtype=int)

fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
tme.plot_laminar_lfp(times, lfp_na_l5, contact_labels=contact_labels,ax=axes[0])
axes[0].set_title("LFP")
tme.plot_laminar_csd(times, csd_na_l5, contact_labels=contact_labels, ax=axes[1])
axes[1].set_title("CSD")



# agg_i_mem includes all transmembrane currents, so should be identical to the LFP originally displayed
lfp_agg_i_mem_l5, src_agg_i_mem_l5, T_agg_i_mem_l5, I_agg_i_mem_l5 = tme.reconstruct_intrinsic_lfp(
    net,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=["agg_i_mem"],
    array_name="probe1",
)

lfp_agg_i_mem_l2, src_agg_i_mem_l2, T_agg_i_mem_l2, I_agg_i_mem_l2 = tme.reconstruct_intrinsic_lfp(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal"],
    channels=["agg_i_mem"],
    array_name="probe1",
)

lfp_agg_i_mem = lfp_agg_i_mem_l5 + lfp_agg_i_mem_l2

csd_agg_i_mem = tme.reconstruct_csd(net, lfp_agg_i_mem, array_name="probe1")
times = np.asarray(net.rec_arrays["probe1"].times)
contact_positions = net.rec_arrays["probe1"].positions

fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
tme.plot_laminar_lfp(times, lfp_agg_i_mem, contact_labels=contact_labels, ax=axes[0])
axes[0].set_title("LFP: agg_i_mem")
tme.plot_laminar_csd(times, csd_agg_i_mem, contact_labels=contact_labels, vmin=-0.03, vmax=0.03, ax=axes[1])
axes[1].set_title("CSD: agg_i_mem")


#STOP HERE

times = np.asarray(net.rec_arrays["probe1"].times)
contact_positions = net.rec_arrays["probe1"].positions

contact_labels = np.asarray(depths, dtype=int)

fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
tme.plot_laminar_lfp(times, lfp_na_l5, contact_labels=contact_labels,ax=axes[0])
axes[0].set_title("LFP")
tme.plot_laminar_csd(times, csd_na_l5, contact_labels=contact_labels, ax=axes[1])
axes[1].set_title("CSD")

#tme.plot_lfp_and_csd(
#    times,
#    lfp_na_l5,
#    csd_na_l5,
#    contact_positions=contact_positions,
#    titles=("LFP: L5 ina_hh2", "CSD: L5 ina_hh2"),
#)


breakpoint


with open(f"/Users/annacattani/Documents/HNN/hnn-core/runscripts/data/simulation_with_net_v7.pkl", "rb") as f:
    results = pickle.load(f)

net_loaded = results["net"]

lfp_na_l5, src_na_l5, T_na_l5, I_na_l5 = tme.reconstruct_intrinsic_lfp(
    net_loaded,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=["ina_hh2"],
    array_name="probe1",
)

'''
csd_na_l5 = tme.reconstruct_csd(net, lfp_na_l5, array_name="probe1")
#times = np.asarray(net.rec_arrays["probe1"].times)
times = np.asarray(net_loaded.cell_response.times)
contact_positions = net.rec_arrays["probe1"].positions


tme.plot_lfp_and_csd(
    times,
    lfp_na_l5,
    csd_na_l5,
    contact_positions=contact_positions,
    titles=("LFP: L5 ina_hh2", "CSD: L5 ina_hh2"),
)
'''



#agg_i_mem includes all transmembrane currents, so should be identical to the LFP originally displayed
lfp_agg_i_mem_l5, src_agg_i_mem_l5, T_agg_i_mem_l5, I_agg_i_mem_l5 = tme.reconstruct_intrinsic_lfp(
    net_loaded,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=["agg_i_mem"],
    array_name="probe1",
)

lfp_agg_i_mem_l2, src_agg_i_mem_l2, T_agg_i_mem_l2, I_agg_i_mem_l2 = tme.reconstruct_intrinsic_lfp(
    net_loaded,
    trial_idx=0,
    cell_types=["L2_pyramidal"],
    channels=["agg_i_mem"],
    array_name="probe1",
)

lfp_agg_i_mem = lfp_agg_i_mem_l5 + lfp_agg_i_mem_l2

csd_agg_i_mem = tme.reconstruct_csd(net, lfp_agg_i_mem, array_name="probe1")
times = np.asarray(net_loaded.cell_response.times)
contact_positions = net.rec_arrays["probe1"].positions

tme.plot_lfp_and_csd(
    times,
    lfp_agg_i_mem,
    csd_agg_i_mem,
    contact_positions=contact_positions,
    csd_vmin = -0.03,
    csd_vmax = 0.03,
    titles=("LFP: agg_i_mem", "CSD: agg_i_mem"),
)
'''
lfp_agg_i_mem, src_agg_i_mem, T_agg_i_mem, I_agg_i_mem = tme.reconstruct_intrinsic_lfp(
    net_loaded,
    trial_idx=0,
    cell_types=["L2_pyramidal"],
    channels=["agg_i_mem"],
    array_name="probe1",
)

csd_agg_i_mem = tme.reconstruct_csd(net, lfp_agg_i_mem, array_name="probe1")
#times = np.asarray(net.rec_arrays["probe1"].times)
times = np.asarray(net_loaded.cell_response.times)
contact_positions = net.rec_arrays["probe1"].positions

tme.plot_lfp_and_csd(
    times,
    lfp_agg_i_mem,
    csd_agg_i_mem,
    contact_positions=contact_positions,
    titles=("LFP: agg_i_mem", "CSD: agg_i_mem"),
)
'''

'''
# Synaptic: midpoint-segment approximation
lfp_syn, src_syn, T_syn, I_syn = tme.reconstruct_synaptic_lfp(
    net_loaded,
    cell_types=["L2_pyramidal", "L5_pyramidal"],
    array_name="probe1",
    midpoint_mode="split_even",
)

csd_syn = tme.reconstruct_csd(net, lfp_syn, array_name="probe1")

tme.plot_lfp_and_csd(
    times,
    lfp_syn,
    csd_syn,
    contact_positions=contact_positions,
    titles=("LFP: synaptic approx", "CSD: synaptic approx"),
)
'''