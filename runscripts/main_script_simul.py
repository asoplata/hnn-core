import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_extracellular_def as tme

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
depths = np.arange(-625, 2150, 100)
#depths = np.arange(-125, 2150, 100)
electrode_pos = [(135, 135, z) for z in depths]
net.add_electrode_array('probe1', electrode_pos)


with open(f"/Users/annacattani/Documents/HNN/hnn-core/runscripts/data/simulation_with_net_v7_deeper.pkl", "rb") as f:
#with open(f"/Users/annacattani/Documents/HNN/hnn-core/runscripts/data/simulation_with_net_v7.pkl", "rb") as f:
    results = pickle.load(f)

net_loaded = results["net"]
dpls = results["dpls"]

lfp_ampa_l5, src_ampa_l5, T_ampa_l5, I_ampa_l5 = tme.reconstruct_synaptic_lfp_by_name(
    net_loaded,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    syn_names=["ampa"],
    array_name="probe1",
    midpoint_mode="split_even",
)

csd_ampa_l5 = tme.reconstruct_csd(net_loaded, lfp_ampa_l5, array_name="probe1") #check why same number of traces as LFP

times = np.asarray(net_loaded.cell_response.times)
contact_positions = net_loaded.rec_arrays["probe1"].positions

'''
tme.plot_lfp_and_csd_AC(
    times,
    lfp_ampa_l5,
    csd_ampa_l5,
    net=net_loaded,
    contact_positions=contact_positions,
    titles=("LFP: L5 AMPA", "CSD: L5 AMPA"),
)
'''

tme.plot_lfp_and_csd_AC(
    times,
    lfp_ampa_l5,
    csd_ampa_l5,
    net=net_loaded,
    #dpl=dpls,
    contact_positions=contact_positions,
    titles=("LFP: L5 AMPA", "CSD: L5 AMPA"),
    trim_csd_edges=True,
    lfp_scale=1.2, 
    csd_traces_scale=1.2e4,
    show_inputs=True
)

# AMPA in L2/3 pyramidal cells
# TO be fixed: plot L2 pyramidal cell in the middle panel (think about plotting both L2 and L5 in the same panel, with different colors)

lfp_ampa_l2, src_ampa_l2, T_ampa_l2, I_ampa_l2 = tme.reconstruct_synaptic_lfp_by_name(
    net_loaded,
    trial_idx=0,
    cell_types=["L2_pyramidal"],
    syn_names=["ampa"],
    array_name="probe1",
    midpoint_mode="split_even",
)

csd_ampa_l2 = tme.reconstruct_csd(net_loaded, lfp_ampa_l2, array_name="probe1") #check why same number of traces as LFP

times = np.asarray(net_loaded.cell_response.times)
contact_positions = net_loaded.rec_arrays["probe1"].positions


tme.plot_lfp_and_csd(
    times,
    lfp_ampa_l2,
    csd_ampa_l2,
    net=net_loaded,
    contact_positions=contact_positions,
    titles=("LFP: L2 AMPA", "CSD: L2 AMPA"),
)


# sodium current in L5 pyramidal cells
lfp_na_l5, src_na_l5, T_na_l5, I_na_l5 = tme.reconstruct_intrinsic_lfp(
    net_loaded,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=["ina_hh2"],
    array_name="probe1",
)
csd_na_l5 = tme.reconstruct_csd(net_loaded, lfp_na_l5, array_name="probe1") #check why same number of traces as LFP

times = np.asarray(net_loaded.cell_response.times)
contact_positions = net_loaded.rec_arrays["probe1"].positions


tme.plot_lfp_and_csd(
    times,
    lfp_na_l5,
    csd_na_l5,
    net=net_loaded,
    contact_positions=contact_positions,
    titles=("LFP: L5 NA", "CSD: L5 NA"),
)




###


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

lfp_agg_i_mem_l5, src_agg_i_mem_l5, T_agg_i_mem_l5, I_agg_i_mem_l5 = tme.reconstruct_intrinsic_lfp(
    net_loaded,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=["agg_i_mem"],
    array_name="probe1",
)

csd_agg_i_mem_l5 = tme.reconstruct_csd(net, lfp_agg_i_mem_l5, array_name="probe1")
times = np.asarray(net_loaded.cell_response.times)
contact_positions = net.rec_arrays["probe1"].positions

tme.plot_lfp_and_csd(
    times,
    lfp_agg_i_mem_l5,
    csd_agg_i_mem_l5,
    contact_positions=contact_positions,
    csd_vmin = -0.03,
    csd_vmax = 0.03,
    titles=("LFP: agg_i_mem", "CSD: agg_i_mem"),
)




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