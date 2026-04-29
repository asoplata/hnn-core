import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_extracellular_v2 as tme
from hnn_core.extracellular import calculate_csd2d
from hnn_core.viz import plot_laminar_csd 
from hnn_core.viz import plot_laminar_lfp
from types import SimpleNamespace

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model

filename = "simulation_with_net_v7_deeper"

net = jones_2009_model()
add_erp_drives_to_jones_model(net)

n_trials = 1

# Laminar probe
#depths = list(list(range(-125,2150,100))) #depths = np.arange(0, 2200, 100) 
depths = np.arange(-625, 2150, 100)
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
# dpl_plot = dpl.plot(
#    layer=["L5"],
#    show=False,
#)

lfp = net.rec_arrays['probe1'].voltages.mean(axis=0)
#print(lfp)
times = dpl.times

delta = np.median(np.diff(depths))   # electrode spacing in um
csd = calculate_csd2d(lfp, delta=delta)   # shape: (n_contacts, n_times)

contact_labels = np.asarray(depths, dtype=int)

fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
plot_laminar_lfp(times, lfp, contact_labels=contact_labels,ax=axes[0])
axes[0].set_title("LFP")
plot_laminar_csd(times, csd, contact_labels=contact_labels, vmin = -0.03, vmax = 0.03,ax=axes[1])
axes[1].set_title("CSD")
#plt.show()
fig.savefig(f"/Users/annacattani/Documents/HNN/hnn-core/runscripts/data/{filename}.png", dpi=300, bbox_inches="tight")
# plot
#fig = tme.plot_lfp_and_csd(
#    times,
#    lfp,
#    csd,
#    contact_positions=contact_labels,
#    titles=("LFP", "CSD"),
#)

def extract_currents(cell_response):
    currents = {
        "times": cell_response.times,
        "isec": cell_response.isec,
    }

    current_attrs = [
        "_agg_i_mem",
        "_agg_i_cap",
        "_agg_ina",
        "_agg_ik",
        "_ina_hh2",
        "_ik_hh2",
        "_ik_kca",
        "_ik_km",
        "_ica_ca",
        "_ica_cat",
        "_il_hh2",
        "_i_ar",
        "_vsec",
        "_ca",
    ]

    for attr in current_attrs:
        if hasattr(cell_response, attr):
            currents[attr] = getattr(cell_response, attr)

    return currents

def save_postproc_net(fname, net, dpls, depths, electrode_pos,
                      tstop, n_trials, scaling_factor, record_config=None):
    from types import SimpleNamespace
    import pickle
    import numpy as np

    saved_net = SimpleNamespace(
        cell_response=net.cell_response,
        rec_arrays=net.rec_arrays,
        _params=net._params,
        cell_types=net.cell_types,
        gid_ranges=net.gid_ranges,
        pos_dict=net.pos_dict,
    )

    currents = extract_currents(net.cell_response)

    results = {
        "tstop": tstop,
        "n_trials": n_trials,
        "scaling_factor": scaling_factor,
        "depths": np.asarray(depths),
        "electrode_pos": electrode_pos,
        "dpls": dpls,
        "net": saved_net,
        "record_config": record_config,
        "currents": currents,
    }

    with open(fname, "wb") as f:
        pickle.dump(results, f)


record_config = {
    "record_agg_i_mem": "all",
    "record_agg_i_cap": "all",
    "record_ina_hh2": "all",
    "record_ik_hh2": "all",
    "record_ik_kca": "all",
    "record_ik_km": "all",
    "record_ica_ca": "all",
    "record_ica_cat": "all",
    "record_il_hh2": "all",
    "record_i_ar": "all",
    "record_isec": "all",
}

save_postproc_net(
    f"/Users/annacattani/Documents/HNN/hnn-core/runscripts/data/{filename}.pkl",
    net=net,
    dpls=dpls,
    depths=depths,
    electrode_pos=electrode_pos,
    tstop=50,
    n_trials=n_trials,
    scaling_factor=scaling_factor,
    record_config=record_config,
)


'''
#import pickle

with open("simulation_with_net_v5.pkl", "rb") as f:
    results = pickle.load(f)

net_loaded = results["net"]



import postproc_tm_currents_extracellular_v2 as tme

lfp_na_l5, src_na_l5, T_na_l5, I_na_l5 = tme.reconstruct_intrinsic_lfp(
    net_loaded,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=["ina_hh2"],
    array_name="probe1",
)

csd_na_l5 = tme.reconstruct_csd(net, lfp_na_l5, array_name="probe1")
#times = np.asarray(net.rec_arrays["probe1"].times)
times = np.asarray(net_loaded.cell_response.times)
contact_positions = net.rec_arrays["probe1"].positions


#tme.plot_lfp_and_csd(
#    times,
#    lfp_na_l5,
#    csd_na_l5,
#    contact_positions=contact_positions,
#    titles=("LFP: L5 ina_hh2", "CSD: L5 ina_hh2"),
#)



#agg_i_mem includes all transmembrane currents, so should be identical to the LFP originally displayed
lfp_agg_i_mem, src_agg_i_mem, T_agg_i_mem, I_agg_i_mem = tme.reconstruct_intrinsic_lfp(
    net_loaded,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
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