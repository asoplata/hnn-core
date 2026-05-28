import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_dipole_lfp_csd as tme
#import tm_currents_utils_forLFP as tme_lfp
from hnn_core.extracellular import calculate_csd2d


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

# ------
# Plot LFP and CSD as done in HNN to have a visual reference
# ------
'''
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8),
                        gridspec_kw={'height_ratios': [3,3]})

lfp_hnn = net.rec_arrays["probe1"].voltages[0]  # HNN's LFP; trial 0

window_len = 10
decimate = [5, 4]  # from 40k to 8k to 2k


net.rec_arrays['probe1'][0].plot_lfp(
    ax=axs[0], show=False)
#net.rec_arrays['probe1'][0].smooth(window_len=window_len).plot_lfp(
#    ax=axs[0], decim=decimate, show=False)


net.rec_arrays['probe1'][0].plot_csd(ax=axs[1], show=False)
#net.rec_arrays['probe1'][0].smooth(window_len=window_len).plot_csd(ax=axs[1], show=False)
plt.tight_layout()
plt.show()
'''

sources_agg, I_agg = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
    channels=["agg_i_mem"],
)

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

tme.plot_laminar_csd_AC(
    times,
    csd_from_sources,
    contact_labels=contact_labels,
    vmin=-100,
    vmax=100)
# y-axis in units of μA/mm³, which is the unit of the CSD computed from the sources!!

#sigma = 0.3  # S/m (HNN default)
sigma = 1 # CHECK THIS, I think in HNN's CSD sigma is actually dropped.

# To compare csd_from_sources (μA/mm³) with HNN's output (μV/μm²):
csd_in_hnn_units = csd_from_sources / (sigma * 1e3)   # → μV/μm²

tme.plot_laminar_csd_AC(
    times,
    csd_in_hnn_units,
    contact_labels=contact_labels,
    vmin=-0.1,
    vmax=0.1)


# Comparison to CSD computed from LFP
lfp_hnn = net.rec_arrays["probe1"].voltages[0]  # HNN's LFP; trial 0
delta = np.median(np.diff(depths))   # electrode spacing in um
csd = calculate_csd2d(lfp_hnn, delta=delta)   # shape: (n_contacts, n_times)

tme.plot_laminar_csd_AC(
    times,
    csd,
    contact_labels=contact_labels,
    vmin=-0.1,
    vmax=0.1)


