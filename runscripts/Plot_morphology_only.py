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


net.pos_dict['L5_pyramidal'][0]
net.cell_types['L5_pyramidal']['cell_object'].sections['soma']._end_pts

#cell_types=('L2_pyramidal', 'L5_pyramidal'),
contact_labels = np.asarray(depths, dtype=int)

cell_types = ('L2_pyramidal', 'L5_pyramidal')
diam_scale = 5.0

gid = None
#diam_scale = 5.0

ax = plott.plot_cell_morphology_for_lfp_csd(
    net,
    contact_labels,
    cell_types=cell_types,
    gid=gid,
    diam_scale=diam_scale,
    show=False,
)

ticks = np.union1d(contact_labels, [0.0])
ax.set_yticks(ticks)
ax.set_yticklabels([f"{z:g}" for z in ticks])


#Alternative figure:
ax = plott.plot_cell_morphology_for_lfp_csd(
    net,
    contact_labels,
    cell_types=cell_types,
    gid=gid,
    diam_scale=diam_scale,
    show=False,
)

# left axis: add 0
ticks = np.union1d(contact_labels, [0.0])
ax.set_yticks(ticks)
ax.set_yticklabels([f"{z:g}" for z in ticks])

# right axis: 0 at L5 soma center
soma_pos = np.array(net.pos_dict['L5_pyramidal'][0])
soma_pts = np.array(net.cell_types['L5_pyramidal']['cell_object'].sections['soma']._end_pts)
soma_z_hnn = soma_pos[2] + soma_pts[:, 2].mean()

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())

ymin, ymax = ax.get_ylim()
rel_ticks = np.arange(
    np.ceil((ymin - soma_z_hnn) / 50) * 50,
    np.floor((ymax - soma_z_hnn) / 50) * 50 + 1,
    50,
)
ax2.set_yticks(rel_ticks + soma_z_hnn)
ax2.set_yticklabels([f"{int(t):g}" for t in rel_ticks])
ax2.set_ylabel('distance from L5 soma center (µm)')

plt.show()
