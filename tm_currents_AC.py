import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.dipole import TransmembraneRecordingConfig

net = jones_2009_model()
add_erp_drives_to_jones_model(net)

n_trials = 1

# new
tm_record_cfg = TransmembraneRecordingConfig(
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
)
#

if "dpls" not in locals():
    with JoblibBackend(8):
        dpls = simulate_dipole(
            net,
            tstop=50,#170.0,
            n_trials=n_trials,
            record_imem=tm_record_cfg,
            record_isec="all",
        )

scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

dpl = dpls[0]
dpl_plot = dpl.plot(
    layer=["L5"],
    show=False,
)


from tm_currents_utils import (
    postproc_tm_currents,
    agg_transmembrane_segment_recordings_by_celltype,
    agg_transmembrane_section_recordings_by_celltype,
    get_channel_section_data,
    plot_segment_recordings_by_section,
    plot_single_channel_by_section_celltype,
    plot_channels_by_single_section,
    plot_flat_neuron,
    plot_3d_neuron,
    postproc_soma_dipole,
)

'''
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 15),
)

test_imem_L5 = postproc_tm_currents(
    net=net,
    from_components=False,
)

ax[1].plot(
    dpl.times,
    test_imem_L5,
)

ax[0].set_ylim(-200, 100)
ax[1].set_ylim(-200, 100)


_ = dpl.plot(
    layer=["L5"],
    ax=ax[0],
)


test_imem_L2 = postproc_tm_currents(
    net=net,
    cell_type="L2_pyramidal",
    from_components=False,
)

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 10),
)

ax[0].plot(
    dpl.times,
    test_imem_L2,
)

ax[0].set_ylim(-30, 50)
ax[1].set_ylim(-30, 50)

_ = dpl.plot(
    layer=["L2"],
    ax=ax[1],
)

'''


ina_hh2_segment_data = agg_transmembrane_segment_recordings_by_celltype(
    net,
    trial_number=0,
    target_channel="ina_hh2",
)

ina_hh2_section_data = agg_transmembrane_section_recordings_by_celltype(
    net,
    trial_number=0,
    target_channel="ina_hh2",
)

channel_section_data = get_channel_section_data(
    net,
    channels=[
        "ina_hh2",
        "ik_hh2",
        "ik_kca",
        "ik_km",
        "ica_ca",
        "ica_cat",
        "il_hh2",
        "i_ar",
    ],
)




cell_type = "L5_pyramidal"
default_cell_searchkey = net.cell_types[cell_type]["cell_object"].name
celltype_sections = list(
    net.cell_types["L5_pyramidal"]["cell_object"].sections.keys(),
)

# reduce _params down to current cell_type only
default_cell_keymatches = [
    key for key in list(net._params.keys()) if default_cell_searchkey in key
]

defaults = {
    key: value for key, value in net._params.items() if key in default_cell_keymatches
}


# values pulled directly from _cell_L5Pyr in cells_default
end_pts = {
    "soma": [[0, 0, 0], [0, 0, 23]],
    "apical_trunk": [[0, 0, 23], [0, 0, 83]],
    "apical_oblique": [[0, 0, 83], [-150, 0, 83]],
    "apical_1": [[0, 0, 83], [0, 0, 483]],
    "apical_2": [[0, 0, 483], [0, 0, 883]],
    "apical_tuft": [[0, 0, 883], [0, 0, 1133]],
    "basal_1": [[0, 0, 0], [0, 0, -50]],
    "basal_2": [[0, 0, -50], [-106, 0, -156]],
    "basal_3": [[0, 0, -50], [106, 0, -156]],
}

# endpoints grabbed from procedurally-generated cell
procedural_end_pts = {}
for section in celltype_sections:
    procedural_end_pts[section] = (
        net.cell_types[cell_type]["cell_object"].sections[section].end_pts
    )

x_offsets = {
    "apical_oblique": -5,
    "apical_1": 0,
    "apical_2": 0,
    "apical_tuft": 0,
    "basal_2": -5,
    "basal_3": 5,
}


distinct_colors = {
    "soma": "orange",
    "apical_trunk": "red",
    "apical_oblique": "purple",
    "apical_1": "magenta",
    "apical_2": "pink",
    "apical_tuft": "brown",
    "basal_1": "yellow",
    "basal_2": "olive",
    "basal_3": "darkgreen",
}

blues = {
    "soma": "#4895ef",
    "apical_trunk": "#4cc9f0",
    "apical_oblique": "#caf0f8",
    "apical_1": "#90e0ef",
    "apical_2": "#ade8f4",
    "apical_tuft": "#e0f7fa",
    "basal_1": "#4cc9f0",
    "basal_2": "#90e0ef",
    "basal_3": "#ade8f4",
}

plot_flat_neuron(
    end_pts,
    defaults,
    x_offsets=x_offsets,
    gap=30,
    colors=blues,
    width_scale=2,
)

plot_flat_neuron(
    procedural_end_pts,
    defaults,
    x_offsets=x_offsets,
    gap=0,
    colors=distinct_colors,
    width_scale=2,
)

#get_ipython().run_line_magic("matplotlib", "tk")

add_gap = False

if add_gap:
    gap = 20
    x_offsets = {
        "apical_oblique": -20,
        "apical_1": 0,
        "apical_2": 0,
        "apical_tuft": 0,
        "basal_2": -20,
        "basal_3": 20,
    }
else:
    gap = 0
    x_offsets = {
        "apical_oblique": 0,
        "apical_1": 0,
        "apical_2": 0,
        "apical_tuft": 0,
        "basal_2": 0,
        "basal_3": 0,
    }

ax_main = plot_3d_neuron(
    end_pts,
    defaults,
    x_offsets=x_offsets,
    gap=gap,
    colors=distinct_colors,
    shade=True,
    width_scale=2,
    show_section_labels=False,
)

plt.show()


l5_seg_fig = plot_segment_recordings_by_section(
    section_name="apical_trunk",
    single_channel_data=ina_hh2_segment_data,
    cell_type="L5_pyramidal",
    overwrite_channel_name="Na+ HH2",
)

l2_seg_fig = plot_segment_recordings_by_section(
    section_name="apical_trunk",
    single_channel_data=ina_hh2_segment_data,
    cell_type="L2_pyramidal",
    overwrite_channel_name="Na+ HH2",
)


# AC: For one channel, how does that current look across the different sections of one cell type?
plot_single_channel_by_section_celltype(
    single_channel_data=ina_hh2_section_data,
    end_pts=end_pts,
    defaults=defaults,
    cell_type="L5_pyramidal",
    overwrite_channel_name="Na+ HH2",
    show_neuron_previews=True,
)

# AC: former function "plot_single_channel_by_section_celltype"
# AC: This: For one section, how do the different channels look in that section for one cell type?
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 15),
)
plt.show()


test_imem_L5 = postproc_soma_dipole(
    net=net,
    from_components=False,
)

ax[0].plot(
    dpl.times[1:],
    test_imem_L5[1:],
)

test_imem_L5 = postproc_soma_dipole(
    net=net,
    from_components=True,
)

ax[1].plot(
    dpl.times[1:],
    test_imem_L5[1:],
)

ax[0].set_ylim(-200, 100)
ax[1].set_ylim(-200, 100)