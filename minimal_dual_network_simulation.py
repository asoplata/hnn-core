"""
Minimal Dual Network Simulation

Simulates two smallest possible HNN-Core networks (one cell per type) and analyzes results.
"""
import copy
from collections import OrderedDict
from hnn_core import jones_2009_model, simulate_dipole, Network
import matplotlib.pyplot as plt
from hnn_core.hnn_io import write_network_configuration

def create_minimal_network(cell_type_suffix=None, gid_start=0, network_separation=0):
    """Create a minimal network with one cell per type and shift positions."""
    net = jones_2009_model()
    # Optionally rename cell types for net2
    if cell_type_suffix:
        mapping = OrderedDict([
            ('L2_basket', f'L2_basket{cell_type_suffix}'),
            ('L2_pyramidal', f'L2_pyramidal{cell_type_suffix}'),
            ('L5_basket', f'L5_basket{cell_type_suffix}'),
            ('L5_pyramidal', f'L5_pyramidal{cell_type_suffix}'),
        ])
        net._rename_cell_types(mapping)
        # Update GID ranges to start at gid_start and avoid overlap
        current_gid = gid_start
        net.gid_ranges = OrderedDict()
        for ct in mapping.values():
            n_cells = len(net.pos_dict[ct])
            net.gid_ranges[ct] = range(current_gid, current_gid + n_cells)
            current_gid += n_cells
        # # Shift all cell positions in x-direction by network_separation
        # for cell_type, positions in net.pos_dict.items():
        #     shifted_positions = []
        #     for pos in positions:
        #         # Only shift if not origin
        #         if isinstance(pos, (tuple, list)) and len(pos) == 3:
        #             shifted_pos = (pos[0] + network_separation, pos[1], pos[2])
        #         else:
        #             shifted_pos = pos  # leave unchanged
        #         shifted_positions.append(shifted_pos)
        #     net.pos_dict[cell_type] = shifted_positions

    return net
def combine_networks(net1, net2):
    """Combine two HNN-Core networks into a single network."""
    combined = Network()
    # Merge cell_types, pos_dict, gid_ranges, and external_drives
    combined.cell_types = {**net1.cell_types, **net2.cell_types}
    combined.pos_dict = {**net1.pos_dict, **net2.pos_dict}
    combined.gid_ranges = OrderedDict({**net1.gid_ranges, **net2.gid_ranges})
    combined.external_drives = {**net1.external_drives, **net2.external_drives}
    # Update _n_gids
    combined._n_gids = max(
        [max(rng) for rng in combined.gid_ranges.values() if len(rng) > 0]
    ) + 1
    return combined
def connect_pyramidal_to_all(net1, net2, weight=0.001, delay=1.0, receptor='gabaa'):
    """Connect all pyramidal cells in net1 to all cells in net2."""
    # Get pyramidal cell types in net1
    pyr_types_net1 = [ct for ct in net1.cell_types if 'pyramidal' in ct]
    # Get all cell types in net2
    all_types_net2 = list(net2.cell_types.keys())

    # For each pyramidal cell in net1
    for src_type in pyr_types_net1:
        for src_gid in net1.gid_ranges[src_type]:
            src_pos = net1.pos_dict[src_type][src_gid - net1.gid_ranges[src_type][0]]
            # For each cell in net2
            for target_type in all_types_net2:
                for target_gid in net2.gid_ranges[target_type]:
                    target_pos = net2.pos_dict[target_type][target_gid - net2.gid_ranges[target_type][0]]
                    # Add connection: you may need to adjust location and receptor for your model
                    net2.add_connection(
                        src_gids=[src_gid],
                        target_gids=[target_gid],
                        loc='soma',  # or appropriate section
                        receptor=receptor,
                        weight=weight,
                        delay=delay,
                        lamtha=3.0,
                        allow_autapses=False
                    )


def plot_combined_spike_raster(net1, net2, title="Combined Spike Raster",plot_net1=True, plot_net2=True):
    """Plot spikes from both networks on a single raster plot."""
    # Gather spike data from both networks
    spikes = []
    gids = []
    types = []
    # # Net1
    if plot_net1:
        for trial_idx in range(len(net1.cell_response.spike_times)):
            spikes.extend(net1.cell_response.spike_times[trial_idx])
            gids.extend(net1.cell_response.spike_gids[trial_idx])
            types.extend(net1.cell_response.spike_types[trial_idx])
    if plot_net2:
    # Net2
        for trial_idx in range(len(net2.cell_response.spike_times)):
            spikes.extend(net2.cell_response.spike_times[trial_idx])
            gids.extend(net2.cell_response.spike_gids[trial_idx])
            types.extend(net2.cell_response.spike_types[trial_idx])

    # Assign each cell type a color
    unique_types = sorted(set(types))
    color_map = {ct: plt.cm.tab10(i % 10) for i, ct in enumerate(unique_types)}
    colors = [color_map[ct] for ct in types]

    # Assign each GID a row for plotting
    unique_gids = sorted(set(gids))
    gid_to_row = {gid: i for i, gid in enumerate(unique_gids)}
    rows = [gid_to_row[gid] for gid in gids]

    plt.figure(figsize=(10, 6))
    plt.scatter(spikes, rows, c=colors, s=10)
    plt.xlabel("Time (ms)")
    plt.ylabel("Cell (GID)")
    plt.title(title)
    # Add legend for cell types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=ct,
                   markerfacecolor=color_map[ct], markersize=8)
        for ct in unique_types
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



def get_next_gid(net):
    max_gid = -1
    for rng in net.gid_ranges.values():
        if len(rng) > 0:
            max_gid = max(max_gid, max(rng))
    return max_gid + 1

def main():
    print("Minimal dual network simulation...")

    net1 = create_minimal_network(cell_type_suffix=None, gid_start=0)
    next_gid = get_next_gid(net1)
    net1.add_evoked_drive(
        'evdist1', mu=5.0, sigma=1.0, numspikes=1, location='distal',
        weights_ampa={'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1},gid_start=next_gid
    )
    next_gid = get_next_gid(net1)
    net2 = create_minimal_network(cell_type_suffix="_net2",gid_start=next_gid)    
   
    # Only call get_next_gid(net2) here, before adding any drives!
    next_gid_net2 = get_next_gid(net2)
    
    net2.add_evoked_drive(
        'evdist2', mu=5.0, sigma=1.0, numspikes=1, location='distal',
        weights_ampa={'L2_pyramidal_net2': 0.1, 'L5_pyramidal_net2': 0.1},gid_start=next_gid_net2
    )
    # After creating net1 and net2, or after combining:
    write_network_configuration(net1, "net1_config.json")
    write_network_configuration(net2, "net2_config.json")
    # Combine networks after adding drives
    # combined_net = combine_networks(net1, net2)

    net_1=net1.copy()
    net_2=net2.copy()
    print("Simulating net1...")
    print("Net1 gid ranges:", net_1.gid_ranges)
    print("Net2 gid ranges:", net_2.gid_ranges)
    dpl_1 = simulate_dipole(net_1, tstop=20, dt=0.025, n_trials=1)
    print("Simulating net2...")
    dpl_2 = simulate_dipole(net_2, tstop=20, dt=0.025, n_trials=1)
    print("before connection")

    print("Net1 dipole peak:", max(abs(dpl_1[0].data['agg'])))
    print("Net2 dipole peak:", max(abs(dpl_2[0].data['agg'])))
    print("Net1 spikes:", len(net_1.cell_response.spike_times[0]))
    print("Net2 spikes:", len(net_2.cell_response.spike_times[0]))
    for ct in net_1.cell_types:
        gids = list(net_1.gid_ranges[ct])
        spikes = [gid for gid in gids if gid in net_1.cell_response.spike_gids[0]]
        print(f"{ct}: {len(spikes)} spiking cells out of {len(gids)}")

    for ct in net_2.cell_types:
        gids = list(net_2.gid_ranges[ct])
        spikes = [gid for gid in gids if gid in net_2.cell_response.spike_gids[0]]
        print(f"{ct}: {len(spikes)} spiking cells out of {len(gids)}")
    plot_combined_spike_raster(net_1, net_2, title="Net 1 + Net 2 (when only net2 with suffix)",plot_net1=True,plot_net2=True)

    # # Connect net1 pyramidal cells to all net2 cells
    # connect_pyramidal_to_all(net1, net2, weight=0.001, delay=1.0, receptor='gabaa')
    # print("Successfully connected net1 pyramidal cells to all net2 cells.")
    # print("Simulating net1...")
    # dpl1 = simulate_dipole(net1, tstop=20, dt=0.025, n_trials=1)
    # print("Simulating net2...")
    # dpl2 = simulate_dipole(net2, tstop=20, dt=0.025, n_trials=1)
    # print("after connection")
    # print("Net1 dipole peak:", max(abs(dpl1[0].data['agg'])))
    # print("Net2 dipole peak:", max(abs(dpl2[0].data['agg'])))
    # print("Net1 spikes:", len(net1.cell_response.spike_times[0]))
    # print("Net2 spikes:", len(net2.cell_response.spike_times[0]))

    # plot_combined_spike_raster(net1, net2, title="Combined Raster (Net1 + Net2) after connection",plot_net1=False, plot_net2=True)    



if __name__ == "__main__":
    main()