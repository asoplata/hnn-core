import numpy as np
from itertools import cycle
import colorsys
import warnings
#from .externals.mne import _validate_type

def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.

    NB copied from :func:`mne.viz.utils.plt_show`.

    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    from matplotlib import get_backend
    import matplotlib.pyplot as plt

    if show and get_backend() != "agg":
        (fig or plt).show(**kwargs)


def _decimate_plot_data(decim, data, times, sfreq=None):
    from scipy.signal import decimate

    if not isinstance(decim, list):
        decim = [decim]

    for dec in decim:
        if not isinstance(dec, int) or dec < 1:
            raise ValueError(
                "each decimation factor must be a positive int, "
                f"but {dec} is a {type(dec)}"
            )
        data = decimate(data, dec)
        times = times[::dec]

    if sfreq is None:
        return data, times
    else:
        sfreq /= np.prod(decim)
        return data, times, sfreq



def plot_laminar_lfp_AC(
    times,
    data,
    depths,
    ax=None,
    decim=None,
    color="cividis",
    show=True,
    baseline_samps=None,
    scale=1.0,
    voltage_scalebar=200,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    times = np.asarray(times)
    data = np.asarray(data)
    depths = np.asarray(depths, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if len(times) != data.shape[1]:
        raise ValueError(f"times length {len(times)} != data columns {data.shape[1]}")
    if len(depths) != data.shape[0]:
        raise ValueError(f"depths length {len(depths)} != data rows {data.shape[0]}")

    n_contacts = data.shape[0]

    # Scale traces so the largest peak-to-peak amplitude fills one inter-contact gap.
    # ``scale`` allows shrinking (< 1) or expanding (> 1) relative to that.
    depth_spacing = np.diff(depths)[0]
    max_amp = np.max(np.ptp(data, axis=1))
    effective_scale = scale * depth_spacing / max_amp

    if isinstance(color, str):
        color = plt.get_cmap(color, n_contacts)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    for i, trace in enumerate(data):
        plot_data, plot_times = trace, times
        if decim is not None:
            plot_data, plot_times = _decimate_plot_data(decim, plot_data, plot_times)
        if baseline_samps is not None:
            plot_data = plot_data - np.mean(plot_data[:baseline_samps])
        col = color(i) if callable(color) else color
        ax.plot(plot_times, depths[i] + effective_scale * plot_data, color=col)

    ax.set_xlim(times[0], times[-1])

    # Left axis: depth from surface (0 at top, increasing downward as in the experimental literature).
    # Right axis: HNN z-coordinate.
    z_max = depths.max()
    ax.set_yticks(depths)
    ax.set_yticklabels([f"{d:g}" for d in (z_max - depths)])
    ax.set_ylim(depths.min() - depth_spacing, depths.max() + depth_spacing)
    ax.set_ylabel("Depth from surface (µm)")
    ax.set_xlabel("Time (ms)")

    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks(depths)
    ax_r.set_yticklabels([f"{d:g}" for d in depths])
    ax_r.set_ylabel("HNN z (µm)")

    if voltage_scalebar is not None:
        scalebar = AnchoredSizeBar(
            ax.transData,
            1,
            f"{voltage_scalebar:.0f} " + r"$\mu V$",
            "upper left",
            size_vertical=voltage_scalebar * effective_scale,
            pad=0.1,
            color="black",
            label_top=False,
            frameon=False,
        )
        ax.add_artist(scalebar)

    plt_show(show)
    return ax.get_figure()


def _draw_cell_morphology(ax, net, cell_type='L5_pyramidal', gid=None,
                          color_by_region=True, center_x=True,
                          diam_scale=5.0, x_shift=0.0):
    """Draw one cell's sections as rectangles in the (x, z) plane.

    Parameters
    ----------
    x_shift : float
        Extra horizontal offset (in µm) added to every section. Useful when
        drawing several cells side by side on the same axis.
    """
    from matplotlib.patches import Polygon

    template = net.cell_types[cell_type]['cell_object']
    if gid is None:
        gid = list(net.gid_ranges[cell_type])[0]
    start_gid = net.gid_ranges[cell_type][0]
    soma_pos = np.asarray(net.pos_dict[cell_type][gid - start_gid], dtype=float)

    region_colors = {
        'soma':           'black',
        'apical_trunk':   '#1f4e79',
        'apical_oblique': '#5b9bd5',
        'apical_1':       '#1f4e79',
        'apical_2':       '#1f4e79',
        'apical_tuft':    '#0b2540',
        'basal_1':        '#a85432',
        'basal_2':        '#d08770',
        'basal_3':        '#d08770',
    }
    x_off = soma_pos[0] if center_x else 0.0

    for name, sec in template.sections.items():
        pts = np.asarray(sec._end_pts, dtype=float) + soma_pos
        x0, z0 = pts[0, 0] - x_off + x_shift, pts[0, 2]
        x1, z1 = pts[1, 0] - x_off + x_shift, pts[1, 2]
        diam = float(sec.diam) * diam_scale  # exaggerate for visibility

        dx, dz = x1 - x0, z1 - z0
        L = np.hypot(dx, dz)
        if L < 1e-6:
            # zero-length projection (rare): draw a small axis-aligned square
            h = diam / 2.0
            corners = [(x0 - h, z0 - h), (x0 + h, z0 - h),
                       (x0 + h, z0 + h), (x0 - h, z0 + h)]
        else:
            # unit vector along section, then perpendicular (90° in-plane)
            ux, uz = dx / L, dz / L
            px, pz = -uz, ux
            hx, hz = px * diam / 2.0, pz * diam / 2.0
            # Square (butt) ends — no rounded caps
            corners = [(x0 + hx, z0 + hz),
                       (x1 + hx, z1 + hz),
                       (x1 - hx, z1 - hz),
                       (x0 - hx, z0 - hz)]

        face = region_colors.get(name, 'gray') if color_by_region else 'lightgray'
        ax.add_patch(Polygon(
            corners, closed=True,
            facecolor=face, edgecolor='black',
            lw=0.4, joinstyle='miter',          # sharp corners
        ))
    return gid, soma_pos


def plot_cell_morphology_for_lfp_csd(
    net,
    contact_positions,
    cell_types=('L2_pyramidal', 'L5_pyramidal'),
    gid=None,
    diam_scale=5.0,
    ax=None,
    z_min_pad=-300,
    z_max_pad=2050,
    x_spacing=50,
    show=True,
):

    import matplotlib.pyplot as plt

    contact_positions = np.asarray(contact_positions, dtype=float)
    z_min = float(min(contact_positions.min(), z_min_pad))
    z_max = float(max(contact_positions.max(), z_max_pad))

    if ax is None:
        _, ax = plt.subplots(1, 1)

    x_shift = 0.0
    gid_labels = []
    for ct in cell_types:
        gid_used, _ = _draw_cell_morphology(
            ax, net, cell_type=ct, gid=gid, diam_scale=diam_scale,
            x_shift=x_shift,
        )
        gid_labels.append(f'{ct}\n(gid={gid_used})')

        # shift next cell by the x-extent of this one + spacing
        template = net.cell_types[ct]['cell_object']
        xs = [pt[0] for sec in template.sections.values() for pt in sec._end_pts]
        x_shift += (max(xs) - min(xs)) + x_spacing

    ax.set_title(' | '.join(gid_labels), fontsize=10)
    ax.set_xlabel('x (µm)')
    ax.set_ylim(z_min, z_max)
    ax.relim()
    ax.autoscale(axis='x', tight=False)
    ax.set_yticks(contact_positions)
    ax.set_yticklabels([f"{z:g}" for z in contact_positions])
    ax.set_ylabel('HNN z (µm)')

    for z in contact_positions:
        ax.axhline(z, color='lightgray', lw=0.3, zorder=0)

    plt_show(show)
    return ax


def plot_laminar_csd_AC(
    times,
    data,
    contact_labels,
    ax=None,
    colorbar=True,
    vmin=None,
    vmax=None,
    sink="b",
    interpolation="spline",
    show=True,
    overlay_csd_traces=False,
    scale_csd_traces=1.0,
    data_lfp=None,
    overlay_lfp_traces=False,
    scale_lfp_traces=1.0,
    unit_csd="µV/µm²", # or "µA/mm³"
    interp_kx=1, # linear interpolation by default
    interp_ky=1,
    overlay_raster_on_csd=False,
    cell_response=None,
    raster_colors=None,
):
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline
    from scipy.interpolate import RegularGridInterpolator
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    if sink[0].lower() == "b":
        cmap = "jet"
    elif sink[0].lower() == "r":
        cmap = "jet_r"
    else:
        raise RuntimeError(
            'Please use sink = "b" or sink = "r".'
            ' Only colormap "jet" is supported for CSD.'
        )
    data_ = data.copy()
    if interpolation == "spline":
        # create interpolation function
        interp_data = RectBivariateSpline(times, contact_labels, data.T,kx=interp_kx, ky=interp_ky)
        #interp_data = RegularGridInterpolator((times, contact_labels), data.T, method='linear')
        # increase number of contacts
        new_depths = np.linspace(
            contact_labels[0],
            contact_labels[-1],
            int(contact_labels[-1] - contact_labels[0]),
        )
        # interpolate
        data = interp_data(times, new_depths).T
        #data = interp_data((times[:, None], new_depths[None, :])) 
        #data = data.T
    elif interpolation is None:
        data = data
        new_depths = contact_labels

    # if vmin and vmax are both None, set colormap such that green = zero
    if vmin is None and vmax is None:
        vmin = -np.max(np.abs(data))
        vmax = np.max(np.abs(data))

    im = ax.pcolormesh(
        times, new_depths, data, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("time (ms)")
    
    # check this
    contact_labels = np.asarray(contact_labels)
    depth_spacing = np.diff(contact_labels)[0]
    z_max = contact_labels.max()
    ax.set_yticks(contact_labels)
    ax.set_yticklabels([f"{d:g}" for d in (z_max - contact_labels)])
    ax.set_ylim(contact_labels.min() - depth_spacing, contact_labels.max() + depth_spacing)
    ax.set_ylabel("Depth from surface (µm)")
    #

    valid_units = ("µV/µm²", "µA/mm³")
    if unit_csd not in valid_units:
        raise ValueError(f"unit_csd must be one of {valid_units}")
    
    if colorbar:
        color_axis = ax.inset_axes([1.05, 0, 0.02, 1], transform=ax.transAxes)
        plt.colorbar(im, ax=ax, cax=color_axis).set_label(f"CSD ({unit_csd})")

    if overlay_csd_traces:
        for i, (z, tr) in enumerate(zip(contact_labels, data_)):
            ax.plot(times, z + scale_csd_traces * tr, color='k', alpha=0.5)
        scalebar = AnchoredSizeBar(
            ax.transData, 1, f"{50:.0f} {unit_csd}",
            "upper left",
            size_vertical= 50 * scale_csd_traces,
            pad=0.1, color="black", frameon=False,
        )
        ax.add_artist(scalebar)
    elif overlay_lfp_traces:
        depth_spacing = np.diff(contact_labels)[0]
        max_amp = np.max(np.ptp(data_lfp, axis=1))
        effective_scale = scale_lfp_traces * depth_spacing / max_amp

        for i, (z, tr) in enumerate(zip(contact_labels, data_lfp)):
            ax.plot(times, z + effective_scale * tr, color='k', alpha=0.5)
        scalebar = AnchoredSizeBar(
            ax.transData, 1, f"{200:.0f} " + r"$\mu V$",
            "upper left",
            #size_vertical=200 * scale_lfp_traces,
            size_vertical=200 * effective_scale,
            pad=0.1, color="black", frameon=False,
        )
        ax.add_artist(scalebar)

    if overlay_raster_on_csd and cell_response is not None:
            if raster_colors is None:
                raster_colors = {'L2_pyramidal': 'C1', 'L5_pyramidal': 'C3'}

            if len(cell_response._spike_times[0]) > 0:
                spike_times_all = np.concatenate(
                    np.array(cell_response._spike_times, dtype=object)
                )
                spike_types_all = np.concatenate(
                    np.array(cell_response._spike_types, dtype=object)
                )
                spike_gids_all = np.concatenate(
                    np.array(cell_response._spike_gids, dtype=object)
                )
            else:
                spike_times_all = np.array([])
                spike_types_all = np.array([])
                spike_gids_all  = np.array([])

            ax_raster = ax.twinx()
            for i, stype in enumerate(['L2_pyramidal', 'L5_pyramidal']):
                mask = spike_types_all == stype
                ax_raster.scatter(
                    spike_times_all[mask],
                    spike_gids_all[mask],
                    s=2,
                    color=raster_colors.get(stype, f'C{i}'),
                    label=stype,
                    zorder=5,
                )

            ax_raster.invert_yaxis()
            ax_raster.set_yticks([])
            #ax_raster.set_ylabel('Cell ID')
            #ax_raster.legend(loc='upper right', fontsize=7, markerscale=4)
    #plt.tight_layout()
    plt_show(show)

    return ax.get_figure()

def plot_spikes_hist(
    cell_response,
    trial_idx=None,
    ax=None,
    spike_types=None,
    color=None,
    invert_spike_types=None,
    show=True,
    **kwargs_hist,
):
    """Plot the histogram of spiking activity across trials.

    Parameters
    ----------
    cell_response : instance of CellResponse
        The CellResponse object from net.cell_response
    trial_idx : int | list of int | None
        Index of trials to be plotted. If None, all trials plotted.
    ax : instance of matplotlib axis | None
        An axis object from matplotlib. If None,
        a new figure is created.
    spike_types: string | list | dictionary | None
        String input of a valid spike type is plotted individually.

        | Ex: ``'poisson'``, ``'evdist'``, ``'evprox'``, ...

        List of valid string inputs will plot each spike type individually.

        | Ex: ``['poisson', 'evdist']``

        Dictionary of valid lists will plot list elements as a group.

        | Ex: ``{'Evoked': ['evdist', 'evprox'], 'Tonic': ['poisson']}``

        If None, all input spike types are plotted individually if any
        are present. Otherwise spikes from all cells are plotted.
        Valid strings also include leading characters of spike types

        | Ex: ``'ev'`` is equivalent to ``['evdist', 'evprox']``
    invert_spike_types: string | list | None
        String input of a valid spike type to be mirrored about the y axis

        | Ex: ``'evdist'``, ``'evprox'``, ...

        List of valid spike types to be mirrored about the y axis

        | Ex: ``['evdist', 'evprox']``

        If None, all input spike types are plotted on the same y axis
    color : str | list of str | dict | None
        Input defining colors of plotted histograms. If str, all
        histograms plotted with same color. If list of str provided,
        histograms for each spike type will be plotted by cycling
        through colors in the list.

        If dict, colors must be specified for all spike_types as a key.
        If a group of spike types is defined by the `spike_types`
        parameter (see dictionary example for `spike_types`),
        the name of this group must be used to specify the colors.

        | Ex: ``{'evdist': 'g', 'evprox': 'r'}``, ``{'Tonic': 'b'}``

        If None, default color cycle used.
    show : bool
        If True, show the figure.
    **kwargs_hist : dict
        Additional keyword arguments to pass to ax.hist.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt

    n_trials = len(cell_response.spike_times)
    if trial_idx is None:
        trial_idx = list(range(n_trials))

    if isinstance(trial_idx, int):
        trial_idx = [trial_idx]
    #_validate_type(trial_idx, list, "trial_idx", "int, list of int")

    # Extract desired trials
    if len(cell_response._spike_times[0]) > 0:
        spike_times = np.concatenate(
            np.array(cell_response._spike_times, dtype=object)[trial_idx]
        )
        spike_types_data = np.concatenate(
            np.array(cell_response._spike_types, dtype=object)[trial_idx]
        )
    else:
        spike_times = np.array([])
        spike_types_data = np.array([])

    unique_types = np.unique(spike_types_data)
    spike_types_mask = {
        s_type: np.isin(spike_types_data, s_type) for s_type in unique_types
    }
    cell_types = ["L5_pyramidal", "L5_basket", "L2_pyramidal", "L2_basket"]
    input_types = np.setdiff1d(unique_types, cell_types)

    if isinstance(spike_types, str):
        spike_types = {spike_types: [spike_types]}

    if spike_types is None:
        if any(input_types):
            spike_types = input_types.tolist()
        else:
            spike_types = unique_types.tolist()
    if isinstance(spike_types, list):
        spike_types = {s_type: [s_type] for s_type in spike_types}
    if isinstance(spike_types, dict):
        for spike_label in spike_types:
            if not isinstance(spike_types[spike_label], list):
                raise TypeError(
                    f"spike_types[{spike_label}] must be a list. "
                    f"Got "
                    f"{type(spike_types[spike_label]).__name__}."
                )

    if not isinstance(spike_types, dict):
        raise TypeError("spike_types should be str, list, dict, or None")

    spike_labels = dict()
    for spike_label, spike_type_list in spike_types.items():
        for spike_type in spike_type_list:
            n_found = 0
            for unique_type in unique_types:
                if unique_type.startswith(spike_type):
                    if unique_type in spike_labels:
                        raise ValueError(
                            f"Elements of spike_types must map to"
                            f" mutually exclusive input types."
                            f" {unique_type} is found more than"
                            f" once."
                        )
                    spike_labels[unique_type] = spike_label
                    n_found += 1
            if n_found == 0:
                raise ValueError(f"No input types found for {spike_type}")

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    #_validate_type(color, (str, list, dict, None), "color", "str, list of str, or dict")

    if color is None:
        color_cycle = cycle(["r", "g", "b", "y", "m", "c"])
    elif isinstance(color, str):
        color_cycle = cycle([color])
    elif isinstance(color, list):
        color_cycle = cycle(color)

    if len(cell_response.times) > 0:
        bins = np.linspace(0, cell_response.times[-1], 50)
    else:
        bins = np.linspace(0, spike_times[-1], 50)

    # Create dictionary to aggregate spike times that have the same spike_label
    spike_type_times = {
        spike_label: list() for spike_label in np.unique(list(spike_labels.values()))
    }
    spike_color = dict()  # Store colors specified for each spike_label
    for spike_type, spike_label in spike_labels.items():
        if spike_label not in spike_color:
            if isinstance(color, dict):
                if spike_label not in color:
                    raise ValueError(
                        f"'{spike_label}' must be defined in color dictionary"
                    )
                #_validate_type(
                #    color[spike_label], str, "Dictionary values of color", "str"
                #)
                spike_color[spike_label] = color[spike_label]
            else:
                spike_color[spike_label] = next(color_cycle)
        spike_type_times[spike_label].extend(spike_times[spike_types_mask[spike_type]])

    if invert_spike_types is None:
        invert_spike_types = list()
    else:
        if not isinstance(invert_spike_types, (str, list)):
            raise TypeError(
                "'invert_spike_types' must be a string or a list of strings"
            )
        if isinstance(invert_spike_types, str):
            invert_spike_types = [invert_spike_types]

        # Check that spike types to invert are correctly specified
        unique_inputs = set(spike_labels.values())
        unique_invert_inputs = set(invert_spike_types)
        check_intersection = unique_invert_inputs.intersection(unique_inputs)
        if not check_intersection == unique_invert_inputs:
            raise ValueError(
                "Elements of 'invert_spike_types' mustmap to valid input types"
            )

    # Initialize secondary axis
    ax1 = None

    # Plot aggregated spike_times
    for spike_label, plot_data in spike_type_times.items():
        hist_color = spike_color[spike_label]

        # Plot on the primary y-axis
        if spike_label not in invert_spike_types:
            ax.hist(plot_data, bins, label=spike_label, color=hist_color, **kwargs_hist)
        # Plot on secondary y-axis
        else:
            if ax1 is None:
                ax1 = ax.twinx()
            ax1.hist(
                plot_data, bins, label=spike_label, color=hist_color, **kwargs_hist
            )
            # Need to add label for easy removal later

    # Set the y-limits based on the maximum across both axes
    if ax1 is not None:
        ax_ylim = ax.get_ylim()[1]
        ax1_ylim = ax1.get_ylim()[1]

        y_max = max(ax_ylim, ax1_ylim)
        ax.set_ylim(0, y_max)
        ax1.set_ylim(0, y_max)
        ax1.invert_yaxis()
        ax1.set_label("Inverted spike histogram")

    if len(cell_response.times) > 0:
        ax.set_xlim(left=0, right=cell_response.times[-1])
    else:
        ax.set_xlim(left=0)

    ax.set_ylabel("Counts")
    ax.set_label("Spike histogram")

    if ax1 is not None:
        # Combine legends
        handles, labels = ax.get_legend_handles_labels()
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles.extend(handles1)
        labels.extend(labels1)

        ax1.legend(handles, labels, loc="upper left")
    else:
        ax.legend()

    plt_show(show)
    return ax.get_figure()


def plot_lfp_morph_csd(
    times,
    data_lfp,
    data_csd,
    contact_labels,
    net,
    # LFP params
    decim=None,
    color_lfp="cividis",
    baseline_samps=None,
    scale_lfp=1.0,
    voltage_scalebar=200,
    # morphology params
    cell_types=('L2_pyramidal', 'L5_pyramidal'),
    gid=None,
    diam_scale=5.0,
    # CSD params
    colorbar=True,
    vmin=None,
    vmax=None,
    sink="b",
    interpolation="spline",
    overlay_csd_traces=False,
    scale_csd_traces=1.0,
    overlay_lfp_traces=False,
    scale_lfp_traces=1.0,
    unit_csd="µV/µm²",
    # layout
    morph_width_ratio=1.5,
    figsize=None,
    show=True,
    ext_inputs=None,
    spike_types=None,
    overlay_raster_on_csd=False,
):
    import matplotlib.pyplot as plt

    contact_labels = np.asarray(contact_labels, dtype=float)

    if ext_inputs is not None:
        fig, axes = plt.subplot_mosaic(
            [['hist_lfp', '.', 'hist_csd'],
            ['lfp',      'morph', 'csd'     ]],
            gridspec_kw={
                'width_ratios': [4, morph_width_ratio, 4],
                'height_ratios': [1, 4],
            },
            constrained_layout=True,
            figsize=figsize,
        )
        ax_lfp      = axes['lfp']
        ax_morph    = axes['morph']
        ax_csd      = axes['csd']
        ax_hist_lfp = axes['hist_lfp']
        ax_hist_csd = axes['hist_csd']

        spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']}
        spike_colors={'Distal': 'green', 'Proximal': 'red'}

        hist_kwargs = dict(
            cell_response=ext_inputs,
            spike_types=spike_types,
            color=spike_colors,
            show=False,
        )
        plot_spikes_hist(ax=ax_hist_lfp, **hist_kwargs)
        plot_spikes_hist(ax=ax_hist_csd, **hist_kwargs)
        ax_hist_lfp.set_xlabel('')
        ax_hist_csd.set_xlabel('')
    else:
        fig, (ax_lfp, ax_morph, ax_csd) = plt.subplots(
            1, 3,
            gridspec_kw={'width_ratios': [4, morph_width_ratio, 4]},
            constrained_layout=True,
            figsize=figsize,
        )

    plot_laminar_lfp_AC(
        times, 
        data_lfp, 
        contact_labels,
        ax=ax_lfp,
        decim=decim, 
        color=color_lfp,
        baseline_samps=baseline_samps, 
        scale=scale_lfp,
        voltage_scalebar=voltage_scalebar, 
        show=False,
    )

    plot_cell_morphology_for_lfp_csd(
        net, 
        contact_labels,
        cell_types=cell_types, 
        gid=gid, 
        diam_scale=diam_scale,
        ax=ax_morph, 
        show=False,
    )

    contact_labels_sorted = np.sort(contact_labels)
    z_max = contact_labels_sorted.max()
    depth_spacing = np.diff(contact_labels_sorted)[0]
    ax_morph.set_yticks(contact_labels_sorted)
    ax_morph.set_yticklabels([f"{d:g}" for d in (z_max - contact_labels_sorted)])
    ax_morph.set_ylim(contact_labels_sorted.min() - depth_spacing,
                    contact_labels_sorted.max() + depth_spacing)
    ax_morph.set_ylabel("Depth from surface (µm)")

    plot_laminar_csd_AC(
        times, 
        data_csd, 
        contact_labels,
        ax=ax_csd, 
        colorbar=colorbar, 
        vmin=vmin, 
        vmax=vmax,
        sink=sink, 
        interpolation=interpolation,
        overlay_csd_traces=overlay_csd_traces,
        scale_csd_traces=scale_csd_traces,
        data_lfp=data_lfp, 
        overlay_lfp_traces=overlay_lfp_traces,
        scale_lfp_traces=scale_lfp_traces, 
        unit_csd=unit_csd, 
        overlay_raster_on_csd=overlay_raster_on_csd,
        cell_response=net.cell_response,
        show=False,
    )

    plt_show(show)
    return fig

'''
def make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_from_sources_, csd_cap_, csd_ionic_, csd_syn_,
    csd_syn_gabab_, csd_syn_gabaa_, csd_syn_ampa_, csd_syn_nmda_,
    suptitle="", vmax_row0=None, vmax_row1=None,
):
    """Build the 2x6 CSD-contribution summary figure"""
    import matplotlib.pyplot as plt

    titles = [
        ["Morphology", "Total (agg_i_mem)", "Capacitive", "Ionic", "Synaptic", ""],
        ["residual (Tot-(cap+ion+syn))",   "GABA-B",            "GABA-A",     "AMPA",  "NMDA",     ""],
    ]

    fig, axes = plt.subplots(
        2, 6,
        constrained_layout=True,
        figsize=(28, 8),
        gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.25]},
    )

    # Morphology
    plot_cell_morphology_for_lfp_csd(
        net,
        contact_positions=contact_labels,
        cell_types=('L2_pyramidal', 'L5_pyramidal'),
        ax=axes[0, 0],
        show=False,
    )

    # Reconstructed total = synaptic + capacitive + ionic; residual vs. the agg_i_mem ground truth
    csd_reconstructed_ = csd_syn_ + csd_cap_ + csd_ionic_
    csd_residual_ = csd_from_sources_ - csd_reconstructed_

    # auto-scale per figure, unless an explicit scale is passed in
    if vmax_row0 is None:
        vmax_row0 = max(np.max(np.abs(csd_from_sources_)),
                         np.max(np.abs(csd_cap_)),
                         np.max(np.abs(csd_ionic_)),
                         np.max(np.abs(csd_syn_)))
    vmin_row0 = -vmax_row0

    if vmax_row1 is None:
        vmax_row1 = max(np.max(np.abs(csd_residual_)),
                         np.max(np.abs(csd_syn_gabab_)),
                         np.max(np.abs(csd_syn_gabaa_)),
                         np.max(np.abs(csd_syn_ampa_)),
                         np.max(np.abs(csd_syn_nmda_)))
    vmin_row1 = -vmax_row1

    csd_panels_row0 = [
        (axes[0, 1], csd_from_sources_),
        (axes[0, 2], csd_cap_),
        (axes[0, 3], csd_ionic_),
        (axes[0, 4], csd_syn_),
    ]
    csd_panels_row1 = [
        (axes[1, 0], csd_residual_),
        (axes[1, 1], csd_syn_gabab_),
        (axes[1, 2], csd_syn_gabaa_),
        (axes[1, 3], csd_syn_ampa_),
        (axes[1, 4], csd_syn_nmda_),
    ]

    for ax, data in csd_panels_row0:
        plot_laminar_csd_AC(
            times_, data, contact_labels,
            ax=ax, vmin=vmin_row0, vmax=vmax_row0,
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="red",
            colorbar=False,
            show=False,
        )

    for ax, data in csd_panels_row1:
        plot_laminar_csd_AC(
            times_, data, contact_labels,
            ax=ax, vmin=vmin_row1, vmax=vmax_row1,
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="red",
            colorbar=False,
            show=False,
        )

    # one shared colorbar per row, hosted in the leftover 6th column
    axes[0, 5].axis('off')
    cax0 = axes[0, 5].inset_axes([0.35, 0.15, 0.06, 0.7])
    cbar0 = fig.colorbar(axes[0, 1].collections[-1], cax=cax0)
    cbar0.set_label("CSD (µA/mm³)")

    axes[1, 5].axis('off')
    cax1 = axes[1, 5].inset_axes([0.35, 0.15, 0.06, 0.7])
    cbar1 = fig.colorbar(axes[1, 0].collections[-1], cax=cax1)
    cbar1.set_label("CSD (µA/mm³)")

    # Titles
    for row, row_titles in enumerate(titles):
        for col, title in enumerate(row_titles):
            axes[row, col].set_title(title)

    # clear x/y labels on the data panels only — leave the two colorbar axes alone
    for row in range(2):
        for col in range(5):
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')

    fig.suptitle(suptitle)
    return fig
'''

'''
def make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_from_sources_, csd_cap_, csd_ionic_, csd_syn_,
    csd_syn_gabab_, csd_syn_gabaa_, csd_syn_ampa_, csd_syn_nmda_,
    suptitle="", vmax_row0=None, vmax_row1=None, vmax_residual=None,
):
    """Build the 2x6 CSD-contribution summary figure"""
    import matplotlib.pyplot as plt

    titles = [
        ["Morphology", "Total (agg_i_mem)", "Capacitive", "Ionic", "Synaptic", ""],
        ["residual (Tot-(cap+ion+syn))",   "GABA-B",            "GABA-A",     "AMPA",  "NMDA",     ""],
    ]

    fig, axes = plt.subplots(
        2, 6,
        constrained_layout=True,
        figsize=(28, 8),
        gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.25]},
    )

    # Morphology
    plot_cell_morphology_for_lfp_csd(
        net,
        contact_positions=contact_labels,
        cell_types=('L2_pyramidal', 'L5_pyramidal'),
        ax=axes[0, 0],
        show=False,
    )

    # Reconstructed total = synaptic + capacitive + ionic; residual vs. the agg_i_mem ground truth
    csd_reconstructed_ = csd_syn_ + csd_cap_ + csd_ionic_
    csd_residual_ = csd_from_sources_ - csd_reconstructed_

    # auto-scale per figure, unless an explicit scale is passed in
    if vmax_row0 is None:
        vmax_row0 = max(np.max(np.abs(csd_from_sources_)),
                         np.max(np.abs(csd_cap_)),
                         np.max(np.abs(csd_ionic_)),
                         np.max(np.abs(csd_syn_)))
    vmin_row0 = -vmax_row0

    # residual now gets its own independent color scale
    if vmax_residual is None:
        vmax_residual = np.max(np.abs(csd_residual_))
    vmin_residual = -vmax_residual

    if vmax_row1 is None:
        vmax_row1 = max(np.max(np.abs(csd_syn_gabab_)),
                         np.max(np.abs(csd_syn_gabaa_)),
                         np.max(np.abs(csd_syn_ampa_)),
                         np.max(np.abs(csd_syn_nmda_)))
    vmin_row1 = -vmax_row1

    csd_panels_row0 = [
        (axes[0, 1], csd_from_sources_),
        (axes[0, 2], csd_cap_),
        (axes[0, 3], csd_ionic_),
        (axes[0, 4], csd_syn_),
    ]
    csd_panels_row1 = [
        (axes[1, 1], csd_syn_gabab_),
        (axes[1, 2], csd_syn_gabaa_),
        (axes[1, 3], csd_syn_ampa_),
        (axes[1, 4], csd_syn_nmda_),
    ]

    for ax, data in csd_panels_row0:
        plot_laminar_csd_AC(
            times_, data, contact_labels,
            ax=ax, vmin=vmin_row0, vmax=vmax_row0,
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="red",
            colorbar=False,
            show=False,
        )

    # residual plotted separately with its own vmin/vmax
    plot_laminar_csd_AC(
        times_, csd_residual_, contact_labels,
        ax=axes[1, 0], vmin=vmin_residual, vmax=vmax_residual,
        overlay_csd_traces=True,
        unit_csd="µA/mm³",
        sink="red",
        colorbar=False,
        show=False,
    )

    for ax, data in csd_panels_row1:
        plot_laminar_csd_AC(
            times_, data, contact_labels,
            ax=ax, vmin=vmin_row1, vmax=vmax_row1,
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="red",
            colorbar=False,
            show=False,
        )

    # one shared colorbar per row, hosted in the leftover 6th column
    axes[0, 5].axis('off')
    cax0 = axes[0, 5].inset_axes([0.35, 0.15, 0.06, 0.7])
    cbar0 = fig.colorbar(axes[0, 1].collections[-1], cax=cax0)
    cbar0.set_label("CSD (µA/mm³)")

    axes[1, 5].axis('off')
    cax1 = axes[1, 5].inset_axes([0.35, 0.15, 0.06, 0.7])
    cbar1 = fig.colorbar(axes[1, 1].collections[-1], cax=cax1)
    cbar1.set_label("CSD (µA/mm³)")

    # separate colorbar for the residual panel, since it has its own scale
    cax_residual = axes[1, 0].inset_axes([1.05, 0.15, 0.06, 0.7])
    cbar_residual = fig.colorbar(axes[1, 0].collections[-1], cax=cax_residual)
    cbar_residual.set_label("CSD (µA/mm³)")

    # Titles
    for row, row_titles in enumerate(titles):
        for col, title in enumerate(row_titles):
            axes[row, col].set_title(title)

    # clear x/y labels on the data panels only — leave the two colorbar axes alone
    for row in range(2):
        for col in range(5):
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')

    fig.suptitle(suptitle)
    return fig
'''

def make_csd_contribution_summary_figure(
    net, contact_labels, times_,
    csd_from_sources_, csd_cap_, csd_ionic_, csd_syn_,
    csd_syn_gabab_, csd_syn_gabaa_, csd_syn_ampa_, csd_syn_nmda_,
    suptitle="", vmax_row0=None, vmax_row1=None, vmax_residual=None,
    cell_response=None, overlay_raster=False, raster_colors=None,
):
    """Build the 2x6 CSD-contribution summary figure"""
    import matplotlib.pyplot as plt

    titles = [
        ["Morphology", "Total (agg_i_mem)", "Capacitive", "Ionic", "Synaptic", ""],
        ["residual (Tot-(cap+ion+syn))",   "GABA-B",            "GABA-A",     "AMPA",  "NMDA",     ""],
    ]

    fig, axes = plt.subplots(
        2, 6,
        constrained_layout=True,
        figsize=(28, 8),
        gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.25]},
    )

    # Morphology
    plot_cell_morphology_for_lfp_csd(
        net,
        contact_positions=contact_labels,
        cell_types=('L2_pyramidal', 'L5_pyramidal'),
        ax=axes[0, 0],
        show=False,
    )

    # Reconstructed total = synaptic + capacitive + ionic; residual vs. the agg_i_mem ground truth
    csd_reconstructed_ = csd_syn_ + csd_cap_ + csd_ionic_
    csd_residual_ = csd_from_sources_ - csd_reconstructed_

    # auto-scale per figure, unless an explicit scale is passed in
    if vmax_row0 is None:
        vmax_row0 = max(np.max(np.abs(csd_from_sources_)),
                         np.max(np.abs(csd_cap_)),
                         np.max(np.abs(csd_ionic_)),
                         np.max(np.abs(csd_syn_)))
    vmin_row0 = -vmax_row0

    # residual now gets its own independent color scale
    if vmax_residual is None:
        vmax_residual = np.max(np.abs(csd_residual_))
    vmin_residual = -vmax_residual

    if vmax_row1 is None:
        vmax_row1 = max(np.max(np.abs(csd_syn_gabab_)),
                         np.max(np.abs(csd_syn_gabaa_)),
                         np.max(np.abs(csd_syn_ampa_)),
                         np.max(np.abs(csd_syn_nmda_)))
    vmin_row1 = -vmax_row1

    csd_panels_row0 = [
        (axes[0, 1], csd_from_sources_),
        (axes[0, 2], csd_cap_),
        (axes[0, 3], csd_ionic_),
        (axes[0, 4], csd_syn_),
    ]
    csd_panels_row1 = [
        (axes[1, 1], csd_syn_gabab_),
        (axes[1, 2], csd_syn_gabaa_),
        (axes[1, 3], csd_syn_ampa_),
        (axes[1, 4], csd_syn_nmda_),
    ]

    for ax, data in csd_panels_row0:
        # only overlay spikes on the "Total" panel (ax is axes[0, 1])
        do_raster = overlay_raster and (ax is axes[0, 1])
        plot_laminar_csd_AC(
            times_, data, contact_labels,
            ax=ax, vmin=vmin_row0, vmax=vmax_row0,
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="red",
            colorbar=False,
            overlay_raster_on_csd=do_raster,
            cell_response=cell_response if do_raster else None,
            raster_colors=raster_colors,
            show=False,
        )

    # residual plotted separately with its own vmin/vmax
    plot_laminar_csd_AC(
        times_, csd_residual_, contact_labels,
        ax=axes[1, 0], vmin=vmin_residual, vmax=vmax_residual,
        overlay_csd_traces=True,
        unit_csd="µA/mm³",
        sink="red",
        colorbar=False,
        show=False,
    )

    for ax, data in csd_panels_row1:
        plot_laminar_csd_AC(
            times_, data, contact_labels,
            ax=ax, vmin=vmin_row1, vmax=vmax_row1,
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="red",
            colorbar=False,
            show=False,
        )

    # one shared colorbar per row, hosted in the leftover 6th column
    axes[0, 5].axis('off')
    cax0 = axes[0, 5].inset_axes([0.35, 0.15, 0.06, 0.7])
    cbar0 = fig.colorbar(axes[0, 1].collections[-1], cax=cax0)
    cbar0.set_label("CSD (µA/mm³)")

    axes[1, 5].axis('off')
    cax1 = axes[1, 5].inset_axes([0.35, 0.15, 0.06, 0.7])
    cbar1 = fig.colorbar(axes[1, 1].collections[-1], cax=cax1)
    cbar1.set_label("CSD (µA/mm³)")

    # separate colorbar for the residual panel, since it has its own scale
    cax_residual = axes[1, 0].inset_axes([1.05, 0.15, 0.06, 0.7])
    cbar_residual = fig.colorbar(axes[1, 0].collections[-1], cax=cax_residual)
    cbar_residual.set_label("CSD (µA/mm³)")

    # Titles
    for row, row_titles in enumerate(titles):
        for col, title in enumerate(row_titles):
            axes[row, col].set_title(title)

    # clear x/y labels on the data panels only — leave the two colorbar axes alone
    for row in range(2):
        for col in range(5):
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')

    fig.suptitle(suptitle)
    return fig