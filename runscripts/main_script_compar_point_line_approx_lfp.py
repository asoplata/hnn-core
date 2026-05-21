import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_dipole_lfp as tme
#import tm_currents_utils_forLFP as tme_lfp

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

# Laminar probe
depths = np.arange(-625, 2150, 100)
electrode_pos = [(135, 135, z) for z in depths]

net.add_electrode_array("probe_psa", electrode_pos, method="psa") #<- LFP with point-source approx
net.add_electrode_array("probe_lsa", electrode_pos, method="lsa") #<- LFP with line-source approx

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

lfp_psa_ = np.asarray(net.rec_arrays["probe_psa"].voltages)[0]   # (n_contacts, n_times)
lfp_lsa_ = np.asarray(net.rec_arrays["probe_lsa"].voltages)[0]


positions = np.asarray(net.rec_arrays["probe_psa"].positions)
z = positions[:, 2]
n_contacts = lfp_psa_.shape[0]

# single global scaling factor (applies to both signals)
gain = 0.005

lfp_psa  = lfp_psa_ * gain
lfp_lsa = lfp_lsa_ * gain
offset = 1.0   # fixed spacing between traces in visual units

# --- scale bar in µV ---
scale_uV = 200.0
bar_height_visual = scale_uV * gain

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(n_contacts):
    ax.plot(times[1:], lfp_psa[c, 1:]  + c * offset,
            color="k", lw=0.8,
            label="LFP psa" if c == 0 else None)
    ax.plot(times[1:], lfp_lsa[c, 1:] + c * offset,
            color="tab:red", lw=0.8, alpha=0.8,
            label="LFP lsa" if c == 0 else None)

ax.set_yticks([c * offset for c in range(n_contacts)])
ax.set_yticklabels([f"{int(z[c])} µm" for c in range(n_contacts)], fontsize=9)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth z")
ax.set_title("LFP per contact ")

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


residual_ = lfp_lsa_ - lfp_psa_

gain = 0.5

residual  = residual_ * gain
offset = 1.0   # fixed spacing between traces in visual units

rmse = np.sqrt(np.mean(residual ** 2))
lfp_lsa_range = np.max(lfp_lsa_) - np.min(lfp_lsa_)
nrmse_pct = (rmse / lfp_lsa_range * 100) if lfp_lsa_range != 0 else 0.0

# --- scale bar in µV ---
scale_uV = 5.0
bar_height_visual = scale_uV * gain

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(n_contacts):
    ax.plot(times[1:], residual[c, 1:]  + c * offset,
            color="k", lw=0.8,
            label="residual" if c == 0 else None)

ax.set_yticks([c * offset for c in range(n_contacts)])
ax.set_yticklabels([f"{int(z[c])} µm" for c in range(n_contacts)], fontsize=9)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth z")

# --- legend (only one entry per signal) ---
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# --- RMSE / NRMSE text box (upper-left, axes-relative coords) ------------
stats_text = (
    f"RMSE = {rmse:.3f} µV\n"
    f"NRMSE = {nrmse_pct:.2f}% of LSA range"
)
ax.text(
    0.02, 0.98, stats_text,
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=10,
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.4",
              facecolor="white", edgecolor="0.6", alpha=0.9),
)

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

print(f"RMSE between LFP (lsa) and LFP (psa): {rmse:.4f} µV")
print(f"Normalized RMSE: {nrmse_pct:.2f}% of the LFP (lsa) range")