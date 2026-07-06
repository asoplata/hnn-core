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
            tstop=70.0,
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

'''
# ------
# Checking if dipole HNN's output matches the dipole from agg_i_mem
# ------

# L5 pyramidal cells
test_imem_L5 = tme.postproc_tm_currents(
    net=net,
    cell_type="L5_pyramidal",
    from_components=False,
)

fig = dpl.plot(layer="L5") # plot HNN's dipole computed from axial currents
ax = fig.axes[0] # add plot from aggregated transmembrane currents (agg_i_mem)
ax.plot(
    dpl.times[1:],
    test_imem_L5[1:],
    label="test_imem",
    color="tab:red",
    lw=1.5,
)
ax.legend()

#L2/3 pyramidal cells
test_imem_L2 = tme.postproc_tm_currents(
    net=net,
    cell_type="L2_pyramidal",
    from_components=False,
)

fig = dpl.plot(layer="L2")
ax = fig.axes[0]
ax.plot(
    dpl.times[1:],
    test_imem_L2[1:],
    label="test_imem",
    color="tab:red",
    lw=1.5,
)
ax.legend()

# RESULT: there's a discrepancy in the dipole for L5 pyramidal cells.
# WIP: checking whether this is due to a difference in the currents included in agg_i_mem vs the axial currents, or if there's a bug in the post-processing of the currents to compute the dipole.


# ------
# Checking if HNN's LFP matches the LFP from agg_i_mem
# ------
lfp_hnn_ = net.rec_arrays["probe1"].voltages[0]  # HNN's LFP; trial 0

sources_agg, I_agg = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L2_pyramidal","L5_pyramidal"],
    channels=["agg_i_mem"],
)

T_agg = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_agg,
    array_name="probe1",
)

lfp_agg_i_mem_ = tme.reconstruct_lfp_from_sources(
    T_agg,
    I_agg,
)

lfp_hnn = lfp_hnn_[:,1:]
lfp_agg_i_mem = lfp_agg_i_mem_[:,1:]

residual = lfp_hnn - lfp_agg_i_mem

print("max abs diff:", np.max(np.abs(residual)))
print("mean abs diff:", np.mean(np.abs(residual)))
print("relative max diff:", np.max(np.abs(residual)) / np.max(np.abs(residual)))

times_plot = times[1:]
contact_idx = 21  # choose electrode/contact

plt.figure(figsize=(8, 4))
plt.plot(times_plot, lfp_hnn[contact_idx], label="HNN LFP")
plt.plot(times_plot, lfp_agg_i_mem[contact_idx], "--", label="agg_i_mem reconstructed")

plt.xlabel("Time (ms)")
plt.ylabel("LFP")
plt.title(f"LFP comparison, contact {contact_idx}, depth {depths[contact_idx]} µm")
plt.legend()
plt.tight_layout()
plt.show()

# plot of all the contacts:
positions = np.asarray(net.rec_arrays["probe1"].positions)
z = positions[:, 2]
n_contacts = lfp_agg_i_mem.shape[0]

# single global scaling factor (applies to both signals)
gain = 0.005

lfp_fig_  = lfp_hnn * gain
lfp_agg_i_mem_fig_ = lfp_agg_i_mem * gain
offset = 1.0   # fixed spacing between traces

# scale bar in µV
scale_uV = 200.0
bar_height_visual = scale_uV * gain

fig, ax = plt.subplots(figsize=(8, 5))
for c in range(n_contacts):
    ax.plot(times[1:], lfp_fig_[c,:]  + c * offset,
            color="k", lw=0.8,
            label="HNN's LFP" if c == 0 else None)
    ax.plot(times[1:], lfp_agg_i_mem_fig_[c, :] + c * offset,
            color="tab:red", lw=0.8, alpha=0.8,
            label="lfp_agg_i_mem" if c == 0 else None)

ax.set_yticks([c * offset for c in range(n_contacts)])
ax.set_yticklabels([f"{int(z[c])} µm" for c in range(n_contacts)], fontsize=9)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth z")
ax.set_title("LFP per contact — HNN's LFP vs agg_i_mem reconstruction")

# legend (only one entry per signal)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# scale bar near the left edge, just below the first trace
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

# RESULT: HNN's LFP matches the LFP reconstructed from agg_i_mem, which suggests that the transfer resistance calculations and LFP reconstruction are working correctly. This is a prerequisite before checking the decomposition of agg_i_mem into intrinsic and synaptic components at the soma.

# ------
# [Validation for dipole] Checking, at the L5 pyr cell soma, if dipole obtained from the sum of capacitive, ionic, and synaptic currents  
# matches the dipole from agg_i_mem
# ------

#tme.check_rmse_and_residuals(net) # Dylan's approach

test_agg_i_mem_L5, test_I_t_over_gid_mem, I_syn_over_gid_mem, I_cap_intr_over_gid_mem = tme.postproc_soma_dipole_AC(
    net=net,
    from_components=False,
)

test_comp_L5, test_comp_I_t_over_gid, I_syn_over_gid_comp, I_cap_intr_over_gid_comp = tme.postproc_soma_dipole_AC(
    net=net,
    from_components=True,
)

plt.figure()
plt.plot(times[1:],test_agg_i_mem_L5[1:], label = 'agg_i_mem')
plt.plot(times[1:],test_comp_L5[1:], label = 'cap + ionic + syn')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('dipole (nAm)')


plt.figure()
plt.plot(times[1:], test_agg_i_mem_L5[1:]-test_comp_L5[1:], label = 'residual agg_i_mem - (cap+intr+syn)')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('[nAm]')

plt.figure() # summed across cells, only for the current at the soma, not for the dipole
plt.plot(times[1:], test_I_t_over_gid_mem[1:], label = 'agg_i_mem') # only L5 pyr cell soma
plt.plot(times[1:],test_comp_I_t_over_gid[1:], label = 'cap + ionic + syn')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('current [nA]')
plt.title('current at soma: agg_i_mem vs sum of components')


plt.figure()
plt.plot(times[1:],test_I_t_over_gid_mem[1:]-test_comp_I_t_over_gid[1:], label = 'residual agg_i_mem - (cap+intr+syn)')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('current [nA]')

total_curr = I_syn_over_gid_comp + I_cap_intr_over_gid_comp
residual = test_comp_I_t_over_gid - total_curr
plt.figure()
plt.plot(times[1:],residual[1:])

plt.figure()
plt.plot(times[1:],I_syn_over_gid_comp[1:], label = 'synaptic current')
plt.plot(times[1:],I_cap_intr_over_gid_comp[1:], label = 'capacitive + ionic current')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('current [nA]')
plt.title('currents at soma')

# RESULTS:
# 1. residual small at the dipole level becomes bigger at the current level
# 2. postproc_soma_dipole_AC_v2 in tme is working as intended


# ------
# [Validation for LFP] Checking, at the L5 pyr cell soma, if the LFP obtained from the sum of capacitive, ionic, and synaptic currents  
# matches the LFP from agg_i_mem
# ------

# 1. agg_i_mem, L5 pyr cells 
sources_agg, I_agg = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=["agg_i_mem"],
)

# soma only
sources_agg_soma, I_agg_soma = tme.filter_sources(
    sources_agg,
    I_agg,
    cell_types=["L5_pyramidal"],
    sections=["soma"],
)

I_agg_soma_sum = np.sum(I_agg_soma, axis=0) # matches previous analysis with Dylan's code!

# 2. sum of capacitive + ionic components 
sources_intr, I_intr = tme.collect_intrinsic_sources(
    net,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
    channels=l5_component_channels,
    cap_current_sign=1.0,
)

# soma only
sources_intr_soma, I_intr_soma = tme.filter_sources(
    sources_intr,
    I_intr,
    cell_types=["L5_pyramidal"],
    sections=["soma"],
)

I_intr_soma_sum = np.sum(I_intr_soma, axis=0)


#3. sum of synaptic currents, soma only
sources_syn, I_syn = tme.collect_synaptic_sources(
    net,
    trial_idx=0,
    cell_types=["L5_pyramidal"],
)

sources_syn_soma, I_syn_soma = tme.filter_sources(
    sources_syn,
    I_syn,
    cell_types=["L5_pyramidal"],
    sections=["soma"],
)

I_syn_soma_sum = np.sum(I_syn_soma, axis=0)

I_intr_soma_sum = I_intr_soma_sum[1:]
I_syn_soma_sum = I_syn_soma_sum[1:]
I_agg_soma_sum = I_agg_soma_sum[1:]

I_reconstructed_soma = I_intr_soma_sum + I_syn_soma_sum

residual = I_agg_soma_sum - I_reconstructed_soma

print("max abs diff:", np.max(np.abs(residual)))
print("mean abs diff:", np.mean(np.abs(residual)))
print("max abs agg:", np.max(np.abs(I_agg_soma_sum)))
print("relative max diff:", np.max(np.abs(residual)) / np.max(np.abs(I_agg_soma_sum)))

plt.figure()
plt.plot(residual)
# RESULT: this graph shows that the functions above to look at currents
# reproduce exactly what seen in the extension of Dylan's approach (see above)

T_agg_soma = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_agg_soma,
    array_name="probe1",
)

lfp_agg_soma = tme.reconstruct_lfp_from_sources(
    T_agg_soma,
    I_agg_soma,
)

T_intr_soma = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_intr_soma,
    array_name="probe1",
)

lfp_intr_soma = tme.reconstruct_lfp_from_sources(
    T_intr_soma,
    I_intr_soma,
)

T_syn_soma = tme.build_transfer_resistance_matrix_for_sources(
    net,
    sources_syn_soma,
    array_name="probe1",
)

lfp_syn_soma = tme.reconstruct_lfp_from_sources(
    T_syn_soma,
    I_syn_soma,
)

contact_number = 8  # choose one electrode/contact
plt.figure()
plt.plot(times[1:], lfp_agg_soma[contact_number,1:], label="agg_i_mem")
plt.plot(times[1:], lfp_intr_soma[contact_number,1:], "--", label="cap + ionic")  
plt.plot(times[1:], lfp_syn_soma[contact_number,1:], ":", label="syn, L5 soma")  
plt.plot(times[1:], lfp_intr_soma[contact_number,1:] + lfp_syn_soma[contact_number,1:], "--", label="cap + ionic + syn")
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("LFP (µV)")
plt.title('LFP at contact 10 (z=375 µm), L5 soma only')

# plot of all the contacts:
# ALL THE TRACES — agg_i_mem vs (cap + ionic + synaptic):
positions = np.asarray(net.rec_arrays["probe1"].positions)
z = positions[:, 2]
n_contacts = lfp_agg_soma.shape[0]

# single global scaling factor (applies to both signals)
gain = 0.005

traces_agg  = lfp_agg_soma * gain
traces_comp = (lfp_intr_soma + lfp_syn_soma) * gain
offset = 1.0   # fixed spacing between traces in visual units

# --- scale bar in µV ---
scale_uV = 200.0
bar_height_visual = scale_uV * gain

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(n_contacts):
    ax.plot(times[1:], traces_agg[c, 1:]  + c * offset,
            color="k", lw=0.8,
            label="agg_i_mem" if c == 0 else None)
    ax.plot(times[1:], traces_comp[c, 1:] + c * offset,
            color="tab:red", lw=0.8, alpha=0.8,
            label="cap + ionic + synaptic" if c == 0 else None)

ax.set_yticks([c * offset for c in range(n_contacts)])
ax.set_yticklabels([f"{int(z[c])} µm" for c in range(n_contacts)], fontsize=9)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth z")
ax.set_title("LFP per contact — agg_i_mem vs cap + ionic + synaptic")

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


# compare summed components against agg_i_mem
lfp_components_sum_soma = lfp_intr_soma + lfp_syn_soma

lfp_error_soma = lfp_agg_soma - lfp_components_sum_soma
plt.figure()
plt.plot(times[1:], lfp_error_soma[contact_number,1:], label="difference")
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("LFP error (µV)")
plt.title('Residual in the LFP at contact 8')

# residual for all the contacts
positions = np.asarray(net.rec_arrays["probe1"].positions)
z = positions[:, 2]
n_contacts = lfp_error_soma.shape[0]

# single global scaling factor
gain = 0.05

traces = lfp_error_soma * gain
offset = 1.0   # fixed spacing between traces in visual units

# scale bar represents some round µV value
scale_uV = 30.0                    
bar_height_visual = scale_uV * gain   

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(n_contacts):
    ax.plot(times[1:], traces[c, 1:] + c * offset, color="k", lw=0.8)

ax.set_yticks([c * offset for c in range(n_contacts)])
ax.set_yticklabels([f"{int(z[c])} µm" for c in range(n_contacts)], fontsize=9)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Depth z")
ax.set_title(f"LFP error per contact")

# --- scale bar near the left edge, just below the first trace ---
x_bar = times[1] + (times[-1] - times[1]) * 0.015
y_bar_bottom = -1.0 * offset
ax.plot([x_bar, x_bar],
        [y_bar_bottom, y_bar_bottom + bar_height_visual],
        color="red", lw=4, solid_capstyle="butt")
ax.text(x_bar + (times[-1] - times[1]) * 0.01,
        y_bar_bottom + bar_height_visual / 2,
        f"{scale_uV:g} µV",
        color="red", ha="left", va="center",
        fontsize=11, fontweight="bold")

plt.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.05)


# QUANTIFY THE ERROR FOR THE RECONTRUCTION AT THE SOMA
# mimics check_rmse_and_residuals() in tm_currents_utils.py

residual_lfp_soma = lfp_agg_soma[:,1:] - lfp_components_sum_soma[:,1:]
rmse_per_contact = np.sqrt(np.mean(residual_lfp_soma**2, axis=1))

positions = np.asarray(net.rec_arrays["probe1"].positions)
z = positions[:, 2]

fig, ax = plt.subplots(figsize=(4, 10))
ax.plot(rmse_per_contact, z, "ko-")
ax.set_xlabel("RMSE (µV)")
ax.set_ylabel("Depth z (µm)")
ax.set_title("LFP error magnitude per contact")

# normalize by the peak-to-peak range of agg_i_mem (our "ground truth")
# peak-to-peak per contact (use np.ptp as a shortcut; equivalent to max - min along axis=1)
lfp_range_per_contact = np.ptp(lfp_agg_soma[:,1:], axis=1)   # shape (n_contacts,)

# normalized RMSE per contact, in percent; safe against zero-range contacts
nrmse_pct_per_contact = np.where(
    lfp_range_per_contact != 0,
    (rmse_per_contact / lfp_range_per_contact) * 100.0,
    0.0,
)

positions = np.asarray(net.rec_arrays["probe1"].positions)
z = positions[:, 2]

fig, ax = plt.subplots(figsize=(4, 10))
ax.plot(nrmse_pct_per_contact, z, "ko-")
ax.set_xlabel("NRMSE (% of peak-to-peak)")
ax.set_ylabel("Depth z (µm)")
ax.set_title("Normalized LFP error per contact")
ax.axvline(0, color="lightgray", lw=0.7)

'''

# START HERE IMPLEMENTING VALIDATION FOR ANOTHER COMPARTMENT!
# Dylan is taking over with this. 

# to be deleted:
test_comp_L5, test_comp_I_t_over_gid, I_syn_over_gid_comp, I_cap_intr_over_gid_comp = tme.postproc_soma_dipole_AC(
    net=net,
    from_components=True,
)
#



test_agg_i_mem_L5, test_I_t_over_gid_mem, I_syn_over_gid_mem, I_cap_ionic_over_gid_mem = tme.postproc_oblique_dipole_AC(
    net=net,
    from_components=False,
)

test_comp_L5, test_comp_I_t_over_gid, I_syn_over_gid_comp, I_cap_ionic_over_gid_comp = tme.postproc_oblique_dipole_AC(
    net=net,
    from_components=True,
)

plt.figure()
plt.plot(times[1:],test_agg_i_mem_L5[1:], label = 'agg_i_mem')
plt.plot(times[1:],test_comp_L5[1:], label = 'cap + ionic + syn')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('dipole (nAm)')


plt.figure()
plt.plot(times[1:], test_agg_i_mem_L5[1:]-test_comp_L5[1:], label = 'residual agg_i_mem - (cap+intr+syn)')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('[nAm]')

plt.figure() # summed across cells, only for the current at the soma, not for the dipole
plt.plot(times[1:], test_I_t_over_gid_mem[1:], label = 'agg_i_mem') # only L5 pyr cell soma
plt.plot(times[1:],test_comp_I_t_over_gid[1:], label = 'cap + ionic + syn')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('current [nA]')
plt.title('current at soma: agg_i_mem vs sum of components')


plt.figure()
plt.plot(times[1:],test_I_t_over_gid_mem[1:]-test_comp_I_t_over_gid[1:], label = 'residual agg_i_mem - (cap+intr+syn)')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('current [nA]')

total_curr = I_syn_over_gid_comp + I_cap_intr_over_gid_comp
residual = test_comp_I_t_over_gid - total_curr
plt.figure()
plt.plot(times[1:],residual[1:])

plt.figure()
plt.plot(times[1:],I_syn_over_gid_comp[1:], label = 'synaptic current')
plt.plot(times[1:],I_cap_intr_over_gid_comp[1:], label = 'capacitive + ionic current')
plt.legend()
plt.xlabel('time [ms]')
plt.ylabel('current [nA]')
plt.title('currents at soma')
