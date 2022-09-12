# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes
from matplotlib import ticker #for setting contour plots to log scale
from matplotlib.colors import LogNorm #for log color scales
import scipy.integrate #for numerical integration
import scipy.misc #for factorial function
from scipy.special import erf #error function, used in computing CDF of normal distribution
import scipy.interpolate #for interpolation functions
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/Edge_of_multis/'
run_number = ''
model_name = 'Maximum_AMD_Model' + run_number

compute_ratios = compute_ratios_adjacent
AD_mod = True
weights_all = load_split_stars_weights_only()
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 'period_ratios_KS',
                 'durations_KS',
                 #'durations_norm_circ_KS',
                 #'durations_norm_circ_singles_KS',
                 #'durations_norm_circ_multis_KS',
                 'duration_ratios_nonmmr_KS',
                 'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radius_ratios_KS',
                 #'radii_partitioning_KS',
                 #'radii_monotonicity_KS',
                 #'gap_complexity_KS',
                 ]





##### To load the files with the systems with observed planets:

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated observed catalog of stars and planets:
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)





##### To plot the simulated and Kepler catalogs as marginal distributions:

subdirectory = '' #'Paper_Figures/Models/Observed/Clustered_P_R/' #'Paper_Figures/'; 'Talk_Figures/'

fig_size = (8,8) # size of each panel (figure)
fig_lbrt = [0.15, 0.15, 0.95, 0.95]

n_bins = 100
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms/shaded regions

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size





##### To load and compute the same statistics for a large number of models:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
runs = 100

sss_per_sys_all = []
sss_all = []

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod)

    sss_per_sys_all.append(sss_per_sys_i)
    sss_all.append(sss_i)





##### To compute and plot CDFs of the outermost periods and period ratios for each multiplicity:

N_sys_Kep = len(ssk_per_sys['P_obs'])
P_out_per_sys_Kep = ssk_per_sys['P_obs'][np.arange(N_sys_Kep), ssk_per_sys['Mtot_obs']-1]
Pratio_out_per_sys_Kep = ssk_per_sys['Rm_obs'][np.arange(N_sys_Kep), ssk_per_sys['Mtot_obs']-2]

P_cdf_evals = np.logspace(np.log10(P_min), np.log10(P_max), n_bins+1)
Pratio_cdf_evals = np.logspace(np.log10(1.), np.log10(100.), n_bins+1)

P_out_cdf_evals_2_all = np.zeros((runs, len(P_cdf_evals)))
P_out_cdf_evals_3_all = np.zeros((runs, len(P_cdf_evals)))
P_out_cdf_evals_4p_all = np.zeros((runs, len(P_cdf_evals)))
Pratio_out_cdf_evals_2_all = np.zeros((runs, len(Pratio_cdf_evals)))
Pratio_out_cdf_evals_3_all = np.zeros((runs, len(Pratio_cdf_evals)))
Pratio_out_cdf_evals_4p_all = np.zeros((runs, len(Pratio_cdf_evals)))
for i in range(runs):
    sss_per_sys_i = sss_per_sys_all[i]
    N_sys = len(sss_per_sys_i['P_obs'])

    # Calculate CDFs of outermost period at P_cdf_evals:
    P_out_per_sys = sss_per_sys_i['P_obs'][np.arange(N_sys), sss_per_sys_i['Mtot_obs']-1]
    P_out_2 = P_out_per_sys[sss_per_sys_i['Mtot_obs'] == 2]
    P_out_3 = P_out_per_sys[sss_per_sys_i['Mtot_obs'] == 3]
    P_out_4p = P_out_per_sys[sss_per_sys_i['Mtot_obs'] >= 4]
    P_out_cdf_evals_2_all[i] = cdf_empirical(P_out_2, P_cdf_evals)
    P_out_cdf_evals_3_all[i] = cdf_empirical(P_out_3, P_cdf_evals)
    P_out_cdf_evals_4p_all[i] = cdf_empirical(P_out_4p, P_cdf_evals)

    # Calculate CDFs of outermost period ratio at Pratio_cdf_evals:
    Pratio_out_per_sys = sss_per_sys_i['Rm_obs'][np.arange(N_sys), sss_per_sys_i['Mtot_obs']-2]
    Pratio_out_2 = Pratio_out_per_sys[sss_per_sys_i['Mtot_obs'] == 2]
    Pratio_out_3 = Pratio_out_per_sys[sss_per_sys_i['Mtot_obs'] == 3]
    Pratio_out_4p = Pratio_out_per_sys[sss_per_sys_i['Mtot_obs'] >= 4]
    Pratio_out_cdf_evals_2_all[i] = cdf_empirical(Pratio_out_2, Pratio_cdf_evals)
    Pratio_out_cdf_evals_3_all[i] = cdf_empirical(Pratio_out_3, Pratio_cdf_evals)
    Pratio_out_cdf_evals_4p_all[i] = cdf_empirical(Pratio_out_4p, Pratio_cdf_evals)

qtls = [0.16,0.5,0.84]
P_out_cdf_evals_2_qtls = np.quantile(P_out_cdf_evals_2_all, qtls, axis=0)
P_out_cdf_evals_3_qtls = np.quantile(P_out_cdf_evals_3_all, qtls, axis=0)
P_out_cdf_evals_4p_qtls = np.quantile(P_out_cdf_evals_4p_all, qtls, axis=0)
Pratio_out_cdf_evals_2_qtls = np.quantile(Pratio_out_cdf_evals_2_all, qtls, axis=0)
Pratio_out_cdf_evals_3_qtls = np.quantile(Pratio_out_cdf_evals_3_all, qtls, axis=0)
Pratio_out_cdf_evals_4p_qtls = np.quantile(Pratio_out_cdf_evals_4p_all, qtls, axis=0)

# Outermost periods:
plot_fig_cdf_simple(fig_size, [], [P_out_per_sys_Kep[ssk_per_sys['Mtot_obs'] == 2]], x_min=P_min, x_max=P_max, log_x=True, lw=lw, labels_Kep=['Kepler'], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P_{\rm out}$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(P_cdf_evals, P_out_cdf_evals_2_qtls[0], P_out_cdf_evals_2_qtls[2], alpha=alpha, color='b', label=r'SysSim (central 68%)')
plt.plot(P_cdf_evals, P_out_cdf_evals_2_qtls[1], lw=lw, color='b', label=r'SysSim (median)')
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N = 2$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_outermost_N2_CDFs.pdf')
    plt.close()

plot_fig_cdf_simple(fig_size, [], [P_out_per_sys_Kep[ssk_per_sys['Mtot_obs'] == 3]], x_min=P_min, x_max=P_max, log_x=True, lw=lw, labels_Kep=['Kepler'], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P_{\rm out}$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(P_cdf_evals, P_out_cdf_evals_3_qtls[0], P_out_cdf_evals_3_qtls[2], alpha=alpha, color='b', label=r'SysSim (central 68%)')
plt.plot(P_cdf_evals, P_out_cdf_evals_3_qtls[1], lw=lw, color='b', label=r'SysSim (median)')
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N = 3$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_outermost_N3_CDFs.pdf')
    plt.close()

plot_fig_cdf_simple(fig_size, [], [P_out_per_sys_Kep[ssk_per_sys['Mtot_obs'] >= 4]], x_min=P_min, x_max=P_max, log_x=True, lw=lw, labels_Kep=['Kepler'], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P_{\rm out}$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(P_cdf_evals, P_out_cdf_evals_4p_qtls[0], P_out_cdf_evals_4p_qtls[2], alpha=alpha, color='b', label=r'SysSim (central 68%)')
plt.plot(P_cdf_evals, P_out_cdf_evals_4p_qtls[1], lw=lw, color='b', label=r'SysSim (median)')
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N \geq 4$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_outermost_N4p_CDFs.pdf')
    plt.close()

plt.show()

# Outermost period ratios:
plot_fig_cdf_simple(fig_size, [], [Pratio_out_per_sys_Kep[ssk_per_sys['Mtot_obs'] == 2]], x_min=1., x_max=80., log_x=True, lw=lw, labels_Kep=['Kepler'], xticks_custom=[1,2,3,5,10,20,40], xlabel_text=r'$P_{\rm out}/P_{\rm 2nd\ out}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.fill_between(Pratio_cdf_evals, Pratio_out_cdf_evals_2_qtls[0], Pratio_out_cdf_evals_2_qtls[2], alpha=alpha, color='b', label=r'SysSim (central 68%)')
plt.plot(Pratio_cdf_evals, Pratio_out_cdf_evals_2_qtls[1], lw=lw, color='b', label=r'SysSim (median)')
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N = 2$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periodratios_outermost_N2_CDFs.pdf')
    plt.close()

plot_fig_cdf_simple(fig_size, [], [Pratio_out_per_sys_Kep[ssk_per_sys['Mtot_obs'] == 3]], x_min=1., x_max=80., log_x=True, lw=lw, labels_Kep=['Kepler'], xticks_custom=[1,2,3,5,10,20,40], xlabel_text=r'$P_{\rm out}/P_{\rm 2nd\ out}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.fill_between(Pratio_cdf_evals, Pratio_out_cdf_evals_3_qtls[0], Pratio_out_cdf_evals_3_qtls[2], alpha=alpha, color='b', label=r'SysSim (central 68%)')
plt.plot(Pratio_cdf_evals, Pratio_out_cdf_evals_3_qtls[1], lw=lw, color='b', label=r'SysSim (median)')
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N = 3$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periodratios_outermost_N3_CDFs.pdf')
    plt.close()

plot_fig_cdf_simple(fig_size, [], [Pratio_out_per_sys_Kep[ssk_per_sys['Mtot_obs'] >= 4]], x_min=1., x_max=80., log_x=True, lw=lw, labels_Kep=['Kepler'], xticks_custom=[1,2,3,5,10,20,40], xlabel_text=r'$P_{\rm out}/P_{\rm 2nd\ out}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.fill_between(Pratio_cdf_evals, Pratio_out_cdf_evals_4p_qtls[0], Pratio_out_cdf_evals_4p_qtls[2], alpha=alpha, color='b', label=r'SysSim (central 68%)')
plt.plot(Pratio_cdf_evals, Pratio_out_cdf_evals_4p_qtls[1], lw=lw, color='b', label=r'SysSim (median)')
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N \geq 4$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periodratios_outermost_N4p_CDFs.pdf')
    plt.close()

plt.show()



##### To compute and plot CDFs of the periods of planets at each position (e.g. first, second, and third planet for triples) for a given multiplicity:

colors_ith_planet = ['b','g','k','r']

P_i_cdf_evals_1_all = np.zeros((runs, len(P_cdf_evals)))
P_i_cdf_evals_2_all = [np.zeros((runs, len(P_cdf_evals))) for i in range(2)]
P_i_cdf_evals_3_all = [np.zeros((runs, len(P_cdf_evals))) for i in range(3)]
P_i_cdf_evals_4_all = [np.zeros((runs, len(P_cdf_evals))) for i in range(4)]
for i in range(runs):
    sss_per_sys_i = sss_per_sys_all[i]
    N_sys = len(sss_per_sys_i['P_obs'])

    # Calculate CDFs of periods for planets at each position, at P_cdf_evals:
    P_per_sys_1 = sss_per_sys_i['P_obs'][sss_per_sys_i['Mtot_obs'] == 1, 0]
    P_i_cdf_evals_1_all[i] = cdf_empirical(P_per_sys_1, P_cdf_evals)

    P_per_sys_2 = sss_per_sys_i['P_obs'][sss_per_sys_i['Mtot_obs'] == 2]
    for j in range(2): # loop through first and second planet
        P_i_cdf_evals_2_all[j][i] = cdf_empirical(P_per_sys_2[:,j], P_cdf_evals) # j-th planet

    P_per_sys_3 = sss_per_sys_i['P_obs'][sss_per_sys_i['Mtot_obs'] == 3]
    for j in range(3): # loop through first, second, and third planet
        P_i_cdf_evals_3_all[j][i] = cdf_empirical(P_per_sys_3[:,j], P_cdf_evals) # j-th planet

    P_per_sys_4 = sss_per_sys_i['P_obs'][sss_per_sys_i['Mtot_obs'] == 4]
    for j in range(4): # loop through first, second, third, and fourth planet
        P_i_cdf_evals_4_all[j][i] = cdf_empirical(P_per_sys_4[:,j], P_cdf_evals) # j-th planet

P_i_cdf_evals_1_qtls = np.quantile(P_i_cdf_evals_1_all, qtls, axis=0)
P_i_cdf_evals_2_qtls = [np.quantile(P_i_cdf_evals_2_all[i], qtls, axis=0) for i in range(2)]
P_i_cdf_evals_3_qtls = [np.quantile(P_i_cdf_evals_3_all[i], qtls, axis=0) for i in range(3)]
P_i_cdf_evals_4_qtls = [np.quantile(P_i_cdf_evals_4_all[i], qtls, axis=0) for i in range(4)]

# Systems with 1 observed planet:
plot_fig_cdf_simple(fig_size, [], [ssk_per_sys['P_obs'][ssk_per_sys['Mtot_obs'] == 1, 0]], x_min=P_min, x_max=P_max, log_x=True, lw=lw, c_Kep=colors_ith_planet, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(P_cdf_evals, P_i_cdf_evals_1_qtls[0], P_i_cdf_evals_1_qtls[2], alpha=alpha, color=colors_ith_planet[0], label=r'SysSim (central 68%)')
plt.plot(P_cdf_evals, P_i_cdf_evals_1_qtls[1], lw=lw, color=colors_ith_planet[0], label=r'SysSim (median)')
#plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N = 1$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_N1_CDFs.pdf')
    plt.close()

# Systems with 2 observed planets:
plot_fig_cdf_simple(fig_size, [], [ssk_per_sys['P_obs'][ssk_per_sys['Mtot_obs'] == 2, i] for i in range(2)], x_min=P_min, x_max=P_max, log_x=True, lw=lw, c_Kep=colors_ith_planet, ls_Kep=['--']*2, labels_Kep=['Kepler $P_1$','Kepler $P_2$'], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,P_i_qtls in enumerate(P_i_cdf_evals_2_qtls):
    plt.fill_between(P_cdf_evals, P_i_qtls[0], P_i_qtls[2], alpha=alpha, color=colors_ith_planet[i], label=r'SysSim (central 68%)')
    plt.plot(P_cdf_evals, P_i_qtls[1], lw=lw, color=colors_ith_planet[i], label=r'$P_%s$' % (i+1)) #label=r'SysSim (median)'
#plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N = 2$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_positions_N2_CDFs.pdf')
    plt.close()

# Systems with 3 observed planets:
plot_fig_cdf_simple(fig_size, [], [ssk_per_sys['P_obs'][ssk_per_sys['Mtot_obs'] == 3, i] for i in range(3)], x_min=P_min, x_max=P_max, log_x=True, lw=lw, c_Kep=colors_ith_planet, ls_Kep=['--']*3, labels_Kep=['Kepler $P_1$','Kepler $P_2$','Kepler $P_3$'], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,P_i_qtls in enumerate(P_i_cdf_evals_3_qtls):
    plt.fill_between(P_cdf_evals, P_i_qtls[0], P_i_qtls[2], alpha=alpha, color=colors_ith_planet[i], label=r'SysSim (central 68%)')
    plt.plot(P_cdf_evals, P_i_qtls[1], lw=lw, color=colors_ith_planet[i], label=r'$P_%s$' % (i+1)) #label=r'SysSim (median)'
#plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
plt.title(r'$N = 3$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_positions_N3_CDFs.pdf')
    plt.close()

# Systems with 4 observed planets:
ax = plot_fig_cdf_simple(fig_size, [], [ssk_per_sys['P_obs'][ssk_per_sys['Mtot_obs'] == 4, i] for i in range(4)], x_min=P_min, x_max=P_max, log_x=True, lw=lw, c_Kep=colors_ith_planet, ls_Kep=['--']*4, labels_Kep=['Kepler $P_1$','Kepler $P_2$','Kepler $P_3$','Kepler $P_4$'], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,P_i_qtls in enumerate(P_i_cdf_evals_4_qtls):
    plt.fill_between(P_cdf_evals, P_i_qtls[0], P_i_qtls[2], alpha=alpha, color=colors_ith_planet[i], label=r'SysSim (central 68%)')
    plt.plot(P_cdf_evals, P_i_qtls[1], lw=lw, color=colors_ith_planet[i], label=r'$P_%s$' % (i+1)) #label=r'SysSim (median)'
handles, labels = ax.get_legend_handles_labels()
legend_colors = plt.legend(handles[4:8], labels[4:8], loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs) # legend for different planet positions (colors)
plt.legend(handles[::4], ['Kepler', 'SysSim (median)', 'SysSim (central 68%)'], loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs) # legend for different sets (line styles)
plt.gca().add_artist(legend_colors)
plt.title(r'$N = 4$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_positions_N4_CDFs.pdf')
    plt.close()

plt.show()
