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
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/PR_grids/'
run_number = ''
model_name = 'Maximum_AMD_Model' + run_number #'Non_Clustered_Model', 'Clustered_P_Model', 'Clustered_P_R_Model'

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





#'''
##### To plot the simulated and Kepler catalogs as marginal distributions:

subdirectory = '' #'Paper_Figures/Models/Observed/Clustered_P_R/' #'Paper_Figures/'; 'Talk_Figures/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
n_bins_sys = 50 # fewer bins for system level metrics
lw = 3 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
#loadfiles_directory = '../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
runs = 100

Mtot_bins = np.arange(10)-0.5
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = []

P_bins = np.logspace(np.log10(P_min), np.log10(P_max), n_bins+1)
P_bins_mid = (P_bins[:-1] + P_bins[1:])/2.
P_counts_all = []

Rm_bins = np.logspace(np.log10(1.), np.log10(30.), n_bins+1)
Rm_bins_mid = (Rm_bins[:-1] + Rm_bins[1:])/2.
Rm_counts_all = []

tdur_bins = np.linspace(0., 15., n_bins+1)
tdur_bins_mid = (tdur_bins[:-1] + tdur_bins[1:])/2.
tdur_counts_all = []

tdur_tcirc_bins = np.linspace(0., 1.5, n_bins+1)
tdur_tcirc_bins_mid = (tdur_tcirc_bins[:-1] + tdur_tcirc_bins[1:])/2.
tdur_tcirc_1_counts_all = []
tdur_tcirc_2p_counts_all = []

D_bins = np.logspace(-5., -1.5, n_bins+1)
D_bins_mid = (D_bins[:-1] + D_bins[1:])/2.
D_counts_all = []

radii_bins = np.linspace(radii_min, radii_max, n_bins+1)
radii_bins_mid = (radii_bins[:-1] + radii_bins[1:])/2.
radii_counts_all = []

Rstar_bins = np.linspace(0.5, 2.5, n_bins+1)
Rstar_bins_mid = (Rstar_bins[:-1] + Rstar_bins[1:])/2.
Rstar_counts_all = []

D_ratio_bins = np.logspace(-1.5, 1.5, n_bins+1)
D_ratio_bins_mid = (D_ratio_bins[:-1] + D_ratio_bins[1:])/2.
D_ratio_counts_all = []

xi_bins = np.linspace(-0.5, 0.5, n_bins+1)
xi_bins_mid = (xi_bins[:-1] + xi_bins[1:])/2.
xi_counts_all = []
xi_res_counts_all = []
xi_nonres_counts_all = []
xi_2_counts_all = []
xi_3_counts_all = []
xi_4p_counts_all = []

# Extra stats:
radii_partitioning_bins = np.logspace(-5., 0., n_bins_sys+1)
radii_partitioning_bins_mid = (radii_partitioning_bins[:-1] + radii_partitioning_bins[1:])/2.
radii_partitioning_counts_all = []

radii_monotonicity_bins = np.linspace(-0.5, 0.6, n_bins_sys+1)
radii_monotonicity_bins_mid = (radii_monotonicity_bins[:-1] + radii_monotonicity_bins[1:])/2.
radii_monotonicity_counts_all = []

gap_complexity_bins = np.linspace(0., 1., n_bins_sys+1)
gap_complexity_bins_mid = (gap_complexity_bins[:-1] + gap_complexity_bins[1:])/2.
gap_complexity_counts_all = []

f_pos_monotonicity = [] # fractions of systems with observed positive monotonicity
f_low_gap_complexity = [] # fractions of systems with observed gap complexity < 0.1

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod)

    # Multiplicities:
    counts, bins = np.histogram(sss_per_sys_i['Mtot_obs'], bins=Mtot_bins)
    Mtot_counts_all.append(counts/float(np.sum(counts)))

    # Periods:
    counts, bins = np.histogram(sss_i['P_obs'], bins=P_bins)
    P_counts_all.append(counts/float(np.sum(counts)))

    # Period ratios:
    counts, bins = np.histogram(sss_i['Rm_obs'], bins=Rm_bins)
    Rm_counts_all.append(counts/float(np.sum(counts)))

    # Durations:
    counts, bins = np.histogram(sss_i['tdur_obs'], bins=tdur_bins)
    tdur_counts_all.append(counts/float(np.sum(counts)))

    # Circular normalized durations (singles and multis):
    counts, bins = np.histogram(sss_i['tdur_tcirc_1_obs'], bins=tdur_tcirc_bins)
    tdur_tcirc_1_counts_all.append(counts/float(np.sum(counts)))
    counts, bins = np.histogram(sss_i['tdur_tcirc_2p_obs'], bins=tdur_tcirc_bins)
    tdur_tcirc_2p_counts_all.append(counts/float(np.sum(counts)))

    # Depths:
    counts, bins = np.histogram(sss_i['D_obs'], bins=D_bins)
    D_counts_all.append(counts/float(np.sum(counts)))

    # Planet radii:
    counts, bins = np.histogram(sss_i['radii_obs'], bins=radii_bins)
    radii_counts_all.append(counts/float(np.sum(counts)))

    # Stellar radii:
    counts, bins = np.histogram(sss_i['Rstar_obs'], bins=Rstar_bins)
    Rstar_counts_all.append(counts/float(np.sum(counts)))

    # Depth ratios:
    counts, bins = np.histogram(sss_i['D_ratio_obs'], bins=D_ratio_bins)
    D_ratio_counts_all.append(counts/float(np.sum(counts)))

    # Log(xi):
    counts, bins = np.histogram(np.log10(sss_i['xi_obs']), bins=xi_bins)
    xi_counts_all.append(counts/float(np.sum(counts)))

    # Log(xi) (res):
    counts, bins = np.histogram(np.log10(sss_i['xi_res_obs']), bins=xi_bins)
    xi_res_counts_all.append(counts/float(np.sum(counts)))

    # Log(xi) (non-res):
    counts, bins = np.histogram(np.log10(sss_i['xi_nonres_obs']), bins=xi_bins)
    xi_nonres_counts_all.append(counts/float(np.sum(counts)))

    # Log(xi) by multiplicity (2,3,4+):
    logxi_2 = np.log10(sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] == 2, 0])
    logxi_3 = np.log10(sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] == 3, :2].flatten())
    xi_4p = sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] >= 4]
    logxi_4p = np.log10(xi_4p[xi_4p != -1])

    counts, bins = np.histogram(logxi_2, bins=xi_bins)
    xi_2_counts_all.append(counts/float(np.sum(counts)))
    counts, bins = np.histogram(logxi_3, bins=xi_bins)
    xi_3_counts_all.append(counts/float(np.sum(counts)))
    counts, bins = np.histogram(logxi_4p, bins=xi_bins)
    xi_4p_counts_all.append(counts/float(np.sum(counts)))

    # Radii partitioning:
    counts, bins = np.histogram(sss_per_sys_i['radii_partitioning'], bins=radii_partitioning_bins)
    radii_partitioning_counts_all.append(counts/float(np.sum(counts)))

    # Radii monotonicity:
    counts, bins = np.histogram(sss_per_sys_i['radii_monotonicity'], bins=radii_monotonicity_bins)
    radii_monotonicity_counts_all.append(counts/float(np.sum(counts)))

    # Gap complexity:
    counts, bins = np.histogram(sss_per_sys_i['gap_complexity'], bins=gap_complexity_bins)
    gap_complexity_counts_all.append(counts/float(np.sum(counts)))

    # Extra stats:
    f_pos_M = np.sum(sss_per_sys_i['radii_monotonicity'] > 0) / float(len(sss_per_sys_i['radii_monotonicity']))
    f_low_C = np.sum(sss_per_sys_i['gap_complexity'] < 0.1) / float(len(sss_per_sys_i['gap_complexity']))
    f_pos_monotonicity.append(f_pos_M)
    f_low_gap_complexity.append(f_low_C)

    # To plot the fraction of planets in observed multis on a period-radius diagram:
    #P_bins = np.logspace(np.log10(P_min), np.log10(P_max), 6+1)
    #R_bins = np.logspace(np.log10(radii_min), np.log10(radii_max), 6+1)
    #plot_fig_period_radius_fraction_multis(sss_per_sys_i, sss_i, P_bins, R_bins, save_name=savefigures_directory + subdirectory + model_name + '_%s_PR_grid_fraction_obs_multis.png' % run_number, save_fig=False)
    #plt.show()

Mtot_counts_all = np.array(Mtot_counts_all)
P_counts_all = np.array(P_counts_all)
Rm_counts_all = np.array(Rm_counts_all)
tdur_counts_all = np.array(tdur_counts_all)
tdur_tcirc_1_counts_all = np.array(tdur_tcirc_1_counts_all)
tdur_tcirc_2p_counts_all = np.array(tdur_tcirc_2p_counts_all)
D_counts_all = np.array(D_counts_all)
radii_counts_all = np.array(radii_counts_all)
Rstar_counts_all = np.array(Rstar_counts_all)
D_ratio_counts_all = np.array(D_ratio_counts_all)
xi_counts_all = np.array(xi_counts_all)
xi_res_counts_all = np.array(xi_res_counts_all)
xi_nonres_counts_all = np.array(xi_nonres_counts_all)
xi_2_counts_all = np.array(xi_2_counts_all)
xi_3_counts_all = np.array(xi_3_counts_all)
xi_4p_counts_all = np.array(xi_4p_counts_all)
radii_partitioning_counts_all = np.array(radii_partitioning_counts_all)
radii_monotonicity_counts_all = np.array(radii_monotonicity_counts_all)
gap_complexity_counts_all = np.array(gap_complexity_counts_all)

f_pos_monotonicity = np.array(f_pos_monotonicity)
f_low_gap_complexity = np.array(f_low_gap_complexity)



Mtot_counts_qtls = np.zeros((len(Mtot_bins_mid),3))
for b in range(len(Mtot_bins_mid)):
    Mtot_counts_qtls[b] = np.quantile(Mtot_counts_all[:,b], [0.16, 0.5, 0.84])

P_counts_qtls = np.zeros((n_bins,3))
Rm_counts_qtls = np.zeros((n_bins,3))
tdur_counts_qtls = np.zeros((n_bins,3))
tdur_tcirc_1_counts_qtls = np.zeros((n_bins,3))
tdur_tcirc_2p_counts_qtls = np.zeros((n_bins,3))
D_counts_qtls = np.zeros((n_bins,3))
radii_counts_qtls = np.zeros((n_bins,3))
Rstar_counts_qtls = np.zeros((n_bins,3))
D_ratio_counts_qtls = np.zeros((n_bins,3))
xi_counts_qtls = np.zeros((n_bins,3))
xi_res_counts_qtls = np.zeros((n_bins,3))
xi_nonres_counts_qtls = np.zeros((n_bins,3))
xi_2_counts_qtls = np.zeros((n_bins,3))
xi_3_counts_qtls = np.zeros((n_bins,3))
xi_4p_counts_qtls = np.zeros((n_bins,3))
radii_partitioning_counts_qtls = np.zeros((n_bins_sys,3))
radii_monotonicity_counts_qtls = np.zeros((n_bins_sys,3))
gap_complexity_counts_qtls = np.zeros((n_bins_sys,3))
for b in range(n_bins):
    # Periods:
    P_counts_qtls[b] = np.quantile(P_counts_all[:,b], [0.16, 0.5, 0.84])

    # Period ratios:
    Rm_counts_qtls[b] = np.quantile(Rm_counts_all[:,b], [0.16, 0.5, 0.84])

    # Durations:
    tdur_counts_qtls[b] = np.quantile(tdur_counts_all[:,b], [0.16, 0.5, 0.84])

    # Circular normalized durations (singles and multis):
    tdur_tcirc_1_counts_qtls[b] = np.quantile(tdur_tcirc_1_counts_all[:,b], [0.16, 0.5, 0.84])
    tdur_tcirc_2p_counts_qtls[b] = np.quantile(tdur_tcirc_2p_counts_all[:,b], [0.16, 0.5, 0.84])

    # Depths:
    D_counts_qtls[b] = np.quantile(D_counts_all[:,b], [0.16, 0.5, 0.84])

    # Planet radii:
    radii_counts_qtls[b] = np.quantile(radii_counts_all[:,b], [0.16, 0.5, 0.84])

    # Stellar radii:
    Rstar_counts_qtls[b] = np.quantile(Rstar_counts_all[:,b], [0.16, 0.5, 0.84])

    # Depth ratios:
    D_ratio_counts_qtls[b] = np.quantile(D_ratio_counts_all[:,b], [0.16, 0.5, 0.84])

    # Log(xi):
    xi_counts_qtls[b] = np.quantile(xi_counts_all[:,b], [0.16, 0.5, 0.84])
    xi_res_counts_qtls[b] = np.quantile(xi_res_counts_all[:,b], [0.16, 0.5, 0.84])
    xi_nonres_counts_qtls[b] = np.quantile(xi_nonres_counts_all[:,b], [0.16, 0.5, 0.84])
    xi_2_counts_qtls[b] = np.quantile(xi_2_counts_all[:,b], [0.16, 0.5, 0.84])
    xi_3_counts_qtls[b] = np.quantile(xi_3_counts_all[:,b], [0.16, 0.5, 0.84])
    xi_4p_counts_qtls[b] = np.quantile(xi_4p_counts_all[:,b], [0.16, 0.5, 0.84])

for b in range(n_bins_sys):
    # Radii partitioning:
    radii_partitioning_counts_qtls[b] = np.quantile(radii_partitioning_counts_all[:,b], [0.16, 0.5, 0.84])

    # Radii monotonicity:
    radii_monotonicity_counts_qtls[b] = np.quantile(radii_monotonicity_counts_all[:,b], [0.16, 0.5, 0.84])

    # Gap complexity:
    gap_complexity_counts_qtls[b] = np.quantile(gap_complexity_counts_all[:,b], [0.16, 0.5, 0.84])
#####





# To make a 'plot' listing the model parameters:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,1,left=fig_lbrt[0],bottom=fig_lbrt[1],right=fig_lbrt[2],top=fig_lbrt[3],wspace=0.1,hspace=0.1)
nrows = 8
for i,param in enumerate(param_vals_all):
    plt.figtext(x=0.05+0.3*int(i/float(nrows)), y=0.875-0.1*(i%nrows), s=r'%s = %s' % (param_symbols[param], np.round(param_vals_all[param],3)), fontsize=lfs)
if savefigures == True:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_sim_params.pdf')
    plt.close()

# Multiplicities:
plot_fig_counts_hist_simple(fig_size, [sss_per_sys['Mtot_obs']], [ssk_per_sys['Mtot_obs']], x_min=0, x_llim=0.5, normalize=True, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, lw=lw, xlabel_text='Observed planets per system', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(Mtot_bins_mid, Mtot_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label=r'16% and 84%')
plt.plot(Mtot_bins_mid, Mtot_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--')
plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_multiplicities_compare.pdf')
    plt.close()

# Periods:
plot_fig_pdf_simple(fig_size, [sss['P_obs']], [ssk['P_obs']], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.plot(P_bins_mid, P_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(P_bins_mid, P_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_compare.pdf')
    plt.close()

# Period ratios (all, with some upper cut-off):
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple(fig_size, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut]], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(Rm_bins_mid, Rm_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(Rm_bins_mid, Rm_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periodratios_compare.pdf')
    plt.close()

# Transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_obs']], [ssk['tdur_obs']], x_min=0., x_max=15., n_bins=n_bins, lw=lw, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(tdur_bins_mid, tdur_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(tdur_bins_mid, tdur_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_durations_compare.pdf')
    plt.close()

# Circular normalized transit durations (separate singles and multis):
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_1_obs']], [ssk['tdur_tcirc_1_obs']], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, lw=lw, extra_text='Observed singles', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_compare.pdf')
    plt.close()

plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_2p_obs']], [ssk['tdur_tcirc_2p_obs']], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, lw=lw, extra_text='Observed multis', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_compare.pdf')
    plt.close()

# Transit depths:
plot_fig_pdf_simple(fig_size, [sss['D_obs']], [ssk['D_obs']], x_min=np.min(D_bins), x_max=np.max(D_bins), log_x=True, lw=lw, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(D_bins_mid, D_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(D_bins_mid, D_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depths_compare.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_simple(fig_size, [sss['radii_obs']], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, n_bins=n_bins, lw=lw, xlabel_text=r'$R_p (R_\oplus)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(radii_bins_mid, radii_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(radii_bins_mid, radii_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_compare.pdf')
    plt.close()

# Stellar radii:
plot_fig_pdf_simple(fig_size, [sss['Rstar_obs']], [ssk['Rstar_obs']], x_min=0.5, x_max=2.5, n_bins=n_bins, lw=lw, xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(Rstar_bins_mid, Rstar_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(Rstar_bins_mid, Rstar_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_stellar_radii_compare.pdf')
    plt.close()

# Transit depth ratios:
plot_fig_pdf_simple(fig_size, [sss['D_ratio_obs']], [ssk['D_ratio_obs']], x_min=np.min(D_ratio_bins), x_max=np.max(D_ratio_bins), n_bins=n_bins, log_x=True, lw=lw, xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(D_ratio_bins_mid, D_ratio_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(D_ratio_bins_mid, D_ratio_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depthratios_compare.pdf')
    plt.close()

# Log(xi):
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_obs'])], [np.log10(ssk['xi_obs'])], x_min=np.min(xi_bins), x_max=np.max(xi_bins), n_bins=n_bins, lw=lw, xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.plot(xi_bins_mid, xi_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
plt.plot(xi_bins_mid, xi_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_all_compare.pdf')
    plt.close()

# Log(xi) by res/non-res:
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_res_obs']), np.log10(sss['xi_nonres_obs'])], [np.log10(ssk['xi_res_obs']), np.log10(ssk['xi_nonres_obs'])], x_min=np.min(xi_bins), x_max=np.max(xi_bins), n_bins=n_bins, c_sim=['m','g'], c_Kep=['m','g'], ls_sim=['-','-'], ls_Kep=['-','-'], lw=lw, labels_sim=['Near MMR', 'Not near MMR'], labels_Kep=[None, None], xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.plot(xi_bins_mid, xi_res_counts_qtls[:,0], drawstyle='steps-mid', color='m', lw=1, ls='--', label='16')
plt.plot(xi_bins_mid, xi_res_counts_qtls[:,2], drawstyle='steps-mid', color='m', lw=1, ls='--', label='84')
plt.plot(xi_bins_mid, xi_nonres_counts_qtls[:,0], drawstyle='steps-mid', color='g', lw=1, ls='--', label='16')
plt.plot(xi_bins_mid, xi_nonres_counts_qtls[:,2], drawstyle='steps-mid', color='g', lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_compare.pdf')
    plt.close()

plt.show()
#plt.close()





##### To plot the xi distribution separated by observed multiplicities (m=2,3,4+):

logxi_Kep_2 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 2, 0])
logxi_Kep_3 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 3, :2].flatten())
xi_Kep_4p = ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] >= 4]
logxi_Kep_4p = np.log10(xi_Kep_4p[xi_Kep_4p != -1])

logxi_2 = np.log10(sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] == 2, 0])
logxi_3 = np.log10(sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] == 3, :2].flatten())
xi_4p = sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] >= 4]
logxi_4p = np.log10(xi_4p[xi_4p != -1])

c2, c3, c4p = 'r', 'b', 'g'

fig = plt.figure(figsize=(8,12))
plot = GridSpec(5,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0])
plot_panel_cdf_simple(ax, [logxi_2, logxi_3, logxi_4p], [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], x_min=np.min(xi_bins), x_max=np.max(xi_bins), c_sim=[c2,c3,c4p], c_Kep=['r','b','g'], ls_sim=['-','-','-'], ls_Kep=[':',':',':'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=['Kepler data',None,None], xlabel_text='', legend=True, afs=afs, tfs=tfs, lfs=lfs, label_dist=False)

ax = plt.subplot(plot[1:3,0])
#plot_panel_pdf_simple(ax, [logxi_2, logxi_3, logxi_4p], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=0.13, n_bins=n_bins, c_sim=[c2,c3,c4p], ls_sim=['-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None], xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plot_panel_pdf_simple(ax, [], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=0.13, n_bins=n_bins, xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.plot(xi_bins_mid, xi_2_counts_qtls[:,1], drawstyle='steps-mid', color=c2, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_3_counts_qtls[:,1], drawstyle='steps-mid', color=c3, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_4p_counts_qtls[:,1], drawstyle='steps-mid', color=c4p, lw=2, ls='-')
plt.text(x=0.98, y=0.9, s='Maximum AMD model', ha='right', fontsize=lfs, transform=ax.transAxes) #s='Maximum AMD model'; s='Two-Rayleigh model'
plt.fill_between(xi_bins_mid, xi_2_counts_qtls[:,0], xi_2_counts_qtls[:,2], step='mid', color=c2, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_3_counts_qtls[:,0], xi_3_counts_qtls[:,2], step='mid', color=c3, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_4p_counts_qtls[:,0], xi_4p_counts_qtls[:,2], step='mid', color=c4p, alpha=alpha)

ax = plt.subplot(plot[3:,0])
plot_panel_pdf_simple(ax, [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=0.13, n_bins=n_bins, c_sim=[c2,c3,c4p], ls_sim=['-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None], xlabel_text=r'$\log{\xi}$', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s='Kepler data', ha='right', fontsize=lfs, transform=ax.transAxes)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_per_mult.pdf')
    plt.close()





##### To plot the proximity to MMR parameter (Pratio/Pratio_MMR - 1):
'''
fig = plt.figure(figsize=(8,10))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)

pr_mmrs = [(j+1)/j for j in range(1,5)]
pr_mmrs_labels = ['2:1', '3:2', '4:3', '5:4']

ax = plt.subplot(plot[:,:])
plot_panel_pdf_simple(ax, [ssk['Rm_obs']/pr - 1. for pr in pr_mmrs], [], x_min=-0.1, x_max=0.1, n_bins=20, c_sim=['r','b','g','m'], ls_sim=['-','-','-','-'], lw=2, labels_sim=pr_mmrs_labels, labels_Kep=[None], xlabel_text=r'$\mathcal{P}/\mathcal{P}_{\rm mmr} - 1$', legend=False, afs=afs, tfs=tfs, lfs=lfs)
'''





##### To plot period ratio distribution again but with MMRs marked:

R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple(fig_size, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut]], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20], labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'Period ratio $P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
#plt.plot(Rm_bins_mid, Rm_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
#plt.plot(Rm_bins_mid, Rm_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
for pr in [1.5, 2.]:
    plt.fill_betweenx([0,1], x1=pr, x2=(1.+res_width)*pr, alpha=0.2, color='r')





##### To remake some marginal distributions again, but with central 68% as shaded regions and the Kepler data as solid histograms, for proposals:

directory = '/Users/hematthi/Documents/GradSchool/Postdoctoral_Applications/Figures/'

fig_size = (6,3) #size of each panel (figure)
fig_lbrt = [0.2, 0.3, 0.95, 0.925]

n_bins = 100
lw = 2 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

# Periods:
plot_fig_pdf_simple(fig_size, [ssk['P_obs']], [], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.fill_between(P_bins_mid, P_counts_qtls[:,0], P_counts_qtls[:,2], step='mid', color='k', alpha=alpha, label=r'Simulated 16-84%')
if savefigures:
    plt.savefig(directory + model_name + '_periods_compare.pdf')
    plt.close()

# Period ratios (all, with some upper cut-off):
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple(fig_size, [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], [], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(Rm_bins_mid, Rm_counts_qtls[:,0], Rm_counts_qtls[:,2], step='mid', color='k', alpha=alpha)
if savefigures:
    plt.savefig(directory + model_name + '_periodratios_compare.pdf')
    plt.close()

# Transit durations:
plot_fig_pdf_simple(fig_size, [ssk['tdur_obs']], [], x_min=0., x_max=15., n_bins=n_bins, lw=lw, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(tdur_bins_mid, tdur_counts_qtls[:,0], tdur_counts_qtls[:,2], step='mid', color='k', alpha=alpha)
if savefigures:
    plt.savefig(directory + model_name + '_durations_compare.pdf')
    plt.close()

# Transit depths:
plot_fig_pdf_simple(fig_size, [ssk['D_obs']], [], x_min=np.min(D_bins), x_max=np.max(D_bins), log_x=True, lw=lw, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(D_bins_mid, D_counts_qtls[:,0], D_counts_qtls[:,2], step='mid', color='k', alpha=alpha)
if savefigures:
    plt.savefig(directory + model_name + '_depths_compare.pdf')
    plt.close()

# Transit depth ratios:
plot_fig_pdf_simple(fig_size, [ssk['D_ratio_obs']], [], x_min=np.min(D_ratio_bins), x_max=np.max(D_ratio_bins), n_bins=n_bins, log_x=True, lw=lw, xlabel_text=r'Transit depth ratios, $\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(D_ratio_bins_mid, D_ratio_counts_qtls[:,0], D_ratio_counts_qtls[:,2], step='mid', color='k', alpha=alpha)
if savefigures:
    plt.savefig(directory + model_name + '_depthratios_compare.pdf')
    plt.close()

# Log(xi):
plot_fig_pdf_simple(fig_size, [np.log10(ssk['xi_obs'])], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), n_bins=n_bins, lw=lw, xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(xi_bins_mid, xi_counts_qtls[:,0], xi_counts_qtls[:,2], step='mid', color='k', alpha=alpha)
if savefigures:
    plt.savefig(directory + model_name + '_logxi_all_compare.pdf')
    plt.close()

# Radius partitioning:
plot_fig_pdf_simple(fig_size, [ssk_per_sys['radii_partitioning']], [], x_min=np.min(radii_partitioning_bins), x_max=np.max(radii_partitioning_bins), n_bins=n_bins_sys, log_x=True, lw=lw, xlabel_text=r'Radius partitioning, $\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(radii_partitioning_bins_mid, radii_partitioning_counts_qtls[:,0], radii_partitioning_counts_qtls[:,2], step='mid', color='k', alpha=alpha)
if savefigures:
    plt.savefig(directory + model_name + '_radii_partitioning_compare.pdf')
    plt.close()

# Radius monotonicity:
plot_fig_pdf_simple(fig_size, [ssk_per_sys['radii_monotonicity']], [], x_min=np.min(radii_monotonicity_bins), x_max=np.max(radii_monotonicity_bins), n_bins=n_bins_sys, log_x=False, lw=lw, xlabel_text=r'Radius monotonicity, $\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(radii_monotonicity_bins_mid, radii_monotonicity_counts_qtls[:,0], radii_monotonicity_counts_qtls[:,2], step='mid', color='k', alpha=alpha)
if savefigures:
    plt.savefig(directory + model_name + '_radii_monotonicity_compare.pdf')
    plt.close()

# Gap complexity:
plot_fig_pdf_simple(fig_size, [ssk_per_sys['gap_complexity']], [], x_min=np.min(gap_complexity_bins), x_max=np.max(gap_complexity_bins), n_bins=n_bins_sys, log_x=False, lw=lw, xlabel_text=r'Gap complexity, $\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(gap_complexity_bins_mid, gap_complexity_counts_qtls[:,0], gap_complexity_counts_qtls[:,2], step='mid', color='k', alpha=alpha)
if savefigures:
    plt.savefig(directory + model_name + '_gap_complexity_compare.pdf')
    plt.close()

plt.show()





##### To make a collection of some marginal distributions for a GIF:
'''
directory = '/Users/hematthi/Documents/GradSchool/Conferences/Exoplanet_Demographics_2020/Figures/GIF_images/Marginals_observed/'

n_bins = 100
lw = 3 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

for i in range(runs):
    fig = plt.figure(figsize=(8,15))
    plot = GridSpec(5,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0.4)

    ax = plt.subplot(plot[0,0]) # multiplicities
    plot_panel_counts_hist_simple(ax, [], [ssk_per_sys['Mtot_obs']], x_min=0, x_llim=0.5, normalize=True, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, lw=lw, xlabel_text='Observed planets per system', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(Mtot_bins_mid, Mtot_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-', label='Simulated')
    plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), ncol=1, frameon=False, fontsize=lfs) #show the legend

    ax = plt.subplot(plot[1,0]) # periods
    plot_panel_pdf_simple(ax, [], [ssk['P_obs']], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(P_bins_mid, P_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-')

    ax = plt.subplot(plot[2,0]) # period ratios
    R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
    plot_panel_pdf_simple(ax, [], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(Rm_bins_mid, Rm_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-')

    ax = plt.subplot(plot[3,0]) # depths
    plot_panel_pdf_simple(ax, [], [ssk['D_obs']], x_min=np.min(D_bins), x_max=np.max(D_bins), log_x=True, lw=lw, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(D_bins_mid, D_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-')

    ax = plt.subplot(plot[4,0]) # depth ratios
    plot_panel_pdf_simple(ax, [], [ssk['D_ratio_obs']], x_min=np.min(D_ratio_bins), x_max=np.max(D_ratio_bins), n_bins=n_bins, log_x=True, lw=lw, xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(D_ratio_bins_mid, D_ratio_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-')

    if savefigures:
        plt.savefig(directory + model_name + '_some_marginals_compare_%s.png' % i)
        plt.close()
    plt.show()
'''
