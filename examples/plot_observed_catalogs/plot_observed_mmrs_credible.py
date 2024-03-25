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
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/H20_model/Observed/Archit-III_notation/' #Archit-III_notation/
run_number = ''
model_name = 'Maximum_AMD_Model' + run_number

compute_ratios = compute_ratios_adjacent
AD_mod = True
weights_all = load_split_stars_weights_only()
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 'period_ratios_KS',
                 #'durations_KS',
                 #'durations_norm_circ_KS',
                 'durations_norm_circ_singles_KS',
                 'durations_norm_circ_multis_KS',
                 'duration_ratios_nonmmr_KS',
                 'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radius_ratios_KS',
                 'radii_partitioning_KS',
                 'radii_monotonicity_KS',
                 'gap_complexity_KS',
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





##### To plot the simulated and Kepler catalogs:

fig_size = (8,5) #size of each panel (figure)
fig_lbrt = [0.15, 0.2, 0.95, 0.925]

n_bins = 40
lw = 2 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





##### To load a large number of models:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models_1000/'
#loadfiles_directory = '../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
runs = 1000

qtls_123sigma_lower_upper = [0.001,0.022,0.159,0.5,0.841,0.978,0.999]

sss_per_sys_all = []
sss_all = []

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod)

    sss_per_sys_all.append(sss_per_sys_i)
    sss_all.append(sss_i)
#####





##### To plot histograms of the zeta statistics and numbers of planets near MMRs:

n_mmr = 2 # 1 => only consider the 1st order MMRs; 2 => consider both 1st and 2nd order MMRs
pratio_max = 3. if n_mmr==1 else 4.
#pratio_max_1, pratio_max_2 = 2.5, 4. # should be 2.5 for zeta_{2,1} and 4 for zeta_{2,2}

zeta_in_mmr_lim_list = [0.25, 0.2] # all |zeta1| <= zeta_in_mmr_lim are considered "near a 1st order MMR"

# First, compute the statistics for the Kepler catalog:
pratios_small_Kep = ssk['Rm_obs'][ssk['Rm_obs'] < pratio_max]
bools_in_1st_Kep, _ = pratio_is_in_any_1st_order_mmr_neighborhood(pratios_small_Kep)
bools_in_2nd_Kep, _ = pratio_is_in_any_2nd_order_mmr_neighborhood(pratios_small_Kep)
zeta1_small_Kep = zeta(pratios_small_Kep[bools_in_1st_Kep]) # zeta_{2,1}
zeta2_small_Kep = zeta(pratios_small_Kep[bools_in_2nd_Kep], order=2) # zeta_{2,2}

zeta1_counts_near_MMRs_Kep_dict = {zeta_lim : np.sum(np.abs(zeta1_small_Kep) <= zeta_lim)/len(zeta1_small_Kep) for zeta_lim in zeta_in_mmr_lim_list}
zeta1_counts_near_above_MMRs_Kep_dict = {zeta_lim : np.sum((zeta1_small_Kep >= -zeta_lim) & (zeta1_small_Kep <= 0.))/len(zeta1_small_Kep) for zeta_lim in zeta_in_mmr_lim_list}
zeta1_counts_near_below_MMRs_Kep_dict = {zeta_lim : np.sum((zeta1_small_Kep >= 0.) & (zeta1_small_Kep <= zeta_lim))/len(zeta1_small_Kep) for zeta_lim in zeta_in_mmr_lim_list}

# Then, compute the statistics for each of the simulated catalogs:
zeta_bins = np.linspace(-1., 1., n_bins+1)
zeta_bins_mid = (zeta_bins[:-1] + zeta_bins[1:])/2.

zeta1_counts_all = [] # to be filled with normalized (fractional) counts per bin for each simulated catalog
zeta2_counts_all = []
zeta1_counts_near_MMRs_all_dict = {zeta_lim : [] for zeta_lim in zeta_in_mmr_lim_list} # to be filled with normalized (fractional) counts of the number of planet-pairs near MMRs (|zeta| < 'zeta_in_mmr_lim')
zeta1_counts_near_above_MMRs_all_dict = {zeta_lim : [] for zeta_lim in zeta_in_mmr_lim_list} # -0.25 < zeta < 0
zeta1_counts_near_below_MMRs_all_dict = {zeta_lim : [] for zeta_lim in zeta_in_mmr_lim_list} # 0 < zeta < 0.25
for i,sss_i in enumerate(sss_all):
    pratios_small_sim_i = sss_i['Rm_obs'][sss_i['Rm_obs'] < pratio_max]
    bools_in_1st_sim_i, _ = pratio_is_in_any_1st_order_mmr_neighborhood(pratios_small_sim_i)
    bools_in_2nd_sim_i, _ = pratio_is_in_any_2nd_order_mmr_neighborhood(pratios_small_sim_i)
    zeta1_small_sim_i = zeta(pratios_small_sim_i[bools_in_1st_sim_i])
    zeta2_small_sim_i = zeta(pratios_small_sim_i[bools_in_2nd_sim_i], order=2)
    
    # Histograms of zetas:
    counts_zeta1, bins = np.histogram(zeta1_small_sim_i, bins=zeta_bins)
    counts_zeta2, bins = np.histogram(zeta2_small_sim_i, bins=zeta_bins)
    zeta1_counts_all.append(counts_zeta1/np.sum(counts_zeta1))
    zeta2_counts_all.append(counts_zeta2/np.sum(counts_zeta2))
    
    # Numbers of planet pairs near MMRs:
    for zeta_lim in zeta_in_mmr_lim_list:
        counts_zeta1_near_MMRs = np.sum(np.abs(zeta1_small_sim_i) <= zeta_lim)/len(zeta1_small_sim_i)
        counts_zeta1_near_above_MMRs = np.sum((zeta1_small_sim_i >= -zeta_lim) & (zeta1_small_sim_i <= 0.))/len(zeta1_small_sim_i)
        counts_zeta1_near_below_MMRs = np.sum((zeta1_small_sim_i >= 0.) & (zeta1_small_sim_i <= zeta_lim))/len(zeta1_small_sim_i)
        assert np.isclose(counts_zeta1_near_MMRs, counts_zeta1_near_above_MMRs + counts_zeta1_near_below_MMRs)
        zeta1_counts_near_MMRs_all_dict[zeta_lim].append(counts_zeta1_near_MMRs)
        zeta1_counts_near_above_MMRs_all_dict[zeta_lim].append(counts_zeta1_near_above_MMRs)
        zeta1_counts_near_below_MMRs_all_dict[zeta_lim].append(counts_zeta1_near_below_MMRs)
    
zeta1_counts_all = np.array(zeta1_counts_all)
zeta2_counts_all = np.array(zeta2_counts_all)
for zeta_lim in zeta_in_mmr_lim_list:
    zeta1_counts_near_MMRs_all_dict[zeta_lim] = np.array(zeta1_counts_near_MMRs_all_dict[zeta_lim])
    zeta1_counts_near_above_MMRs_all_dict[zeta_lim] = np.array(zeta1_counts_near_above_MMRs_all_dict[zeta_lim])
    zeta1_counts_near_below_MMRs_all_dict[zeta_lim] = np.array(zeta1_counts_near_below_MMRs_all_dict[zeta_lim])

zeta1_counts_qtls = np.quantile(zeta1_counts_all, qtls_123sigma_lower_upper, axis=0) # quantiles per bin in zeta
zeta2_counts_qtls = np.quantile(zeta2_counts_all, qtls_123sigma_lower_upper, axis=0) # quantiles per bin in zeta

zeta1_counts_near_MMRs_qtls_dict = {zeta_lim : np.quantile(zeta1_counts_near_MMRs_all_dict[zeta_lim], [0.16,0.5,0.84]) for zeta_lim in zeta_in_mmr_lim_list} # quantiles in the fraction of planet-pairs near MMRs
zeta1_counts_near_above_MMRs_qtls_dict = {zeta_lim : np.quantile(zeta1_counts_near_above_MMRs_all_dict[zeta_lim], [0.16,0.5,0.84]) for zeta_lim in zeta_in_mmr_lim_list}
zeta1_counts_near_below_MMRs_qtls_dict = {zeta_lim : np.quantile(zeta1_counts_near_below_MMRs_all_dict[zeta_lim], [0.16,0.5,0.84]) for zeta_lim in zeta_in_mmr_lim_list}

# Plot histograms of zeta_{2,1}:
ax = plot_fig_pdf_simple([zeta1_small_Kep], [], x_min=-1., x_max=1., n_bins=n_bins, normalize=True, lw=lw, labels_sim=['Kepler catalog'], xlabel_text=r'$\zeta_{2,1}$', ylabel_text='Normalized fraction', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
ax.tick_params(right=True)
plt.plot(zeta_bins_mid, zeta1_counts_qtls[3], drawstyle='steps-mid', color='b', lw=lw) #, label='SysSim catalogs (median)'
plt.fill_between(zeta_bins_mid, zeta1_counts_qtls[2], zeta1_counts_qtls[-3], step='mid', color='b', alpha=alpha, label='SysSim catalogs \n(16-84%)')
plt.fill_between(zeta_bins_mid, zeta1_counts_qtls[1], zeta1_counts_qtls[-2], step='mid', color='b', alpha=alpha/2., label='')
plt.fill_between(zeta_bins_mid, zeta1_counts_qtls[0], zeta1_counts_qtls[-1], step='mid', color='b', alpha=alpha/4., label='')
zeta_in_mmr_lim = zeta_in_mmr_lim_list[0]
plt.vlines([-zeta_in_mmr_lim, zeta_in_mmr_lim], 0., 1., colors='r', linestyles='dashed')
plt.fill_betweenx([0,1], -zeta_in_mmr_lim, zeta_in_mmr_lim, color='r', alpha=0.1)
plt.text(0., 0.95, 'Near MMRs', color='k', ha='center', fontsize=12, transform=ax.get_xaxis_transform())
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_zeta2_1_small_compare_credible.pdf')
    plt.close()

# Plot histograms of zeta_{2,2}:
ax = plot_fig_pdf_simple([zeta2_small_Kep], [], x_min=-1., x_max=1., y_max=0.12, n_bins=n_bins, normalize=True, lw=lw, labels_sim=['Kepler catalog'], xlabel_text=r'$\zeta_{2,2}$', ylabel_text='Normalized fraction', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
ax.tick_params(right=True)
plt.plot(zeta_bins_mid, zeta2_counts_qtls[3], drawstyle='steps-mid', color='b', lw=lw) #, label='SysSim catalogs (median)'
plt.fill_between(zeta_bins_mid, zeta2_counts_qtls[2], zeta2_counts_qtls[-3], step='mid', color='b', alpha=alpha, label='SysSim catalogs (16-84%)')
plt.fill_between(zeta_bins_mid, zeta2_counts_qtls[1], zeta2_counts_qtls[-2], step='mid', color='b', alpha=alpha/2., label='')
plt.fill_between(zeta_bins_mid, zeta2_counts_qtls[0], zeta2_counts_qtls[-1], step='mid', color='b', alpha=alpha/4., label='')
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_zeta2_2_small_compare_credible.pdf')
    plt.close()

plt.show()

# Plot histograms of the numbers (fractions) of planet-pairs near MMRs:
lfs = 12

zeta_lim_colors = ['r', 'm']
zeta_lim_lines = ['--', '--']

# Fractions of planet-pairs near MMRs:
ax = plot_fig_pdf_simple([zeta1_counts_near_MMRs_all_dict[zeta_lim] for zeta_lim in zeta_in_mmr_lim_list], [], x_min=0.1, x_max=0.35, y_max=200, n_bins=30, normalize=False, lw=lw, ls_sim=zeta_lim_lines, c_sim=zeta_lim_colors, labels_sim=['SysSim catalogs, $|\zeta_{1}| \leq %s$' % zeta_lim for zeta_lim in zeta_in_mmr_lim_list], xlabel_text=r'Fraction of planet-pairs near MMR', ylabel_text='Counts', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for i,zeta_lim in enumerate(zeta_in_mmr_lim_list):
    plt.axvline(x=zeta1_counts_near_MMRs_Kep_dict[zeta_lim], lw=lw, color=zeta_lim_colors[i], label=r'Kepler catalog, $|\zeta_{1}| \leq %s$' % zeta_lim)
    plt.text(zeta1_counts_near_MMRs_Kep_dict[zeta_lim], 0.95, '{:0.2f}'.format(zeta1_counts_near_MMRs_Kep_dict[zeta_lim]), color=zeta_lim_colors[i], ha='left', fontsize=12, transform=ax.get_xaxis_transform())
    plt.text(zeta1_counts_near_MMRs_qtls_dict[zeta_lim][1], 0.4, '${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(zeta1_counts_near_MMRs_qtls_dict[zeta_lim][1], zeta1_counts_near_MMRs_qtls_dict[zeta_lim][1]-zeta1_counts_near_MMRs_qtls_dict[zeta_lim][0], zeta1_counts_near_MMRs_qtls_dict[zeta_lim][2]-zeta1_counts_near_MMRs_qtls_dict[zeta_lim][1]), color=zeta_lim_colors[i], ha='center', fontsize=12, transform=ax.get_xaxis_transform())
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_planet_fraction_near_MMRs_credible.pdf')
    plt.close()

# Fractions of planet-pairs just wide of MMRs:
ax = plot_fig_pdf_simple([zeta1_counts_near_above_MMRs_all_dict[zeta_lim] for zeta_lim in zeta_in_mmr_lim_list], [], x_min=0.025, x_max=0.225, y_max=200, n_bins=30, normalize=False, lw=lw, ls_sim=zeta_lim_lines, c_sim=zeta_lim_colors, labels_sim=['SysSim catalogs, $-%s \leq \zeta_{1} \leq 0$' % zeta_lim for zeta_lim in zeta_in_mmr_lim_list], xlabel_text=r'Fraction of planet-pairs just wide of MMR', ylabel_text='Counts', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for i,zeta_lim in enumerate(zeta_in_mmr_lim_list):
    plt.axvline(x=zeta1_counts_near_above_MMRs_Kep_dict[zeta_lim], lw=lw, color=zeta_lim_colors[i], label='Kepler catalog, $-%s \leq \zeta_{1} \leq 0$' % zeta_lim)
    plt.text(zeta1_counts_near_above_MMRs_Kep_dict[zeta_lim], 0.95, '{:0.2f}'.format(zeta1_counts_near_above_MMRs_Kep_dict[zeta_lim]), color=zeta_lim_colors[i], ha='left', fontsize=12, transform=ax.get_xaxis_transform())
    plt.text(zeta1_counts_near_above_MMRs_qtls_dict[zeta_lim][1], 0.5, '${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(zeta1_counts_near_above_MMRs_qtls_dict[zeta_lim][1], zeta1_counts_near_above_MMRs_qtls_dict[zeta_lim][1]-zeta1_counts_near_above_MMRs_qtls_dict[zeta_lim][0], zeta1_counts_near_above_MMRs_qtls_dict[zeta_lim][2]-zeta1_counts_near_above_MMRs_qtls_dict[zeta_lim][1]), color=zeta_lim_colors[i], ha='center', fontsize=12, transform=ax.get_xaxis_transform())
    
    # Compute + print some numbers:
    print('##### For zeta_lim = %s #####' % zeta_lim)
    print('Number of simulated catalogs with a fraction of planet-pairs just wide of MMRs (-%s < zeta_1 < 0) as high as in the Kepler catalog: %s' % (zeta_lim, np.sum(zeta1_counts_near_above_MMRs_all_dict[zeta_lim] >= zeta1_counts_near_above_MMRs_Kep_dict[zeta_lim])))
    fratios_Kepler_to_draws = zeta1_counts_near_above_MMRs_Kep_dict[zeta_lim]/zeta1_counts_near_above_MMRs_all_dict[zeta_lim]
    fratios_qtls = np.quantile(fratios_Kepler_to_draws, [0.16,0.5,0.84])
    print('F_Kep/F_draws = ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(fratios_qtls[1], fratios_qtls[1]-fratios_qtls[0], fratios_qtls[2]-fratios_qtls[1]))
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_planet_fraction_near_above_MMRs_credible.pdf')
    plt.close()

# Fractions of planet-pairs just narrow of MMRs:
ax = plot_fig_pdf_simple([zeta1_counts_near_below_MMRs_all_dict[zeta_lim] for zeta_lim in zeta_in_mmr_lim_list], [], x_min=0.025, x_max=0.225, y_max=200, n_bins=30, normalize=False, lw=lw, ls_sim=zeta_lim_lines, c_sim=zeta_lim_colors, labels_sim=['SysSim catalogs, $0 \leq \zeta_{1} \leq %s$' % zeta_lim for zeta_lim in zeta_in_mmr_lim_list], xlabel_text=r'Fraction of planet-pairs just narrow of MMR', ylabel_text='Counts', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for i,zeta_lim in enumerate(zeta_in_mmr_lim_list):
    plt.axvline(x=zeta1_counts_near_below_MMRs_Kep_dict[zeta_lim], lw=lw, color=zeta_lim_colors[i], label='Kepler catalog, $0 \leq \zeta_{1} \leq %s$' % zeta_lim)
    plt.text(zeta1_counts_near_below_MMRs_Kep_dict[zeta_lim], 0.95, '{:0.2f}'.format(zeta1_counts_near_below_MMRs_Kep_dict[zeta_lim]), color=zeta_lim_colors[i], ha='left', fontsize=12, transform=ax.get_xaxis_transform())
    plt.text(zeta1_counts_near_below_MMRs_qtls_dict[zeta_lim][1], 0.55, '${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(zeta1_counts_near_below_MMRs_qtls_dict[zeta_lim][1], zeta1_counts_near_below_MMRs_qtls_dict[zeta_lim][1]-zeta1_counts_near_below_MMRs_qtls_dict[zeta_lim][0], zeta1_counts_near_below_MMRs_qtls_dict[zeta_lim][2]-zeta1_counts_near_below_MMRs_qtls_dict[zeta_lim][1]), color=zeta_lim_colors[i], ha='center', fontsize=12, transform=ax.get_xaxis_transform())
    
    # Compute + print some numbers:
    print('##### For zeta_lim = %s #####' % zeta_lim)
    print('Number of simulated catalogs with a fraction of planet-pairs just narrow of MMRs (0 < zeta_1 < %s) as low as in the Kepler catalog: %s' % (zeta_lim, np.sum(zeta1_counts_near_below_MMRs_all_dict[zeta_lim] <= zeta1_counts_near_below_MMRs_Kep_dict[zeta_lim])))
    fratios_Kepler_to_draws = zeta1_counts_near_below_MMRs_Kep_dict[zeta_lim]/zeta1_counts_near_below_MMRs_all_dict[zeta_lim]
    fratios_qtls = np.quantile(fratios_Kepler_to_draws, [0.16,0.5,0.84])
    print('F_Kep/F_draws = ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(fratios_qtls[1], fratios_qtls[1]-fratios_qtls[0], fratios_qtls[2]-fratios_qtls[1]))
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_planet_fraction_near_below_MMRs_credible.pdf')
    plt.close()

plt.show()





##### To plot the multiplicity distribution for systems with vs. without planets near a resonance:
lfs = 16

n_mmr = 1 # 1 => only consider the 1st order MMRs; 2 => consider both 1st and 2nd order MMRs
pratio_max = 3. if n_mmr==1 else 4.

zeta_in_mmr_lim = 0.25 # all |zeta1| <= zeta_in_mmr_lim are considered "near a 1st order MMR"

def compute_mult_obs_with_and_without_mmrs(ss_per_sys, n_mmr=n_mmr, pratio_max=pratio_max, zeta_in_mmr_lim=zeta_in_mmr_lim):
    
    bools_mult = ss_per_sys['Mtot_obs'] >= 2
    bools_pr = [any((ss_per_sys['Rm_obs'][i] > 1.) & (ss_per_sys['Rm_obs'][i] < pratio_max)) for i in range(len(ss_per_sys['Rm_obs']))] # select small period ratios
    idx_sys_selected = np.where(bools_mult & bools_pr)[0]

    mult_obs_all = [] # to be filled with the observed multiplicity of each system with at least 2 observed planets; should be similar to the overall multiplicity distribution (without singles), but excluding some systems that don't have any small period ratios (mostly 2-planet systems)
    mult_obs_mmr = [] # to be filled with the observed multiplicity of each system with at least one period ratio near an MMR
    mult_obs_above_mmr = [] # to be filled with the observed multiplicity of each system with at least one period ratio near and above an MMR
    mult_obs_below_mmr = [] # to be filled with the observed multiplicity of each system with at least one period ratio near and below an MMR
    mult_obs_no_mmr = [] # to be filled with the observed multiplicity of each system with no period ratios near any MMRs
    for idx in idx_sys_selected:
        P_sys = ss_per_sys['P_obs'][idx]
        Pr_sys = ss_per_sys['Rm_obs'][idx]
        Rp_sys = ss_per_sys['radii_obs'][idx]
        
        Rp_sys = Rp_sys[P_sys > 0]
        P_sys = P_sys[P_sys > 0]
        Pr_sys = Pr_sys[Pr_sys > 0]
        
        mult = len(P_sys)
        assert mult == ss_per_sys['Mtot_obs'][idx]

        if n_mmr == 1:
            zeta_sys = zeta(Pr_sys, n=1, order=1) # zeta_{1,1}
            zeta_sys[Pr_sys > pratio_max] = np.nan
        elif n_mmr == 2:
            zeta_sys, in_1st_sys, in_2nd_sys = np.full(len(Pr_sys), np.nan), np.full(len(Pr_sys), False), np.full(len(Pr_sys), False)
            zeta_sys_small, in_1st_sys_small, in_2nd_sys_small = zeta_2_order(Pr_sys[Pr_sys < pratio_max]) # zeta_{2,1} or zeta_{2,2}; can also be NaN if did not check enough indices for period ratios -> 1
            zeta_sys[Pr_sys < pratio_max] = zeta_sys_small
            in_1st_sys[Pr_sys < pratio_max] = in_1st_sys_small
            in_2nd_sys[Pr_sys < pratio_max] = in_2nd_sys_small
        
        abs_zeta_min = np.nanmin(np.abs(zeta_sys))
        
        mult_obs_all.append(mult)
        if abs_zeta_min <= zeta_in_mmr_lim:
            mult_obs_mmr.append(mult)
            if any((zeta_sys >= -zeta_in_mmr_lim) & (zeta_sys <= 0.)):
                mult_obs_above_mmr.append(mult)
            if any((zeta_sys >= 0.) & (zeta_sys <= zeta_in_mmr_lim)):
                mult_obs_below_mmr.append(mult)
        else:
            mult_obs_no_mmr.append(mult)

    mult_obs_all = np.array(mult_obs_all)
    mult_obs_mmr = np.array(mult_obs_mmr)
    mult_obs_above_mmr = np.array(mult_obs_above_mmr)
    mult_obs_below_mmr = np.array(mult_obs_below_mmr)
    mult_obs_no_mmr = np.array(mult_obs_no_mmr)

    # Perform some checks/print some values:
    for n in range(2,8):
        N_mmr_n = np.sum(mult_obs_mmr == n)
        N_above_mmr_n = np.sum(mult_obs_above_mmr == n)
        N_below_mmr_n = np.sum(mult_obs_below_mmr == n)
        N_no_mmr_n = np.sum(mult_obs_no_mmr == n)
        assert N_mmr_n + N_no_mmr_n == np.sum(mult_obs_all == n)
        assert N_above_mmr_n <= N_mmr_n
        assert N_below_mmr_n <= N_mmr_n
        print('{:1}: {:<3} | {:<3} | {:<3}'.format(n, N_no_mmr_n, N_mmr_n, N_mmr_n/(N_mmr_n + N_no_mmr_n)))
        #print(n, ':', N_no_mmr_n, ' | ', N_mmr_n)

    # Put and return all results in a dict:
    mult_obs_dict = {'all': mult_obs_all, 'mmr': mult_obs_mmr, 'above_mmr': mult_obs_above_mmr, 'below_mmr': mult_obs_below_mmr, 'no_mmr': mult_obs_no_mmr}
    return mult_obs_dict

def compute_Nmult_obs_with_and_without_mmrs(mult_obs_dict, mult_max=None, m_geq=5):
    if mult_max is None:
        mult_max = np.max(mult_obs_dict['all'])
    
    Nmult_obs_dict = {}
    Nmult_obs_dict['all'] = bin_Nmult(np.array([np.sum(mult_obs_dict['all'] == x) for x in range(1,mult_max+1)]), m_geq=m_geq)[1:] # must apply 'bin_Nmult' before getting rid of first bin (number of ones)
    Nmult_obs_dict['mmr'] = bin_Nmult(np.array([np.sum(mult_obs_dict['mmr'] == x) for x in range(1,mult_max+1)]), m_geq=m_geq)[1:]
    Nmult_obs_dict['above_mmr'] = bin_Nmult(np.array([np.sum(mult_obs_dict['above_mmr'] == x) for x in range(1,mult_max+1)]), m_geq=m_geq)[1:]
    Nmult_obs_dict['below_mmr'] = bin_Nmult(np.array([np.sum(mult_obs_dict['below_mmr'] == x) for x in range(1,mult_max+1)]), m_geq=m_geq)[1:]
    Nmult_obs_dict['no_mmr'] = bin_Nmult(np.array([np.sum(mult_obs_dict['no_mmr'] == x) for x in range(1,mult_max+1)]), m_geq=m_geq)[1:]
    return Nmult_obs_dict



m_geq = 5 # bin the multiplicities higher than this

# First, compute the statistics for the Kepler catalog:
mult_obs_dict_Kep = compute_mult_obs_with_and_without_mmrs(ssk_per_sys)
Nmult_obs_dict_Kep = compute_Nmult_obs_with_and_without_mmrs(mult_obs_dict_Kep, m_geq=m_geq)
f_Nmult_obs_mmr_dict_Kep = {'mmr': Nmult_obs_dict_Kep['mmr']/Nmult_obs_dict_Kep['all'], 'above_mmr': Nmult_obs_dict_Kep['above_mmr']/Nmult_obs_dict_Kep['all'], 'below_mmr': Nmult_obs_dict_Kep['below_mmr']/Nmult_obs_dict_Kep['all']}

# Then, compute the statistics for each of the simulated catalogs:
Nmult_obs_sim_all_dict = {'all': [], 'mmr': [], 'above_mmr': [], 'below_mmr': [], 'no_mmr': []}
f_Nmult_obs_mmr_sim_all_dict = {'mmr': [], 'above_mmr': [], 'below_mmr': []}
for i,sss_per_sys_i in enumerate(sss_per_sys_all):
    mult_obs_dict_sim_i = compute_mult_obs_with_and_without_mmrs(sss_per_sys_i)
    Nmult_obs_dict_sim_i = compute_Nmult_obs_with_and_without_mmrs(mult_obs_dict_sim_i, m_geq=m_geq)
    
    Nmult_obs_sim_all_dict['all'].append(Nmult_obs_dict_sim_i['all'])
    Nmult_obs_sim_all_dict['mmr'].append(Nmult_obs_dict_sim_i['mmr'])
    Nmult_obs_sim_all_dict['above_mmr'].append(Nmult_obs_dict_sim_i['above_mmr'])
    Nmult_obs_sim_all_dict['below_mmr'].append(Nmult_obs_dict_sim_i['below_mmr'])
    Nmult_obs_sim_all_dict['no_mmr'].append(Nmult_obs_dict_sim_i['no_mmr'])
    
    f_Nmult_obs_mmr_sim_all_dict['mmr'].append(Nmult_obs_dict_sim_i['mmr']/Nmult_obs_dict_sim_i['all'])
    f_Nmult_obs_mmr_sim_all_dict['above_mmr'].append(Nmult_obs_dict_sim_i['above_mmr']/Nmult_obs_dict_sim_i['all'])
    f_Nmult_obs_mmr_sim_all_dict['below_mmr'].append(Nmult_obs_dict_sim_i['below_mmr']/Nmult_obs_dict_sim_i['all'])

Nmult_obs_sim_all_dict['all'] = np.array(Nmult_obs_sim_all_dict['all'])
Nmult_obs_sim_all_dict['mmr'] = np.array(Nmult_obs_sim_all_dict['mmr'])
Nmult_obs_sim_all_dict['above_mmr'] = np.array(Nmult_obs_sim_all_dict['above_mmr'])
Nmult_obs_sim_all_dict['below_mmr'] = np.array(Nmult_obs_sim_all_dict['below_mmr'])
Nmult_obs_sim_all_dict['no_mmr'] = np.array(Nmult_obs_sim_all_dict['no_mmr'])
f_Nmult_obs_mmr_sim_all_dict['mmr'] = np.array(f_Nmult_obs_mmr_sim_all_dict['mmr'])
f_Nmult_obs_mmr_sim_all_dict['above_mmr'] = np.array(f_Nmult_obs_mmr_sim_all_dict['above_mmr'])
f_Nmult_obs_mmr_sim_all_dict['below_mmr'] = np.array(f_Nmult_obs_mmr_sim_all_dict['below_mmr'])

Nmult_obs_sim_qtls_dict = {
    'all': np.quantile(Nmult_obs_sim_all_dict['all'], [0.16,0.5,0.84], axis=0),
    'mmr': np.quantile(Nmult_obs_sim_all_dict['mmr'], [0.16,0.5,0.84], axis=0),
    'above_mmr': np.quantile(Nmult_obs_sim_all_dict['above_mmr'], [0.16,0.5,0.84], axis=0),
    'below_mmr': np.quantile(Nmult_obs_sim_all_dict['below_mmr'], [0.16,0.5,0.84], axis=0),
    'no_mmr': np.quantile(Nmult_obs_sim_all_dict['no_mmr'], [0.16,0.5,0.84], axis=0)}
f_Nmult_obs_mmr_sim_qtls_dict = {
    'mmr': np.nanquantile(f_Nmult_obs_mmr_sim_all_dict['mmr'], [0.16,0.5,0.84], axis=0),
    'above_mmr': np.nanquantile(f_Nmult_obs_mmr_sim_all_dict['above_mmr'], [0.16,0.5,0.84], axis=0),
    'below_mmr': np.nanquantile(f_Nmult_obs_mmr_sim_all_dict['below_mmr'], [0.16,0.5,0.84], axis=0)}


# Plot the multiplicity distributions:
mult_edges = np.arange(1.5,len(f_Nmult_obs_mmr_dict_Kep['mmr'])+1+1.5)
mults = (mult_edges[1:]+mult_edges[:-1])/2

keys = ['all', 'mmr', 'no_mmr'] # keys for plotting 'Nmult_obs_sim_qtls_dict'
labels_keys = ['All', 'With MMR', 'No MMR']
colors_keys = ['k','r','b']
offsets_keys = [0, -0.1, 0.1]
#plot_fig_counts_hist_simple([mult_obs_dict_Kep['all'], mult_obs_dict_Kep['mmr'], mult_obs_dict_Kep['no_mmr']], [], x_min=1, x_max=7, y_max=200, x_llim=1.5, normalize=False, log_y=False, c_sim=colors_keys, ls_sim=['-','-','-'], lw=lw, labels_sim=['All', 'With MMR', 'No MMR'], xlabel_text='Observed planets per system', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_sim=False, show_counts_Kep=False, fig_size=fig_size, fig_lbrt=fig_lbrt, save_name=savefigures_directory + model_name + '_multiplicities_mmr.pdf', save_fig=savefigures)
plot_fig_counts_hist_simple([], [], x_min=1, x_max=5, y_min=0, y_max=250, x_llim=1.5, log_y=True, xlabel_text='Observed planets per system', ylabel_text='Number of systems', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
plt.xticks([2,3,4,5], [2,3,4,'5+'])
for i,key in enumerate(keys): #enumerate(Nmult_obs_sim_qtls_dict.keys())
    lerr = Nmult_obs_sim_qtls_dict[key][1]-Nmult_obs_sim_qtls_dict[key][0]
    uerr = Nmult_obs_sim_qtls_dict[key][2]-Nmult_obs_sim_qtls_dict[key][1]
    plt.stairs(Nmult_obs_dict_Kep[key], mult_edges, ls='--', lw=lw, color=colors_keys[i], label=labels_keys[i])
    plt.stairs(Nmult_obs_sim_qtls_dict[key][1], mult_edges, ls='-', lw=lw, color=colors_keys[i])
    eb = plt.errorbar(mults+offsets_keys[i], Nmult_obs_sim_qtls_dict[key][1], yerr=[lerr, uerr], ls='', elinewidth=2, c=colors_keys[i])
    #eb[-1][0].set_linestyle(':') # set the linestyle of the errorbars
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_multiplicities_mmr_compare_credible.pdf')
    plt.close()
plt.show(block=False)

# Plot the fraction of systems with planets near, near-above, and near-below resonances:
keys = ['mmr', 'above_mmr', 'below_mmr'] # keys for plotting 'f_Nmult_obs_mmr_sim_qtls_dict'
titles_keys = ['Planets near MMRs ($|\zeta_{2,1}| \leq %s$)' % zeta_in_mmr_lim, 'Planets just wide of MMRs ($-%s \leq \zeta_{2,1} \leq 0$)' % zeta_in_mmr_lim, 'Planets just narrow of MMRs ($0 \leq \zeta_{2,1} \leq %s$)' % zeta_in_mmr_lim]
for k,key in enumerate(keys):
    fig = plt.figure(figsize=(8,8))
    plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
    ax = plt.subplot(plot[0,0])
    plt.title(titles_keys[k], fontsize=tfs)
    plt.stairs(f_Nmult_obs_mmr_dict_Kep[key], mult_edges, color='k', label='Kepler catalog')
    plt.stairs(f_Nmult_obs_mmr_sim_qtls_dict[key][1], mult_edges, color='r') #, label='SysSim (median)'
    lerr = f_Nmult_obs_mmr_sim_qtls_dict[key][1]-f_Nmult_obs_mmr_sim_qtls_dict[key][0]
    uerr = f_Nmult_obs_mmr_sim_qtls_dict[key][2]-f_Nmult_obs_mmr_sim_qtls_dict[key][1]
    plt.errorbar(range(2,6), f_Nmult_obs_mmr_sim_qtls_dict[key][1], yerr=[lerr, uerr], ls='', c='r', label='SysSim catalogs (16-84%)')
    plt.violinplot([x[~np.isnan(x)] for x in [f_Nmult_obs_mmr_sim_all_dict[key][:,i] for i in range(4)]], positions=range(2,6), widths=0.5, showextrema=False) #, quantiles=np.transpose(np.tile(qtls_123sigma_lower_upper, (4,1)))
    for i,mult in enumerate(mults):
        plt.text(mult, 1.01*f_Nmult_obs_mmr_dict_Kep[key][i], '%s/%s = {:0.2f}'.format(f_Nmult_obs_mmr_dict_Kep[key][i]) % (Nmult_obs_dict_Kep[key][i], Nmult_obs_dict_Kep['all'][i]), color='k', ha='center', fontsize=12)
        plt.text(mult, 0.98*f_Nmult_obs_mmr_sim_qtls_dict[key][1][i], r'${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(f_Nmult_obs_mmr_sim_qtls_dict[key][1][i], lerr[i], uerr[i]), color='r', ha='left', va='top', fontsize=12)
    ax.tick_params(axis='both', labelsize=afs)
    #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim([1.5, 5.5])
    plt.xticks([2,3,4,5], [2,3,4,'5+'])
    plt.ylim([0., 1.])
    plt.xlabel('Observed planets per system', fontsize=tfs)
    plt.ylabel('Fraction of systems', fontsize=tfs)
    plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
    if savefigures:
        plt.savefig(savefigures_directory + model_name + '_multiplicities_fraction_with_planets_near_%s_compare_credible.pdf' % key)
        plt.close()
plt.show()
