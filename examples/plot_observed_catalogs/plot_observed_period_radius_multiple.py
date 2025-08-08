# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm # for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for specifying plot attributes
from matplotlib import ticker # for setting contour plots to log scale
from matplotlib.colors import LogNorm # for log color scales
from scipy.stats import wasserstein_distance, gaussian_kde

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Observed/' + 'Radius_valley_measures/Fit_some8_KS_params10/'
run_number = ''
model_name = 'Hybrid_NR20_AMD_model1' + run_number

compute_ratios = compute_ratios_adjacent
weights_all = load_split_stars_weights_only()
dists_include = ['depths_KS', 'radii_KS']





##### To load the files with the systems with observed planets:

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated observed catalog of stars and planets:
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor)





#####

n_bins = 100
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

##### To load and compute the same statistics for a large number of models:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/'
runs = 100

sss_all = []
sss_per_sys_all = []
params_all = []

radii_measures = {'KS_dist_w': [],
                  'AD_dist_w': [],
                  'EMD': [],
                  'depth_binned': [],
                  'depth_kde': []}

bw_factor = 0.25 # factor for multiplying the KDE bandwidth from Scott's rule

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor)
    
    EMD_radii_i = wasserstein_distance(sss_i['radii_obs'], ssk['radii_obs'])
    depth_binned_i = measure_and_plot_radius_valley_depth_using_global_bins(sss_i['radii_obs'], n_bins=n_bins)
    depth_kde_i = measure_and_plot_radius_valley_depth_using_kde(sss_i['radii_obs'], bw_scotts_factor=bw_factor)
    
    sss_all.append(sss_i)
    sss_per_sys_all.append(sss_per_sys_i)
    params_all.append(params_i)
    
    radii_measures['KS_dist_w'].append(dists_w_i['radii_KS'])
    radii_measures['AD_dist_w'].append(dists_w_i['radii_AD'])
    radii_measures['EMD'].append(EMD_radii_i)
    radii_measures['depth_binned'].append(depth_binned_i)
    radii_measures['depth_kde'].append(depth_kde_i)

#####





##### To sort the catalogs by the 'depth' of the radius valley (measured from the marginal distribution), and then plot a gaussian KDE fit to the period-radius distribution:

sort_by = 'depth_kde' # 'KS_dist_w', 'AD_dist_w', 'EMD', 'depth_binned', 'depth_kde'

print(f'Sorting the simulated catalogs by {sort_by}...')
iSort = np.argsort(radii_measures[sort_by])
if 'depth' in sort_by:
    # If sorting by a measure of depth, reverse so the indices are sorted by decreasing order:
    iSort = iSort[::-1]

# Define the grid for evaluating the KDEs:
use_log_radii = False # choose whether to use a log-scale for the radii
radii_min, radii_max = 0.5, 4. # optionally, further restrict the radii bounds (NOTE: this overwrites 'radii_min' and 'radii_max' read from the files)

logP_min, logP_max = np.log10(P_min), np.log10(P_max)
# NOTE: variable names still use 'logR_' regardless of whether a log-scale is used or not
if use_log_radii:
    logR_min, logR_max = np.log10(radii_min), np.log10(radii_max)
else:
    logR_min, logR_max = radii_min, radii_max
logP_grid, logR_grid = np.mgrid[logP_min:logP_max:100j, logR_min:logR_max:100j] # complex step size '100j' to include the upper bound
positions = np.vstack([logP_grid.ravel(), logR_grid.ravel()])

# Plot the period-radius distributions of the Kepler and simulated catalogs side-by-side:
for i in iSort[:10]:
    run_number = i+1 # the catalog/run numbers are 1-based
    sss_i = sss_all[i]
    
    # If want to verify the measured depths from the marginal radii distribution after sorting:
    plot_marginal = False
    if sort_by == 'depth_binned':
        depth = measure_and_plot_radius_valley_depth_using_global_bins(sss_i['radii_obs'], plot_fig=plot_marginal)
        print(f'i={i}: depth = {depth}')
    elif sort_by == 'depth_kde':
        depth = measure_and_plot_radius_valley_depth_using_kde(sss_i['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=plot_marginal)
        print(f'i={i}: depth = {depth}')
    else:
        # Sorted by a distance to the Kepler catalog, so just use your favorite measure:
        depth = measure_and_plot_radius_valley_depth_using_kde(sss_i['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=plot_marginal)
        print(f'i={i}: depth = {depth}, {sort_by} = {radii_measures[sort_by][i]}')
    
    # To perform a gaussian KDE for the log(period)-log(radius) distributions:
    radii_obs_sim = np.log10(sss_i['radii_obs']) if use_log_radii else sss_i['radii_obs']
    radii_obs_Kep = np.log10(ssk['radii_obs']) if use_log_radii else ssk['radii_obs']
    values_sim = np.vstack([np.log10(sss_i['P_obs']), radii_obs_sim])
    values_Kep = np.vstack([np.log10(ssk['P_obs']), radii_obs_Kep])
    kde_sim = gaussian_kde(values_sim)
    kde_Kep = gaussian_kde(values_Kep)
    f_sim = np.reshape(kde_sim(positions).T, np.shape(logP_grid))
    f_Kep = np.reshape(kde_Kep(positions).T, np.shape(logP_grid))
    
    # To plot the period-radius distributions of the simulated and Kepler catalogs:
    fig = plt.figure(figsize=(16,8))
    plot = GridSpec(1,2,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0.1,hspace=0)
    cmap = 'Blues' #'Blues' #'viridis'

    ax = plt.subplot(plot[:,0]) # for the Kepler distribution
    # KDE contours:
    plt.contourf(logP_grid, logR_grid, f_Kep, cmap=cmap)
    # Scatter points:
    plt.scatter(np.log10(ssk['P_obs']), radii_obs_Kep, s=5, marker='o', edgecolor='k', facecolor='none', label='Kepler')
    ax.tick_params(axis='both', labelsize=afs)
    xtick_vals = np.array([3,10,30,100,300])
    plt.xticks(np.log10(xtick_vals), xtick_vals)
    if use_log_radii:
        ytick_vals = np.array([0.5,1,2,4,10])
        plt.yticks(np.log10(ytick_vals), ytick_vals)
    plt.xlim([logP_min, logP_max])
    plt.ylim([logR_min, logR_max])
    plt.xlabel(r'Orbital period, $P$ [days]', fontsize=tfs)
    plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
    plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)

    ax = plt.subplot(plot[:,1]) # for the simulated distribution
    # KDE contours:
    plt.contourf(logP_grid, logR_grid, f_sim, cmap=cmap)
    # Scatter points:
    plt.scatter(np.log10(sss_i['P_obs']), radii_obs_sim, s=5, marker='o', edgecolor='b', facecolor='none', label='Simulated')
    ax.tick_params(axis='both', labelsize=afs)
    xtick_vals = np.array([3,10,30,100,300])
    plt.xticks(np.log10(xtick_vals), xtick_vals)
    if use_log_radii:
        ytick_vals = np.array([0.5,1,2,4,10])
        plt.yticks(np.log10(ytick_vals), [])
    plt.xlim([logP_min, logP_max])
    plt.ylim([logR_min, logR_max])
    plt.xlabel(r'Orbital period, $P$ [days]', fontsize=tfs)
    #plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
    plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)

    if savefigures:
        save_name = savefigures_directory + model_name + '_period_radius_catalog%s_vs_Kepler.pdf' % run_number
        plt.savefig(save_name)
        plt.close()

plt.show()

# Plot the period-radius distribution of just the simulated catalog (OR optionally, the difference in the KDEs between the Kepler and simulated catalogs), with the marginal distributions as side panels:
plot_difference = False
cmap = 'coolwarm' if plot_difference else 'Blues'
for i in iSort[:10]:
    run_number = i+1 # the catalog/run numbers are 1-based
    sss_i = sss_all[i]
    
    # To perform a gaussian KDE for the log(period)-log(radius) distributions:
    radii_obs_sim = np.log10(sss_i['radii_obs']) if use_log_radii else sss_i['radii_obs']
    radii_obs_Kep = np.log10(ssk['radii_obs']) if use_log_radii else ssk['radii_obs']
    values_sim = np.vstack([np.log10(sss_i['P_obs']), radii_obs_sim])
    values_Kep = np.vstack([np.log10(ssk['P_obs']), radii_obs_Kep])
    kde_sim = gaussian_kde(values_sim)
    kde_Kep = gaussian_kde(values_Kep)
    f_sim = np.reshape(kde_sim(positions).T, np.shape(logP_grid))
    f_Kep = np.reshape(kde_Kep(positions).T, np.shape(logP_grid))
    
    # To plot the period-radius distributions of the simulated and Kepler catalogs:
    fig = plt.figure(figsize=(12,8))
    plot = GridSpec(5, 7, left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

    ax = plt.subplot(plot[1:,:-1]) # main panel
    # KDE contours:
    if plot_difference:
        plt.contourf(logP_grid, logR_grid, f_sim - f_Kep, cmap=cmap)
    else:
        plt.contourf(logP_grid, logR_grid, f_sim, cmap=cmap)
    # Scatter points:
    plt.scatter(np.log10(sss_i['P_obs']), radii_obs_sim, s=5, marker='o', edgecolor='b', facecolor='none', label='Simulated planets')
    ax.tick_params(axis='both', labelsize=afs)
    xtick_vals = np.array([3,10,30,100,300])
    plt.xticks(np.log10(xtick_vals), xtick_vals)
    if use_log_radii:
        ytick_vals = np.array([0.5,1,2,4,10])
        plt.yticks(np.log10(ytick_vals), ytick_vals)
    plt.xlim([logP_min, logP_max])
    plt.ylim([logR_min, logR_max])
    plt.xlabel(r'Orbital period, $P$ [days]', fontsize=tfs)
    plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
    plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
    
    ax = plt.subplot(plot[0,:-1]) # top histogram
    bins = np.linspace(logP_min, logP_max, n_bins+1)
    plt.hist(np.log10(sss_i['P_obs']), bins=bins, histtype='step', color='b', ls='-', label='Simulated')
    plt.hist(np.log10(ssk['P_obs']), bins=bins, histtype='step', color='k', ls='-', label='Kepler')
    plt.xlim([logP_min, logP_max])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)
    
    ax = plt.subplot(plot[1:,-1]) # side histogram
    bins = np.linspace(logR_min, logR_max, n_bins+1)
    plt.hist(radii_obs_sim, bins=bins, histtype='step', orientation='horizontal', color='b', ls='-', label='Simulated')
    plt.hist(radii_obs_Kep, bins=bins, histtype='step', orientation='horizontal', color='k', ls='-', label='Kepler')
    plt.ylim([logR_min, logR_max])
    plt.xticks([])
    plt.yticks([])

    if savefigures:
        save_fn = model_name + '_period_radius_catalog%s_minus_Kepler.pdf' % run_number if plot_difference else model_name + '_period_radius_catalog%s.pdf' % run_number
        plt.savefig(savefigures_directory + save_fn)
        plt.close()

plt.show()
