# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm # for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for specifying plot attributes
from scipy.stats import wasserstein_distance # the "Earth-mover's" distance

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Observed/Radius_valley_measures/'
save_name = 'Models_compare'

compute_ratios = compute_ratios_adjacent
weights_all = load_split_stars_weights_only()
weights_all['all']['radii_delta_gap_KS'] = weights_all['all']['radii_KS'] # for now, set the weights for the radii deltas to be the same as for the radii
dists_include = ['depths_KS', 'radii_KS']





##### To load the Kepler catalog:

P_min, P_max = 3., 300.
radii_min, radii_max = 0.5, 10.

P_min_subsample, P_max_subsample = 0., 100.
radii_min_subsample, radii_max_subsample = 0.5, 4.
params_gapfit = {'m': -0.10, 'Rgap0': 2.4}

bw_factor = 0.25 # factor for multiplying the KDE bandwidth from Scott's rule, for fitting the radius distributions

ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)
compute_additional_stats_for_subsample_from_summary_stats(ssk, P_min=P_min_subsample, P_max=P_max_subsample, radii_min=radii_min_subsample, radii_max=radii_max_subsample, params=params_gapfit) # also compute the radii deltas of the restricted sample

# To calculate and plot the depth for the Kepler catalog:
depth_kde_Kep = measure_and_plot_radius_valley_depth_using_kde(ssk['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=False)
delta_depth_kde_Kep = measure_and_plot_radius_valley_depth_using_kde(ssk['radii_delta_gap_obs'], radius_valley_bounds=(-0.2,0.2), x_min=-1.5, x_max=3.5, bw_scotts_factor=bw_factor, xlabel_text=r'Gap subtracted radius, $R_p - R_{\rm gap}$ [$R_\oplus$]', plot_fig=False)
print(f'Kepler catalog: depth (kde) = {depth_kde_Kep}')
print(f'Kepler catalog: radii delta depth (kde) = {delta_depth_kde_Kep}')
plt.show()





##### To load and compute the same statistics for a large number of models:

#loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8p1_KS/Params9_fix_highM/GP_best_models_100/'
loadfiles_directory3 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8_KS/Params9_fix_highM/GP_best_models_100/'
#loadfiles_directory3 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_1000/'

model_names = ['Hybrid Model 1 (Run 5)', 'Hybrid Model 1 (Run 4)']
model_linestyles = ['--', '--']
model_colors = ['b', 'g']
model_alphas = [0.2, 0.2]
model_load_dirs = [loadfiles_directory2, loadfiles_directory3]
models = len(model_load_dirs)

runs_all = [100, 100]

sss_all = []
sss_per_sys_all = []
params_all = []
radii_measures_all = []

for r,loadfiles_dir in enumerate(model_load_dirs):
    runs = runs_all[r]
    
    sss_dir = []
    sss_per_sys_dir = []
    params_dir = []
    radii_measures = {'KS_dist_w': [],
                      'AD_dist_w': [],
                      'depth_kde': [],
                      'delta_depth_kde': []}
    
    for i in range(1,runs+1):
        print(i)
        run_number = i
        sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_dir, run_number=run_number, compute_ratios=compute_ratios)
        compute_additional_stats_for_subsample_from_summary_stats(sss_i, P_min=P_min_subsample, P_max=P_max_subsample, radii_min=radii_min_subsample, radii_max=radii_max_subsample, params=params_gapfit) # also compute the radii deltas of the restricted sample
        params_i = read_sim_params(loadfiles_dir + 'periods%s.out' % run_number)
        N_sim = params_i['num_targets_sim_pass_one']
        dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim)
        
        depth_kde_i = measure_and_plot_radius_valley_depth_using_kde(sss_i['radii_obs'], bw_scotts_factor=bw_factor)
        delta_depth_kde_i = measure_and_plot_radius_valley_depth_using_kde(sss_i['radii_delta_gap_obs'], radius_valley_bounds=(-0.2,0.2), x_min=-1.5, x_max=3.5, bw_scotts_factor=bw_factor)
        
        sss_all.append(sss_i)
        sss_per_sys_all.append(sss_per_sys_i)
        params_all.append(params_i)
        
        radii_measures['KS_dist_w'].append(dists_w_i['radii_KS'])
        radii_measures['AD_dist_w'].append(dists_w_i['radii_AD'])
        radii_measures['depth_kde'].append(depth_kde_i)
        radii_measures['delta_depth_kde'].append(delta_depth_kde_i)

    sss_all.append(sss_dir)
    sss_per_sys_all.append(sss_per_sys_dir)
    params_all.append(params_dir)
    radii_measures_all.append(radii_measures)
#####





##### To plot histograms of the measured depths:

fig_size = (8,5)
fig_lbrt = [0.15, 0.15, 0.95, 0.95]

n_bins = 50
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

x_max = 1.
plot_cdf = True # TODO: also implement the histogram version

# Plot the distributions:
fig = plt.figure(figsize=(8,5))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
plot_lines = []
for i,model_name in enumerate(model_names):
    # For the depths of the radius distribution:
    x = radii_measures_all[i]['depth_kde']
    l0, = plt.plot(np.sort(x), (np.arange(len(x))+1.)/float(len(x)), drawstyle='steps-post', color=model_colors[i], ls='-', lw=lw, label=model_name)
    
    # For the depths of the gap-substracted radius distribution:
    x = radii_measures_all[i]['delta_depth_kde']
    l1, = plt.plot(np.sort(x), (np.arange(len(x))+1.)/float(len(x)), drawstyle='steps-post', color=model_colors[i], ls='--', lw=lw)
    
    plot_lines.append([l0, l1])
lk0 = plt.axvline(x=depth_kde_Kep, ls='-', lw=lw, c='k', label='Kepler')
lk1 = plt.axvline(x=delta_depth_kde_Kep, ls='--', lw=lw, c='k')
plot_lines.append([lk0, lk1])
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.,x_max])
plt.ylim([0.,1.])
plt.xlabel(r'Depth of radius valley, $\Delta_{\rm valley}$', fontsize=tfs)
plt.ylabel('CDF', fontsize=tfs)
legend1 = plt.legend([l[0] for l in plot_lines], model_names + ['Kepler'], loc='upper right', bbox_to_anchor=(1,0.95), ncol=1, frameon=False, fontsize=lfs) # legend for different models (colors)
labels_linestyles = ['Radius distribution', 'Gap-subtracted radii \n($P < %s$d, $R_p < %s R_\oplus$)' % ('{:0.0f}'.format(P_max_subsample), '{:0.0f}'.format(radii_max_subsample))]
plt.legend([lk0, lk1], labels_linestyles, loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs) # legend for radii vs. gap-subtracted radii (linestyles)
plt.gca().add_artist(legend1)

if savefigures:
    fig_name = '_radius_valley_depth_cdfs.pdf' if plot_cdf else '_radius_valley_depth_hists.pdf'
    plt.savefig(savefigures_directory + save_name + fig_name)
    plt.close()
plt.show()
