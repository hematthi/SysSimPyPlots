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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Observed/' + 'Radius_valley_measures/Fit_some8p1_KS_params10/' #catalog62_repeated/'
model_name = 'Hybrid_NR20_AMD_model1'

compute_ratios = compute_ratios_adjacent
weights = load_split_stars_weights_only()['all']





##### Restrict the sample to focus on the radius valley:

P_min, P_max = 0., 100.
radii_min, radii_max = 0.5, 4.
params_gapfit = {'m': -0.10, 'Rgap0': 2.4}

str_range = 'P{:.0f}-{:.0f}_R{:.1f}-{:.1f}'.format(P_min, P_max, radii_min, radii_max)
model_name += '_%s' % str_range

# To load the Kepler catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(3., 300., 0.5, 10., compute_ratios=compute_ratios)

ssk_radii = ssk['radii_obs'][(ssk['radii_obs'] >= radii_min) & (ssk['radii_obs'] <= radii_max) & (ssk['P_obs'] >= P_min) & (ssk['P_obs'] <= P_max)]
ssk_periods = ssk['P_obs'][(ssk['radii_obs'] >= radii_min) & (ssk['radii_obs'] <= radii_max) & (ssk['P_obs'] >= P_min) & (ssk['P_obs'] <= P_max)]





#####

n_bins = 100
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

##### To load and compute the same statistics for a large number of models:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/' #Radius_valley_model62_repeated_100/'
runs = 100

sss_all = []
sss_radii_all = []
sss_periods_all = []
params_all = []

radii_measures = {'KS_dist_w': [],
                  'AD_dist_w': [],
                  'EMD': [],
                  'depth_binned': [],
                  'depth_kde': [],
                  'delta_depth_kde': []}

bw_factor = 0.25 # factor for multiplying the KDE bandwidth from Scott's rule

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    compute_additional_stats_for_subsample_from_summary_stats(sss_i, P_min=P_min, P_max=P_max, radii_min=radii_min, radii_max=radii_max, params=params_gapfit) # also compute the radii deltas of the restricted sample
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    
    # Restrict the sample:
    # NOTE: the summary statistics in 'sss_per_sys_i' and 'sss_i' (except for the radii deltas) are still of the full sample, NOT the restricted sample!
    sss_i_radii = sss_i['radii_obs'][(sss_i['radii_obs'] >= radii_min) & (sss_i['radii_obs'] <= radii_max) & (sss_i['P_obs'] >= P_min) & (sss_i['P_obs'] <= P_max)]
    sss_i_periods = sss_i['P_obs'][(sss_i['radii_obs'] >= radii_min) & (sss_i['radii_obs'] <= radii_max) & (sss_i['P_obs'] >= P_min) & (sss_i['P_obs'] <= P_max)]

    # Calculate the distances and measures on the restricted sample:
    KS_w_radii_i = weights['radii_KS'] * KS_dist(sss_i_radii, ssk_radii)[0]
    AD_w_radii_i = weights['radii_AD'] * AD_mod_dist(sss_i_radii, ssk_radii)
    EMD_radii_i = wasserstein_distance(sss_i_radii, ssk_radii) # Earth mover's distance
    depth_binned_i = measure_and_plot_radius_valley_depth_using_global_bins(sss_i_radii, n_bins=n_bins)
    depth_kde_i = measure_and_plot_radius_valley_depth_using_kde(sss_i_radii, bw_scotts_factor=bw_factor)
    
    # Measure the depth from the distribution of radii differences from the location of the period-radius gap in the Kepler data:
    radii_delta_i = sss_i['radii_delta_gap_obs'] #radii_delta_from_period_radius_gap(sss_i_radii, sss_i_periods, m=-0.10, Rgap0=2.40)
    delta_depth_kde_i = measure_and_plot_radius_valley_depth_using_kde(radii_delta_i, radius_valley_bounds=(-0.2,0.2), x_min=-1.5, x_max=3.5, bw_scotts_factor=bw_factor)
    
    sss_all.append(sss_i)
    sss_radii_all.append(sss_i_radii)
    sss_periods_all.append(sss_i_periods)
    params_all.append(params_i)
    
    radii_measures['KS_dist_w'].append(KS_w_radii_i)
    radii_measures['AD_dist_w'].append(AD_w_radii_i)
    radii_measures['EMD'].append(EMD_radii_i)
    radii_measures['depth_binned'].append(depth_binned_i)
    radii_measures['depth_kde'].append(depth_kde_i)
    radii_measures['delta_depth_kde'].append(delta_depth_kde_i)
    
    # Print the statistics:
    print('#####')
    print('Planets obs: (Kepler, simulated) = (%s, %s)' % (len(ssk_radii), len(sss_i_radii)))
    print('{:<8}: dist_w_KS = {:<8}, dist_w_AD = {:<8}, dist_EMD = {:<8}'.format('radii', np.round(KS_w_radii_i,4), np.round(AD_w_radii_i,4), np.round(EMD_radii_i,4)))
    print('#####')
#####





##### To calculate the depth for the Kepler catalog first:

save_dir_Kepler = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Observed/Radius_valley_measures/' # save the figures for the Kepler catalog here for now
savefigures_Kepler = False

depth_binned_Kep = measure_and_plot_radius_valley_depth_using_global_bins(ssk_radii, x_min=radii_min, x_max=radii_max, plot_fig=True, save_name=save_dir_Kepler + 'Kepler_%s_radius_valley_depth_binned.pdf' % str_range, save_fig=savefigures_Kepler)
depth_kde_Kep = measure_and_plot_radius_valley_depth_using_kde(ssk_radii, x_min=radii_min, x_max=radii_max, bw_scotts_factor=bw_factor, plot_fig=True, save_name=save_dir_Kepler + 'Kepler_%s_radius_valley_depth_kde.pdf' % str_range, save_fig=savefigures_Kepler)

radii_delta_min, radii_delta_max = radii_min-2., radii_max-1.5 # for plotting purposes

radii_delta_Kep = radii_delta_from_period_radius_gap(ssk_radii, ssk_periods, m=params_gapfit['m'], Rgap0=params_gapfit['Rgap0'])
radii_delta_depth_kde_Kep = measure_and_plot_radius_valley_depth_using_kde(radii_delta_Kep, radius_valley_bounds=(-0.2,0.2), x_min=radii_delta_min, x_max=radii_delta_max, bw_scotts_factor=bw_factor, xlabel_text=r'Gap subtracted radius, $R_p - R_{\rm gap}$ [$R_\oplus$]', plot_fig=True, save_name=save_dir_Kepler + 'Kepler_%s_radius_delta_valley_depth_kde.pdf' % str_range, save_fig=savefigures_Kepler)
print(f'Kepler catalog: depth (binned) = {depth_binned_Kep}')
print(f'Kepler catalog: depth (kde) = {depth_kde_Kep}')
print(f'Kepler catalog: radii delta depth (kde) = {radii_delta_depth_kde_Kep}')
plt.show()





##### To sort the catalogs by the 'depth' of the radius valley and plot them:

sort_by = 'delta_depth_kde' # 'KS_dist_w', 'AD_dist_w', 'EMD', 'depth_binned', 'depth_kde', 'delta_depth_kde'

print(f'Sorting the simulated catalogs by {sort_by}...')
iSort = np.argsort(radii_measures[sort_by])
if 'depth' in sort_by:
    # If sorting by a measure of depth, reverse so the indices are sorted by decreasing order:
    iSort = iSort[::-1]



fig_enlarged_size = (10,6)
fig_enlarged_lbrt = [0.15, 0.15, 0.95, 0.95]
y_max = 0.035

# First, plot the credible regions for the gap-subtracted radius distribution (i.e. all catalogs) along with the Kepler catalog:
plot_fig_pdf_credible([sss_i['radii_delta_gap_obs'] for sss_i in sss_all], [], [radii_delta_Kep], x_min=radii_delta_min, x_max=radii_delta_max, y_max=y_max, lw=lw, label_sim1=r'Simulated 16-84%', alpha=alpha, xlabel_text=r'Gap subtracted radius, $R_p - R_{\rm gap}$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_enlarged_size, fig_lbrt=fig_enlarged_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_delta_compare_enlarged.pdf')
    plt.close()

# Plot all the individual catalogs as thin lines along with the Kepler catalog:
plot_fig_pdf_simple([sss_i['radii_delta_gap_obs'] for sss_i in sss_all[:runs]] + [radii_delta_Kep], [], x_min=radii_delta_min, x_max=radii_delta_max, y_max=y_max, c_sim=['b']*runs + ['k'], ls_sim=['-']*runs + ['-'], lw=[0.05]*runs + [2.], labels_sim=['Simulated catalogs'] + [None]*(runs-1) + ['Kepler'], xlabel_text=r'Gap subtracted radius, $R_p - R_{\rm gap}$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_enlarged_size, fig_lbrt=fig_enlarged_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_delta_compare_enlarged_draws.pdf')
    plt.close()

# Plot the top individual catalogs with the largest depths along with the Kepler catalog:
N_plot = 10
norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
cmap = cm.ScalarMappable(norm=norm, cmap='Blues_r')
cmap.set_array([])

plot_fig_pdf_simple([sss_all[i]['radii_delta_gap_obs'] for i in iSort[:N_plot]] + [radii_delta_Kep], [], x_min=radii_delta_min, x_max=radii_delta_max, y_max=y_max, c_sim=[cmap.to_rgba(i) for i in np.linspace(0.1,0.6,N_plot)] + ['k'], ls_sim=['-']*N_plot + ['-'], lw=[1.]*N_plot + [2.], labels_sim=['Simulated catalogs \n''with largest valleys'] + [None]*(N_plot-1) + ['Kepler'], xlabel_text=r'Gap subtracted radius, $R_p - R_{\rm gap}$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_enlarged_size, fig_lbrt=fig_enlarged_lbrt) # ls_sim=[(0,(1,1))]*N_plot + ['-']
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_delta_compare_enlarged_draws_best%s_%s.pdf' % (N_plot, sort_by))
    plt.close()

plt.show()



# Finally, plot the individual catalogs with their measured depths:
for i in iSort[:10]:
    run_number = i+1 # the catalog/run numbers are 1-based
    save_name = savefigures_directory + model_name + '_radii_valley_%s_catalog%s.pdf' % (sort_by, run_number)
    if sort_by == 'depth_binned':
        depth = measure_and_plot_radius_valley_depth_using_global_bins(sss_radii_all[i], x_min=radii_min, x_max=radii_max, plot_fig=True, save_name=save_name, save_fig=savefigures)
    elif sort_by == 'depth_kde':
        depth = measure_and_plot_radius_valley_depth_using_kde(sss_radii_all[i], x_min=radii_min, x_max=radii_max, bw_scotts_factor=bw_factor, plot_fig=True, save_name=save_name, save_fig=savefigures)
    elif sort_by == 'delta_depth_kde':
        radii_delta = radii_delta_from_period_radius_gap(sss_radii_all[i], sss_periods_all[i], m=params_gapfit['m'], Rgap0=params_gapfit['Rgap0'])
        depth = measure_and_plot_radius_valley_depth_using_kde(radii_delta, radius_valley_bounds=(-0.2,0.2), x_min=radii_delta_min, x_max=radii_delta_max, bw_scotts_factor=bw_factor, xlabel_text=r'Gap subtracted radius, $R_p - R_{\rm gap}$ [$R_\oplus$]', plot_fig=True, save_name=save_name, save_fig=savefigures)
    else:
        # Sorted by a distance to the Kepler catalog, so just use your favorite measure:
        depth = measure_and_plot_radius_valley_depth_using_kde(sss_radii_all[i], x_min=radii_min, x_max=radii_max, bw_scotts_factor=bw_factor, plot_fig=True)
        print(f'i={i}: depth = {depth}, {sort_by} = {radii_measures[sort_by][i]}')
    print(f'i={i}: depth = {depth}')
plt.show()





##### To plot the radii measures vs. each other to see how they correlate with each other:

radii_measures_pairs = [("KS_dist_w", "AD_dist_w"),
                        ("KS_dist_w", "EMD"),
                        #("KS_dist_w", "depth_binned"),
                        ("KS_dist_w", "depth_kde"),
                        ("AD_dist_w", "EMD"),
                        #("AD_dist_w", "depth_binned"),
                        ("AD_dist_w", "depth_kde"),
                        #("EMD", "depth_binned"),
                        ("EMD", "depth_kde"),
                        #("depth_binned", "depth_kde"),
                        ("depth_kde", "delta_depth_kde"),
                        ]

N_panels = len(radii_measures_pairs)
cols = int(np.ceil(np.sqrt(N_panels))) # number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols # number of rows, such that rows*cols >= N_panels

# To plot radii measures as scatter plots:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.95,wspace=0.3,hspace=0.5)
for i,pair in enumerate(radii_measures_pairs):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])
    sc = plt.scatter(radii_measures[pair[0]], radii_measures[pair[1]], marker='.', c='k', s=10, alpha=1)
    if 'depth' in pair[0] and 'depth' in pair[1]:
        plt.plot([0,1],[0,1], ls='--')
        plt.xlim([0, 1.1*np.max(radii_measures[pair[0]])])
        plt.ylim([0, 1.1*np.max(radii_measures[pair[1]])])
    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel(pair[0], fontsize=16)
    plt.ylabel(pair[1], fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_measures_2D.pdf')
    plt.close()
plt.show()

