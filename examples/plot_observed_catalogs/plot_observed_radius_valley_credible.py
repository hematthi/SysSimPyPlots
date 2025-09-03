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
model_name = 'Hybrid_model'

compute_ratios = compute_ratios_adjacent
weights_all = load_split_stars_weights_only()
dists_include = ['depths_KS', 'radii_KS']





# To load the Kepler catalog:
P_min, P_max = 3., 300.
radii_min, radii_max = 0.5, 10.
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)





#####

n_bins = 100
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

##### To load and compute the same statistics for a large number of models:

#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_dtotmax15.0_depthmin0.29_models/' #GP_dtotmax15.0_depthmin0.29_models/' #GP_best_models_100/' #Radius_valley_model62_repeated_100/'
runs = 1000

sss_all = []
sss_per_sys_all = []
params_all = []

radii_measures = {'KS_dist_w': [],
                  'AD_dist_w': [],
                  'EMD': [],
                  #'IDCD': [],
                  'depth_binned': [],
                  'depth_kde': [],
                  #'depth_two_kdes': [],
                  'delta_depth_kde': []}

bw_factor = 0.25 # factor for multiplying the KDE bandwidth from Scott's rule

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    N_sim = params_i['num_targets_sim_pass_one']
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim)
    
    EMD_radii_i = wasserstein_distance(sss_i['radii_obs'], ssk['radii_obs']) # Earth mover's distance
    #IDCD_radii_i = integrate_difference_cdfs_dist(sss_i['radii_obs'], ssk['radii_obs']) # integrated differences in CDFs distance
    depth_binned_i = measure_and_plot_radius_valley_depth_using_global_bins(sss_i['radii_obs'], n_bins=n_bins)
    depth_kde_i = measure_and_plot_radius_valley_depth_using_kde(sss_i['radii_obs'], bw_scotts_factor=bw_factor)
    #depth_two_kdes_i = measure_and_plot_radius_valley_depth_using_two_kdes(sss_i['radii_obs'], bw_low_scotts_factor=bw_factor, bw_high_scotts_factor=2.)
    
    # Measure the depth from the distribution of radii differences from the location of the period-radius gap in the Kepler data:
    radii_delta_i = radii_delta_from_period_radius_gap(sss_i['radii_obs'], sss_i['P_obs'], m=-0.10, Rgap0=2.40)
    delta_depth_kde_i = measure_and_plot_radius_valley_depth_using_kde(radii_delta_i, radius_valley_bounds=(-0.2,0.2), x_min=-1.5, x_max=3.5, bw_scotts_factor=bw_factor)
    
    sss_all.append(sss_i)
    sss_per_sys_all.append(sss_per_sys_i)
    params_all.append(params_i)
    
    radii_measures['KS_dist_w'].append(dists_w_i['radii_KS'])
    radii_measures['AD_dist_w'].append(dists_w_i['radii_AD'])
    radii_measures['EMD'].append(EMD_radii_i)
    #radii_measures['IDCD'].append(IDCD_radii_i)
    radii_measures['depth_binned'].append(depth_binned_i)
    radii_measures['depth_kde'].append(depth_kde_i)
    #radii_measures['depth_two_kdes'].append(depth_two_kdes_i)
    radii_measures['delta_depth_kde'].append(delta_depth_kde_i)

#####





##### To calculate and plot the depth for the Kepler catalog:

save_dir_Kepler = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Observed/Radius_valley_measures/' # save the figures for the Kepler catalog here for now
savefigures_Kepler = False

depth_binned_Kep = measure_and_plot_radius_valley_depth_using_global_bins(ssk['radii_obs'], plot_fig=True, save_name=save_dir_Kepler + 'Kepler_radius_valley_depth_binned.pdf', save_fig=savefigures_Kepler)
depth_kde_Kep = measure_and_plot_radius_valley_depth_using_kde(ssk['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=True, save_name=save_dir_Kepler + 'Kepler_radius_valley_depth_kde.pdf', save_fig=savefigures_Kepler)

radii_delta_Kep = radii_delta_from_period_radius_gap(ssk['radii_obs'], ssk['P_obs'], m=-0.10, Rgap0=2.40)
radii_delta_depth_kde_Kep = measure_and_plot_radius_valley_depth_using_kde(radii_delta_Kep, radius_valley_bounds=(-0.2,0.2), x_min=-1.5, x_max=3.5, bw_scotts_factor=bw_factor, xlabel_text=r'Gap subtracted radius, $R_p - R_{\rm gap}$ [$R_\oplus$]', plot_fig=True, save_name=save_dir_Kepler + 'Kepler_radius_delta_valley_depth_kde.pdf', save_fig=savefigures_Kepler)
print(f'Kepler catalog: depth (binned) = {depth_binned_Kep}')
print(f'Kepler catalog: depth (kde) = {depth_kde_Kep}')
print(f'Kepler catalog: radii delta depth (kde) = {radii_delta_depth_kde_Kep}')
plt.show()





##### To sort the catalogs by the 'depth' of the radius valley and plot them:

sort_by = 'depth_kde' # 'KS_dist_w', 'AD_dist_w', 'EMD', 'depth_binned', 'depth_kde', 'delta_depth_kde'

print(f'Sorting the simulated catalogs by {sort_by}...')
iSort = np.argsort(radii_measures[sort_by])
if 'depth' in sort_by:
    # If sorting by a measure of depth, reverse so the indices are sorted by decreasing order:
    iSort = iSort[::-1]



fig_enlarged_size = (10,6)
fig_enlarged_lbrt = [0.15, 0.15, 0.95, 0.95]
radii_max_plot = 6.
y_max = 0.045

# First, plot the credible regions for the radius distribution (i.e. all catalogs) along with the Kepler catalog:
plot_fig_pdf_credible([[sss_i['radii_obs'] for sss_i in sss_all]], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max_plot, y_max=y_max, lw=lw, labels_sim_all=[r'Simulated 16-84%'], alpha_all=[alpha], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_enlarged_size, fig_lbrt=fig_enlarged_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_compare_enlarged.pdf')
    plt.close()

# Plot all the individual catalogs as thin lines along with the Kepler catalog:
plot_fig_pdf_simple([sss_i['radii_obs'] for sss_i in sss_all[:runs]] + [ssk['radii_obs']], [], x_min=radii_min, x_max=radii_max_plot, y_max=y_max, c_sim=['b']*runs + ['k'], ls_sim=['-']*runs + ['-'], lw=[0.05]*runs + [2.], labels_sim=['Simulated catalogs'] + [None]*(runs-1) + ['Kepler'], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_enlarged_size, fig_lbrt=fig_enlarged_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_compare_enlarged_draws.pdf')
    plt.close()

# Plot the top individual catalogs with the largest depths along with the Kepler catalog:
N_plot = 10
norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
cmap = cm.ScalarMappable(norm=norm, cmap='Blues_r')
cmap.set_array([])

plot_fig_pdf_simple([sss_all[i]['radii_obs'] for i in iSort[:N_plot]] + [ssk['radii_obs']], [], x_min=radii_min, x_max=radii_max_plot, y_max=y_max, c_sim=[cmap.to_rgba(i) for i in np.linspace(0.1,0.6,N_plot)] + ['k'], ls_sim=['-']*N_plot + ['-'], lw=[1.]*N_plot + [2.], labels_sim=['Simulated catalogs \n''with largest valleys'] + [None]*(N_plot-1) + ['Kepler'], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_enlarged_size, fig_lbrt=fig_enlarged_lbrt) # ls_sim=[(0,(1,1))]*N_plot + ['-']
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_compare_enlarged_draws_best%s_%s.pdf' % (N_plot, sort_by))
    plt.close()

plt.show()



# Finally, plot the individual catalogs with their measured depths:
for i in iSort[:10]:
    run_number = i+1 # the catalog/run numbers are 1-based
    save_name = savefigures_directory + model_name + '_radii_valley_%s_catalog%s.pdf' % (sort_by, run_number)
    if sort_by == 'depth_binned':
        depth = measure_and_plot_radius_valley_depth_using_global_bins(sss_all[i]['radii_obs'], plot_fig=True, save_name=save_name, save_fig=savefigures)
    elif sort_by == 'depth_kde':
        depth = measure_and_plot_radius_valley_depth_using_kde(sss_all[i]['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=True, save_name=save_name, save_fig=savefigures)
    elif sort_by == 'delta_depth_kde':
        radii_delta = radii_delta_from_period_radius_gap(sss_all[i]['radii_obs'], sss_all[i]['P_obs'], m=-0.10, Rgap0=2.40)
        depth = measure_and_plot_radius_valley_depth_using_kde(radii_delta, radius_valley_bounds=(-0.2,0.2), x_min=-1.5, x_max=3.5, bw_scotts_factor=bw_factor, xlabel_text=r'Gap subtracted radius, $R_p - R_{\rm gap}$ [$R_\oplus$]', plot_fig=True, save_name=save_name, save_fig=savefigures)
    else:
        # Sorted by a distance to the Kepler catalog, so just use your favorite measure:
        depth = measure_and_plot_radius_valley_depth_using_kde(sss_all[i]['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=True)
        print(f'i={i}: depth = {depth}, {sort_by} = {radii_measures[sort_by][i]}')
    print(f'i={i}: depth = {depth}')
plt.show()



# FOR 1000 SIMULATED CATALOGS PASSING A DEPTH THRESHOLD:
# Select and plot many individual catalogs above a given depth threshold:
depth_thres = 0.26 # 0.256 is about the 90% percentile from the 100 catalogs drawn from posterior
N_plot = 100

N_pass = np.sum(np.array(radii_measures[sort_by]) >= depth_thres)
print(f'Number of catalogs passing depth threshold of {depth_thres}: {N_pass}/{runs}')
assert N_pass >= N_plot
iSort_pass_thres = iSort[np.array(radii_measures[sort_by])[iSort] >= depth_thres]
iN_plot = np.sort(iSort_pass_thres)[:N_plot] # select the first 'N_plot' catalogs that pass the threshold; these will be in the original order (i.e. not sorted by depth)
assert np.all(np.array(radii_measures[sort_by])[iN_plot] >= depth_thres)

# Plot the individual catalogs:
radii_max_plot = 6.
y_max = 0.045

fig = plt.figure(figsize=(8,10))
plot = GridSpec(5,1,left=0.2,bottom=0.1,right=0.95,top=0.98,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0]) # credible regions of CDFs
plot_panel_cdf_credible(ax, [[sss_all[i]['radii_obs'] for i in iN_plot]], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max_plot, c_sim_all=['g'], ls_sim_all=['--'], lw=lw, labels_sim_all=['Hybrid Model 2,\nTop 10% of catalogs'], alpha_all=[alpha], xlabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=False)
ax.set_xticklabels([])
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:3,0]) # credible regions of histograms
plot_panel_pdf_credible(ax, [[sss_all[i]['radii_obs'] for i in iN_plot]], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max_plot, y_max=y_max, c_sim_all=['g'], ls_sim_all=['--'], lw=lw, labels_sim_all=[''], alpha_all=[alpha], xlabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=False)
ax.set_xticklabels([])
plt.text(x=0.98, y=0.95, s='16%-84% credible region', ha='right', va='top', fontsize=lfs, transform=ax.transAxes)

ax = plt.subplot(plot[3:,0]) # individual catalog draws/histograms
plot_panel_pdf_simple(ax, [sss_all[i]['radii_obs'] for i in iN_plot] + [ssk['radii_obs']], [], x_min=radii_min, x_max=radii_max_plot, y_max=y_max, c_sim=['g']*N_plot + ['k'], ls_sim=['-']*(N_plot+1), lw=[0.1]*N_plot + [lw], labels_sim=['catalogs'] + [None]*(N_plot-1) + ['Kepler'], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=False)
plt.text(x=0.98, y=0.95, s='Individual catalogs', ha='right', va='top', fontsize=lfs, transform=ax.transAxes)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_compare_enlarged_multipanel_depththres%s.pdf' % depth_thres)
    plt.close()
plt.show()





##### To plot the (active) model parameters of the simulated catalogs as a corner scatter plot, with the points colored by the depth:

# NOTE: when loading the parameters, the 'log' params are actually converted to unlogged values; see 'read_sim_params()':
active_params_names_symbols = {#'break_mass (M_earth)': r'$M_{p,\rm break}$ $(M_\oplus)$',
                               'log_rate_clusters': r'$\lambda_c$',
                               'log_rate_planets_per_cluster': r'$\lambda_p$',
                               'log_α_pret': r'$\alpha_{\rm ret}$',
                               'mean_ln_mass (ln M_earth)': r'$\mu_M$',
                               'norm_radius (R_earth)': r'$R_{p,\rm norm}$',
                               'power_law_P': r'$\alpha_P$',
                               'power_law_γ0': r'$\gamma_0$',
                               #'power_law_γ1': r'$\gamma_1$',
                               'power_law_σ0': r'$\sigma_0$',
                               #'power_law_σ1': r'$\sigma_1$',
                               'sigma_ln_mass (ln M_earth)': r'$\sigma_M$',
                               'sigma_ln_mass_in_cluster (ln M_earth)': r'$\sigma_{M,\rm cluster}$',
                               #'sigma_logperiod_per_pl_in_cluster': r'$\sigma_P$',
                               }
active_params_names = list(active_params_names_symbols.keys())
active_params_symbols = list(active_params_names_symbols.values())
active_params_all = np.array([[params[key] for key in active_params_names] for params in params_all]) # double list comprehension!

plot_points_corner(active_params_symbols, active_params_all, fpoints=radii_measures['depth_kde'], f_label='Depth (KDE)', cmap='Reds', points_size=10., fig_size=(16,16), save_name=savefigures_directory + model_name + '_params_depths_corner.pdf', save_fig=savefigures)
plt.show()





##### To plot the radii measures vs. each other to see how they correlate with each other:

radii_measures_pairs = [("KS_dist_w", "AD_dist_w"),
                        ("KS_dist_w", "EMD"),
                        #("KS_dist_w", "IDCD"),
                        #("KS_dist_w", "depth_binned"),
                        ("KS_dist_w", "depth_kde"),
                        ("AD_dist_w", "EMD"),
                        #("AD_dist_w", "IDCD"),
                        #("AD_dist_w", "depth_binned"),
                        ("AD_dist_w", "depth_kde"),
                        #("EMD", "IDCD"),
                        #("EMD", "depth_binned"),
                        ("EMD", "depth_kde"),
                        #("IDCD", "depth_kde"),
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
