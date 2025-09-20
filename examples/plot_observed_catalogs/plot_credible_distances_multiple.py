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

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Observed/Compare_to_hybrid_nonclustered_and_H20/'
save_name = 'Models_compare'

compute_ratios = compute_ratios_adjacent
AD_mod = True
weights_all = load_split_stars_weights_only()
weights_all['all']['radii_delta_gap_KS'] = weights_all['all']['radii_KS'] # for now, set the weights for the radii deltas to be the same as for the radii
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 'period_ratios_KS',
                 'durations_KS',
                 #'durations_norm_circ_KS',
                 #'durations_norm_circ_singles_KS',
                 #'durations_norm_circ_multis_KS',
                 'duration_ratios_KS',
                 #'duration_ratios_nonmmr_KS',
                 #'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radii_KS',
                 'radii_delta_gap_KS',
                 'radius_ratios_KS',
                 'radii_partitioning_KS',
                 'radii_monotonicity_KS',
                 #'gap_complexity_KS',
                 ]
dists_symbols_KS = {
    "delta_f": r'$w D_f$',
    "mult_CRPD_r": r'$w \rho_{\rm CRPD}$',
    "periods_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{P\}$',
    "period_ratios_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\mathcal{P}\}$',
    "durations_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{t_{\rm dur}\}$',
    "durations_norm_circ_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{t_{\rm dur}/t_{\rm circ}\}$',
    "durations_norm_circ_singles_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{t_{\rm dur}/t_{\rm circ}\}_{1}$',
    "durations_norm_circ_multis_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{t_{\rm dur}/t_{\rm circ}\}_{2+}$',
    "duration_ratios_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\xi\}$',
    "duration_ratios_mmr_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\xi_{\rm res}\}$',
    "duration_ratios_nonmmr_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\xi_{\rm non-res}\}$',
    "depths_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\delta\}$',
    "radii_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{R_p\}$',
    "radii_delta_gap_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{R_p - R_{\rm gap}\}$',
    "radius_ratios_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\delta_{i+1}/\delta_i\}$',
    "radii_partitioning_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\mathcal{Q}_R\}$',
    "radii_monotonicity_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\mathcal{M}_R\}$',
    "gap_complexity_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\mathcal{C}\}$',
}
dists_summarystats_symbols = {
    "delta_f": r'$N_{p,\rm obs}/N_\bigstar$',
    "mult_CRPD_r": r'$\{N_m\}$',
    "periods_KS": r'$\{P\}$',
    "period_ratios_KS": r'$\{\mathcal{P}\}$',
    "durations_KS": r'$\{t_{\rm dur}\}$',
    "durations_norm_circ_KS": r'$\{t_{\rm dur}/t_{\rm circ}\}$',
    "durations_norm_circ_singles_KS": r'$\{t_{\rm dur}/t_{\rm circ}\}_{1}$',
    "durations_norm_circ_multis_KS": r'$\{t_{\rm dur}/t_{\rm circ}\}_{2+}$',
    "duration_ratios_KS": r'$\{\xi\}$',
    "duration_ratios_mmr_KS": r'$\{\xi_{\rm res}\}$',
    "duration_ratios_nonmmr_KS": r'$\{\xi_{\rm non-res}\}$',
    "depths_KS": r'$\{\delta\}$',
    "radii_KS": r'$\{R_p\}$',
    "radii_delta_gap_KS": r'$\{R_p - R_{\rm gap}\}$',
    "radius_ratios_KS": r'$\{\delta_{i+1}/\delta_i\}$',
    "radii_partitioning_KS": r'$\{\mathcal{Q}_R\}$',
    "radii_monotonicity_KS": r'$\{\mathcal{M}_R\}$',
    "gap_complexity_KS": r'$\{\mathcal{C}\}$',
}

# For computing the radii deltas for a sub-sample of planets:
P_min_subsample = 0.
P_max_subsample = 100.
radii_min_subsample = 0.5
radii_max_subsample = 4.
params_gapfit = {'m': -0.10, 'Rgap0': 2.4}





##### Functions for loading catalogs and computing distances:

def load_catalogs_and_compute_distances(load_dir, ssk_per_sys, ssk, weights, dists_include, runs=100, compute_ratios=compute_ratios_adjacent, AD_mod=True):

    sss_all = []
    sss_per_sys_all = []
    params_all = []
    dists_w_all = []
    for i in range(1,runs+1):
        run_number = i
        sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=load_dir, run_number=run_number, compute_ratios=compute_ratios)
        compute_additional_stats_for_subsample_from_summary_stats(sss_i, P_min=P_min_subsample, P_max=P_max_subsample, radii_min=radii_min_subsample, radii_max=radii_max_subsample, params=params_gapfit) # also compute the radii deltas (NOTE: will use some global variables)
        params_i = read_sim_params(load_dir + 'periods%s.out' % run_number)
        
        N_sim = params_i['num_targets_sim_pass_one']
        dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, AD_mod=AD_mod)
        
        sss_all.append(sss_i)
        sss_per_sys_all.append(sss_per_sys_i)
        params_all.append(params_i)
        dists_w_all.append(dists_w_i)
    
    # Create a dictionary to hold all the outputs:
    outputs = {}
    outputs['sss_all'] = sss_all
    outputs['sss_per_sys_all'] = sss_per_sys_all
    outputs['params_all'] = params_all
    outputs['dists_w_all'] = dists_w_all
    
    return outputs

def collect_distances_and_compute_qtls(dists_w_all, dists_include=dists_include, qtls=[0.16, 0.5, 0.84]):
    # 'dists_w_all': list of dicts containing the weighted distances for each summary statistic, for each catalog

    print(f'Computing quantiles in the weighted distances from {len(dists_w_all)} catalogs...')
    
    # Create a dictionary to hold all the outputs:
    outputs = {}
    for dist in dists_include:
        dist_w_all = [dists_w[dist] for dists_w in dists_w_all]
        dist_w_qtls = np.quantile(dist_w_all, qtls)
        
        outputs[dist] = {} # dictionary for this distance term, to contain the distances from all the catalogs and their quantiles
        outputs[dist]['all'] = dist_w_all
        outputs[dist]['qtls'] = dist_w_qtls
        
        print('%s : %s_{-%s}^{+%s}' % ('{:<20}'.format(dist), '{:0.2f}'.format(dist_w_qtls[1]), '{:0.2f}'.format(dist_w_qtls[1]-dist_w_qtls[0]), '{:0.2f}'.format(dist_w_qtls[2]-dist_w_qtls[1])))
    
    # Also calculate the total weighted distance for each catalog:
    dtot_w_all = np.sum([outputs[dist]['all'] for dist in dists_include], axis=0)
    dtot_w_qtls = np.quantile(dtot_w_all, qtls)
    
    outputs['dtot_w'] = {}
    outputs['dtot_w']['all'] = dtot_w_all
    outputs['dtot_w']['qtls'] = dtot_w_qtls
    
    return outputs
        




##### To load and compute the distances for a large number of models compared to the Kepler catalog, and compute their confidence intervals:

# To load the Kepler catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(3., 300., 0.5, 10., compute_ratios=compute_ratios)
compute_additional_stats_for_subsample_from_summary_stats(ssk, P_min=P_min_subsample, P_max=P_max_subsample, radii_min=radii_min_subsample, radii_max=radii_max_subsample, params=params_gapfit) # also compute the radii deltas

# To load the simulated catalogs and compute the distances from multiple models/sets of catalogs:
runs = 100

load_dirs = [
    '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/',
    '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8p1_KS/Params9_fix_highM/GP_best_models_100/',
    '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/',
    #'/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_all12_KS/Params10_fix_highM/GP_best_models_100/',
    #'/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_dtotmax12_depthmin0.29_models/',
    #'/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8_KS/Params9_fix_highM/Radius_valley_model62_repeated_100/',
    #'/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8_KS/Params9_fix_highM/Radius_valley_model89_repeated_100/',
]
model_names = ['H20 model', 'Hybrid Model 1', 'Hybrid model 2'] #['Posterior', 'Catalog 62 repeated', 'Catalog 89 repeated'] #['Hybrid model', 'Hybrid model, clustered initial masses']
model_linestyles = ['-', '-', '-']
model_colors = ['k', 'b', 'g']
model_outputs_catalogs = []
model_outputs_distances = []
for load_dir in load_dirs:
    outputs_catalogs = load_catalogs_and_compute_distances(load_dir, ssk_per_sys, ssk, weights_all['all'], dists_include, runs=runs, compute_ratios=compute_ratios, AD_mod=AD_mod)
    outputs_distances = collect_distances_and_compute_qtls(outputs_catalogs['dists_w_all'], dists_include=dists_include)
    model_outputs_catalogs.append(outputs_catalogs)
    model_outputs_distances.append(outputs_distances)

n_models = len(model_outputs_distances)





##### To plot histograms of the weighted distances:

fig_size = (8,5) # size of each panel (figure)
fig_lbrt = [0.15, 0.2, 0.95, 0.925]

n_bins = 20
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size



# Total weighted distance:
plot_fig_pdf_simple([outputs_distances['dtot_w']['all'] for outputs_distances in model_outputs_distances], [], n_bins=n_bins, normalize=False, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$\mathcal{D}_{W} (\rm KS)$', ylabel_text='Catalogs', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_sum_dtot_w.pdf')
    plt.close()

# Individual distance terms:
for dist in dists_include:
    plot_fig_pdf_simple([outputs_distances[dist]['all'] for outputs_distances in model_outputs_distances], [], n_bins=n_bins, normalize=False, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=dists_symbols_KS[dist], ylabel_text='Catalogs', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
    if savefigures:
        plt.savefig(savefigures_directory + save_name + '_%s.pdf' % dist)
        plt.close()

plt.show()



##### Make a single figure with all of the individual distance terms as stacked panels, with the same bounds:
##### Also be able to distinguish which distances were excluded from the total distance function but are still plotted (and thus in 'dists_include')
dists_excluded = ['period_ratios_KS', 'durations_KS', 'duration_ratios_KS'] # the distances excluded from the total distance function, but we still wish to plot

plot_cdfs = True # whether to plot the CDFs instead of histograms

n_dists = len(dists_include)
x_min, x_max = 0., 4. # bounds for plotting the weighted distances
x_lines = [1.] # draw vertical lines for these

fig = plt.figure(figsize=(16,16))
plot = GridSpec(1, 1, left=0.08, bottom=0.83, right=0.98, top=0.98, wspace=0, hspace=0)
# First, plot the total weighted distance:
dtot_w_all_models = [outputs_distances['dtot_w']['all'] - np.sum([outputs_distances[dist]['all'] for dist in dists_excluded], axis=0) for outputs_distances in model_outputs_distances]
ax = plt.subplot(plot[:,:])
if plot_cdfs:
    plot_panel_cdf_simple(ax, dtot_w_all_models, [], c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'Total weighted distance, $\mathcal{D}_{W} = \sum{w \mathcal{D}}$', ylabel_text='CDF', legend=True, afs=afs, tfs=tfs, lfs=lfs)
    # Plot the total weighted distances including the excluded distances anyway, as faded lines:
    for m,outputs_distances in enumerate(model_outputs_distances):
        x = outputs_distances['dtot_w']['all']
        ax.plot(np.sort(x), (np.arange(len(x))+1.)/float(len(x)), drawstyle='steps-post', color=model_colors[m], alpha=alpha, ls='-', lw=lw)
else:
    plot_panel_pdf_simple(ax, dtot_w_all_models, [], n_bins=n_bins, normalize=False, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'Total weighted distance, $\mathcal{D}_{W} = \sum{w \mathcal{D}}$', ylabel_text='Catalogs', legend=True, afs=afs, tfs=tfs, lfs=lfs)
    # TODO: Plot the total weighted distances including the excluded distances anyway
# Now plot the individual distance terms:
cols = 2
rows = int(np.ceil(n_dists/cols))
plot = GridSpec(rows, cols, left=0.08, bottom=0.06, right=0.98, top=0.75, wspace=0.2, hspace=0)
for i,dist in enumerate(dists_include):
    #row, col = np.divmod(i,cols) # to plot panels from left to right, then top to bottom
    col, row = np.divmod(i,rows) # to plot panels from top to bottom, then left to right
    ax = plt.subplot(plot[row,col])
    is_bottom_panel = row==rows-1 #i==n_dists-1
    if plot_cdfs:
        plot_panel_cdf_simple(ax, [outputs_distances[dist]['all'] for outputs_distances in model_outputs_distances], [], x_min=x_min, x_max=x_max, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, extra_text=dists_summarystats_symbols[dist], xlabel_text=r'Weighted distance, $w \mathcal{D}$' if is_bottom_panel else '', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
        if dist in dists_excluded:
            for line in ax.lines:
                line.set_alpha(alpha)
        plt.yticks([0., 0.5, 1.] if is_bottom_panel else [0.5, 1.])
    else:
        plot_panel_pdf_simple(ax, [outputs_distances[dist]['all'] for outputs_distances in model_outputs_distances], [], x_min=x_min, x_max=x_max, n_bins=n_bins, normalize=False, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, extra_text=dists_summarystats_symbols[dist], xlabel_text=r'Weighted distance, $w \mathcal{D}$' if is_bottom_panel else '', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
        if dist in dists_excluded:
            for line in ax.lines:
                line.set_alpha(alpha)
    for x in x_lines:
        plt.axvline(x=x, ls=':', lw=1, color='r')
    if not is_bottom_panel:
        plt.xticks([])
fig.supylabel('CDF' if plot_cdfs else 'Catalogs', fontsize=tfs)
if savefigures:
    fig_name = '_all_dists_multipanel_cdfs.pdf' if plot_cdfs else '_all_dists_multipanel_hists.pdf'
    plt.savefig(savefigures_directory + save_name + fig_name)
    plt.close()
plt.show()
