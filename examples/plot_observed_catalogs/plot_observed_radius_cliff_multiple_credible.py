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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Observed/Compare_to_hybrid_nonclustered_and_H20/'
save_name = 'Models_compare'

compute_ratios = compute_ratios_adjacent
weights_all = load_split_stars_weights_only()
weights_all['all']['radii_delta_gap_KS'] = weights_all['all']['radii_KS'] # for now, set the weights for the radii deltas to be the same as for the radii
dists_include = ['depths_KS', 'radii_KS']





##### To load the Kepler catalog:

P_min, P_max = 3., 300.
radii_min, radii_max = 0.5, 10.

radii_min_cliff, radii_max_cliff = 2.5, 5.5 # bounds for the radius cliff
bw_factor = 0.25 # factor for multiplying the KDE bandwidth from Scott's rule, for fitting the (log) radius distributions

ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

# To fit and plot the radius cliff for the Kepler catalog:
m_cliff_Kep, _ = fit_and_plot_radius_cliff_using_kde(ssk['radii_obs'], x_min_cliff=radii_min_cliff, x_max_cliff=radii_max_cliff, bw_scotts_factor=bw_factor, xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', verbose=True, plot_fig=True, legend=True)
print(f'Kepler catalog: m_cliff = {m_cliff_Kep}')
plt.show()





##### To load and compute the same statistics for a large number of models:

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8p1_KS/Params9_fix_highM/GP_best_models_100/'
loadfiles_directory3 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/'

model_names = ['H20 model', 'Hybrid Model 1', 'Hybrid Model 2']
model_linestyles = ['-', '-', '-']
model_colors = ['silver', 'b', 'g']
model_alphas = [0.2, 0.2, 0.2]
model_load_dirs = [loadfiles_directory1, loadfiles_directory2, loadfiles_directory3]
models = len(model_load_dirs)

runs_all = [100, 100, 100]

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
                      'm_cliff': []}
    
    for i in range(1,runs+1):
        print(i)
        run_number = i
        sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_dir, run_number=run_number, compute_ratios=compute_ratios)
        params_i = read_sim_params(loadfiles_dir + 'periods%s.out' % run_number)
        N_sim = params_i['num_targets_sim_pass_one']
        dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim)
        
        m_cliff_i, _ = fit_and_plot_radius_cliff_using_kde(sss_i['radii_obs'], x_min_cliff=radii_min_cliff, x_max_cliff=radii_max_cliff, bw_scotts_factor=bw_factor)
        
        sss_all.append(sss_i)
        sss_per_sys_all.append(sss_per_sys_i)
        params_all.append(params_i)
        
        radii_measures['KS_dist_w'].append(dists_w_i['radii_KS'])
        radii_measures['AD_dist_w'].append(dists_w_i['radii_AD'])
        radii_measures['m_cliff'].append(m_cliff_i)

    sss_all.append(sss_dir)
    sss_per_sys_all.append(sss_per_sys_dir)
    params_all.append(params_dir)
    radii_measures_all.append(radii_measures)
#####





##### To plot histograms of the measured radius cliff slopes:

fig_size = (8,5)
fig_lbrt = [0.15, 0.15, 0.95, 0.95]

n_bins = 20
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

x_max = 1.

# Plot the distributions:
plot_fig_pdf_simple([radii_measures['m_cliff'] for radii_measures in radii_measures_all], [], n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$m_{\rm cliff}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
plt.axvline(x=m_cliff_Kep, ls='-', lw=lw, c='k', label='Kepler')
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_radius_cliff_slope_hists.pdf')
    plt.close()
plt.show()
