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
import scipy.integrate #for numerical integration
import scipy.misc #for factorial function
from scipy.special import erf #error function, used in computing CDF of normal distribution
import scipy.interpolate #for interpolation functions
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

from syssimpyplots.plot_params import *
from syssimpyplots.optim import *





##### This module will be used to plot results of the optimization runs of our clustered model using bboptimize:

savefigures = False
plt.ioff()

#run_directory = 'Hybrid_NR20_AMD_model1/Fit_all_KS/Params12/'
#loadfiles_directory = '/Users/hematthi/Documents/NotreDame_Postdoc/CRC/Files/SysSim/Model_Optimization/' + run_directory + 'GP_files/'
run_directory = 'Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_all12_KS/Params10_fix_highM/'
loadfiles_directory = '/Users/hematthi/Documents/NPP_ARC_Modernize_Kepler/Personal_research/SysSim/Model_Optimization/' + run_directory + 'GP_files/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Model_Optimization/' + run_directory

model_name = 'Hybrid_NR20_AMD_model1'
split_stars = False





##### To iterate through each of the optimization runs (files), and extract the results:

active_params_symbols = [#r'$M_{\rm break,1}$',
                         r'$\ln{(\lambda_c)}$',
                         r'$\ln{(\lambda_p)}$',
                         r'$\ln{(\alpha_{\rm ret})}$',
                         r'$\mu_M$',
                         r'$R_{p,\rm norm}$',
                         r'$\alpha_P$',
                         r'$\gamma_0$',
                         #r'$\gamma_1$',
                         r'$\sigma_0$',
                         #r'$\sigma_1$',
                         r'$\sigma_M$',
                         r'$\sigma_{M,\rm cluster}$',
                         #r'$\sigma_P$',
                         ] # this list of parameter symbols must match the order of parameters in 'active_params_names'!

##### To read the file of recomputed distances:

N_best_save, keep_every = 10000, 1 #100000, 10
if split_stars:
    results = load_recomputed_distances_split_stars_file(loadfiles_directory + 'Hybrid1_recompute_optim_best%s_every%s_targs86760.txt' % (N_best_save, keep_every))
else:
    results = load_recomputed_distances_file(loadfiles_directory + 'Hybrid1_recompute_optim_best%s_every%s_targs86760.txt' % (N_best_save, keep_every))

##### To save the best parameter values and the recomputed distances for training a GP emulator:
#'''
save_path_name = loadfiles_directory + 'Active_params_recomputed_distances_table_best%s_every%s.txt' % (N_best_save, keep_every)
if split_stars:
    savetxt_active_params_recomputed_distances_table_split_stars(results, save_path_name)
else:
    savetxt_active_params_recomputed_distances_table(results, save_path_name)
#'''





##### To plot corner plots of the same points, with the new distance terms as a colorscale:
#'''
plot_function_heatmap_averaged_grid_given_irregular_points_corner(active_params_symbols, results['active_params_evals'], results['dtot_w_evals'], flabel=r'$\mathcal{D}_W$', show_points=False, save_name=savefigures_directory + model_name + '_recomputed_best%s_every%s_corner_cmap_dtot_w.pdf' % (N_best_save, keep_every), save_fig=savefigures)

#dist_terms = ['radii_partitioning_KS', 'radii_monotonicity_KS', 'gap_complexity_KS']
#dist_terms = ['radii_partitioning_AD', 'radii_monotonicity_AD', 'gap_complexity_AD']
#dist_terms = ['delta_f', 'mult_CRPD_r', 'periods_KS', 'period_ratios_KS', 'durations_KS', 'duration_ratios_KS', 'depths_KS', 'radii_KS', 'radius_ratios_KS']
#dist_terms = ['delta_f', 'mult_CRPD_r', 'periods_KS', 'depths_KS', 'radii_KS', 'radii_delta_valley_KS', 'radius_ratios_KS', 'radii_partitioning_KS', 'radii_monotonicity_KS']
dist_terms = ['delta_f', 'mult_CRPD_r', 'periods_KS', 'period_ratios_KS', 'durations_KS', 'duration_ratios_KS', 'depths_KS', 'radii_KS', 'radii_delta_valley_KS', 'radius_ratios_KS', 'radii_partitioning_KS', 'radii_monotonicity_KS']
for (i,key) in enumerate(dist_terms):
    results_cmap_key = results['d_used_vals_w_evals']['all'][key] if split_stars else results['d_used_vals_w_evals'][key] # still the weighted distances for the full sample, but this is in a dictionary if split_stars=True
    plot_function_heatmap_averaged_grid_given_irregular_points_corner(active_params_symbols, results['active_params_evals'], results_cmap_key, flabel=key, show_points=False, save_name=savefigures_directory + model_name + '_recomputed_best%s_every%s_corner_cmap_%s.pdf' % (N_best_save, keep_every, key), save_fig=savefigures)
plt.show()
#'''





##### To also plot corner plots of the *distances* (not the parameters!), to see if there are any (anti-)correlations implying tensions between distributions:
#'''
N_best_plot = 100
dists_plot = np.array(results['d_used_vals_w_evals'][:N_best_plot].tolist())
dtots_plot = results['dtot_w_evals'][:N_best_plot]
plot_points_corner(dist_terms, dists_plot, fpoints=dtots_plot, f_label=r'$\mathcal{D}_W$', cmap='Blues_r', points_size=10., fig_size=(16,16), save_name=savefigures_directory + model_name + '_recomputed_best%s_every%s_best%s_distances_corner.pdf' % (N_best_save, keep_every, N_best_plot), save_fig=savefigures)
plt.show()
#'''
