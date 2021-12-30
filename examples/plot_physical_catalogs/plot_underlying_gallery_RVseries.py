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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *
from src.functions_plot_catalogs import *
from src.functions_plot_params import *
from src.functions_compute_RVs import *





##### To load the underlying and observed populations:

savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Conditional_Venus_transiting/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Systems_conditional/Conditional_Venus/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number

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

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)
N_factor = N_sim/N_Kep

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)

# To load and combine multiple simulated catalogs:
#loadfiles_directory = '../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
#sssp_per_sys, sssp = load_cat_phys_multiple_and_compute_combine_summary_stats(loadfiles_directory, run_numbers=range(1,101), load_full_tables=True)





##### To plot galleries of systems conditioned on a given planet:

afs = 12 #axes labels font size
tfs = 12 #text labels font size
lfs = 12 #legend labels font size

P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [215.,235.], [0.9,1.0], [0.77,0.86] # Venus
det = False
conds = conditionals_dict(P_cond_bounds=P_cond_bounds, Rp_cond_bounds=Rp_cond_bounds, Mp_cond_bounds=Mp_cond_bounds, det=det)

N_obs_all = np.array([int(round(x)) for x in np.logspace(np.log10(5), np.log10(1000), 20)])
σ_1obs = 0.1
N_sample, repeat = 1000, 100

fname = 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p1_cases.txt' % (N_sample, repeat)
outputs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('i4','i4')+('f8',)*33)

#fig_name = savefigures_directory + model_name + '_systems_with_P%s_%s_R%s_%s_detected' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1])
fig_name = savefigures_directory + model_name + '_systems_with_transiting_Venus'
plot_systems_gallery_with_RVseries_conditional(sssp_per_sys, sssp, conds, outputs_RVs=outputs, mark_undet=False, fit_RVs=False, N_obs_all=N_obs_all, repeat=repeat, σ_1obs=σ_1obs, fig_size=(9,6), seed=1234, N_sample=10, N_per_plot=10, afs=afs, tfs=tfs, save_name_base=fig_name, save_fig=True)





##### To simulate and fit RV observations of systems conditioned on a given planet, to see how the measured K varies with number of observations: ##### OUTDATED CODE
'''
N_obs_all = np.array([5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 500])

P_cond, Rp_cond = [9.5,10.5], [1.,1.5]
#P_cond, Rp_cond = [4.5,5.5], [1.8,2.2]
#P_cond, Rp_cond = [4.5,5.5], [1.,1.1]
#P_cond, Rp_cond = [19.,21.], [3.,4.]
outputs = plot_and_fit_RVobs_systems_conditional(sssp_per_sys, sssp, P_cond, Rp_cond, N_obs_all, det=True, N_sample=10, repeat=100, save_name_base='no_name_fig', save_fig=False, show_fig=False)

fig = plt.figure(figsize=(16,8))
plot = GridSpec(11,2,left=0.1,bottom=0.1,right=0.975,top=0.95,wspace=0.2,hspace=5.)

ax = plt.subplot(plot[0:5,0])
plt.loglog(outputs['N_obs_min_20p'], outputs['K_cond']/outputs['K_sum'], 'o')
ax.tick_params(axis='both', labelsize=afs)
plt.xlabel(r'Number of obs. needed for RMSD(K_cond)/K_cond < 0.2', fontsize=tfs)
plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=tfs)

ax = plt.subplot(plot[5:10,0])
sc = plt.scatter(outputs['K_cond'], outputs['K_cond']/outputs['K_sum'], c=outputs['N_obs_min_20p'])
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xlabel(r'$K_{\rm cond}$', fontsize=tfs)
plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=tfs)

cax = plt.subplot(plot[10,0])
cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
cbar.set_label(r'Number of obs. needed for RMSD(K_cond)/K_cond < 0.2', fontsize=10)

ax = plt.subplot(plot[0:5,1])
plt.loglog(outputs['N_obs_min_20p'], outputs['K_cond'], 'o', color='r', label=r'$K_{\rm cond}$')
plt.loglog(outputs['N_obs_min_20p'], outputs['K_max'], 'o', color='b', label=r'$K_{\rm max}$')
plt.loglog(outputs['N_obs_min_20p'], outputs['K_sum'], 'o', color='k', label=r'$\sum{K}$')
ax.tick_params(axis='both', labelsize=afs)
plt.xlabel(r'Number of obs. needed for RMSD(K_cond)/K_cond < 0.2', fontsize=tfs)
plt.ylabel(r'$K$ (m/s)', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize=lfs)

ax = plt.subplot(plot[5:10,1])
sc = plt.scatter(outputs['K_cond'], outputs['K_cond']/outputs['K_sum'], c=outputs['rmsd_best'])
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xlabel(r'$K_{\rm cond}$', fontsize=tfs)
plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=tfs)

cax = plt.subplot(plot[10,1])
cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
cbar.set_label(r'RMSD(K_cond)', fontsize=10)

plt.show()
'''





##### To make scatter plots of K vs. P conditioned on a given planet:

#fig_name = savefigures_directory + model_name + '_systems_with_P%s_%s_R%s_%s_detected_K_P.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1])
fig_name = savefigures_directory + model_name + '_systems_with_transiting_Venus_K_P.pdf'
plot_scatter_K_vs_P_conditional(sssp_per_sys, sssp, conds, log_y=True, fig_size=(8,5), save_name=fig_name, save_fig=savefigures)
