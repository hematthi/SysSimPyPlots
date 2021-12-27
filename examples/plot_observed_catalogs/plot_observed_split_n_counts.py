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





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
savefigures_directory = ''
run_number = ''
model_name = 'Clustered_P_R_Model' + run_number

compute_ratios = compute_ratios_adjacent
AD_mod = 'true' # 'true' or 'false'
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
param_vals = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
stars_cleaned = load_Kepler_stars_cleaned()
Rstar_med = np.nanmedian(stars_cleaned['radius'])
Mstar_med = np.nanmedian(stars_cleaned['mass'])
teff_med = np.nanmedian(stars_cleaned['teff'])
#bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
bp_rp_corr_med = np.nanmedian(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'])

ssk_per_sys0, ssk0 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
ssk_per_sys1, ssk1 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_max=bp_rp_corr_med) #_max=_med
ssk_per_sys2, ssk2 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=bp_rp_corr_med) #_min=_med

sss_per_sys0, sss0 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios) # combined sample
sss_per_sys1, sss1 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_max=bp_rp_corr_med, compute_ratios=compute_ratios)
sss_per_sys2, sss2 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_min=bp_rp_corr_med, compute_ratios=compute_ratios)





#'''
##### To plot the simulated and Kepler catalogs as marginal distributions:

subdirectory = ''

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 2 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
runs = 100

samples = 2
sample_names = ['all'] + [str(i) for i in range(samples)]
sample_bprp_bounds = np.quantile(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'], np.linspace(0,1,samples+1))

Nmult_bins = ['1','2','3','4','5+']

Nmult_Kep = {key: [] for key in sample_names}
Nmult_runs = {key: [] for key in sample_names}

# Kepler counts first:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
Nmult_Kep['all'] = bin_Nmult(ssk['Nmult_obs'])
for j in range(samples):
    ssk_per_sys_j, ssk_j = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=sample_bprp_bounds[j], bp_rp_max=sample_bprp_bounds[j+1])
    print('Kepler (sample %s): ' % j, ssk_j['Nmult_obs'])
    Nmult_Kep[str(j)] = bin_Nmult(ssk_j['Nmult_obs'])

param_vals_all = []
for i in range(1,runs+1): #range(1,runs+1)
    run_number = i
    
    param_vals_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    param_vals_all.append(param_vals_i)
    
    # Combined sample first:
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
    Nmult_runs['all'].append(tuple(bin_Nmult(sss_i['Nmult_obs'])))
    
    # For each of the subsamples:
    for j in range(samples):
        sss_per_sys_i_j, sss_i_j = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_min=sample_bprp_bounds[j], bp_rp_max=sample_bprp_bounds[j+1], compute_ratios=compute_ratios)
        print('Sim %s (sample %s): ' % (i,j), sss_i_j['Nmult_obs'])
        Nmult_runs[str(j)].append(tuple(bin_Nmult(sss_i_j['Nmult_obs'])))

Nmult_runs['all'] = np.array(Nmult_runs['all'], dtype=[(str(n), 'i8') for n in Nmult_bins])
for j in range(samples):
    Nmult_runs[str(j)] = np.array(Nmult_runs[str(j)], dtype=[(str(n), 'i8') for n in Nmult_bins])





fig = plt.figure(figsize=(8,8))
plot = GridSpec(samples+1,1,left=0.15,bottom=0.1,right=0.95,top=0.98,wspace=0,hspace=0)
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,0])
    plt.axhline(y=1., ls=':', c='k', label='Exact match')

    for j,bin in enumerate(Nmult_bins):
        nsim_q16, nsim_med, nsim_q84 = np.quantile(Nmult_runs[sample][bin], [0.16,0.5,0.84])
        ratio_q16, ratio_med, ratio_q84 = np.quantile(Nmult_runs[sample][bin]/float(Nmult_Kep[sample][j]), [0.16,0.5,0.84])
        plt.plot((j+0.5, j+1.5), (ratio_med, ratio_med), color='k', ls='-', lw=2)
        plt.plot((j+1,j+1), (ratio_q16, ratio_q84), color='k', ls='-', lw=1)
        plt.text(j+1, 1.3, r'${:0.0f}_{{-{:0.0f} }}^{{+{:0.0f} }}$'.format(nsim_med, nsim_med-nsim_q16, nsim_q84-nsim_med), ha='center')
        plt.text(j+1, 0.7, str(Nmult_Kep[sample][j]), ha='center')

    plt.text(0.98, 0.95, sample, ha='right', va='top', color='k', fontsize=lfs, transform=ax.transAxes)
    plt.yticks([0.5,1,1.5])
    plt.xlim([0.5,5.5])
    plt.ylim([0.4,1.6])
    a = np.array(ax.get_xticks(), dtype='int').tolist()
    a[5] = '5+'
    ax.set_xticklabels(a)
    if i==2:
        plt.ylabel(r'$N_{\rm sim}(m)/N_{\rm Kep}(m)$', fontsize=tfs)
    if i==4:
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_normed_many_binned5plus.pdf')
    plt.close()
