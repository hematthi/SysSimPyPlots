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
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp/Params12_KS/durations_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Misc_Presentations/PhD_Thesis_Defense/Figures/'
run_number = ''
model_name = 'Clustered_P_R_Model' + run_number

compute_ratios = compute_ratios_adjacent
AD_mod = True
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

# To load and process the observed Kepler catalog:
stars_cleaned = load_Kepler_stars_cleaned()
Rstar_med = np.nanmedian(stars_cleaned['radius'])
Mstar_med = np.nanmedian(stars_cleaned['mass'])
teff_med = np.nanmedian(stars_cleaned['teff'])
#bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
bp_rp_corr_med = np.nanmedian(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'])

P_min, P_max, radii_min, radii_max = 3., 300., 0.5, 10.

ssk_per_sys0, ssk0 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
ssk_per_sys1, ssk1 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_max=bp_rp_corr_med) #_max=_med
ssk_per_sys2, ssk2 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=bp_rp_corr_med) #_min=_med





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

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp/Params12_KS/durations_KS/GP_best_models/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
runs = 100

samples = 2
sample_names = ['all'] + [str(i) for i in range(samples)]
sample_labels = ['All', 'Bluer', 'Redder'] # for samples=2 only
sample_colors = ['k', 'b', 'r'] # for samples=2 only
sample_bprp_bounds = np.quantile(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'], np.linspace(0,1,samples+1))

Nmult_bins = ['1','2','3','4','5+']

Nmult_Kep = {key: [] for key in sample_names}
Nmult_runs = {key: [] for key in sample_names}

N_pl_Kep = {}
N_pl_runs = {key: [] for key in sample_names} # total number of planets

# Kepler counts first:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
N_pl_Kep['all'] = np.sum(ssk['Nmult_obs'] * np.arange(1,len(ssk['Nmult_obs'])+1))
Nmult_Kep['all'] = bin_Nmult(ssk['Nmult_obs'])
for j in range(samples):
    ssk_per_sys_j, ssk_j = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=sample_bprp_bounds[j], bp_rp_max=sample_bprp_bounds[j+1])
    print('Kepler (sample %s): ' % j, ssk_j['Nmult_obs'])
    Nmult_Kep[str(j)] = bin_Nmult(ssk_j['Nmult_obs'])
    N_pl_Kep[str(j)] = np.sum(ssk_j['Nmult_obs'] * np.arange(1,len(ssk_j['Nmult_obs'])+1))

for i in range(1,runs+1): #range(1,runs+1)
    run_number = i
    print('#=====#')
    
    # Combined sample first:
    Mtot_obs, Nmult_obs = count_planets_from_loading_cat_obs_stars_only(file_name_path=loadfiles_directory, run_number=run_number)
    print('Sim %s (all): ' % i, Nmult_obs)
    Nmult_runs['all'].append(tuple(bin_Nmult(Nmult_obs)))
    N_pl_runs['all'].append(np.sum(Mtot_obs))
    
    # For each of the subsamples:
    for j in range(samples):
        Mtot_obs, Nmult_obs = count_planets_from_loading_cat_obs_stars_only(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_min=sample_bprp_bounds[j], bp_rp_max=sample_bprp_bounds[j+1])
        print('Sim %s (sample %s): ' % (i,j), Nmult_obs)
        Nmult_runs[str(j)].append(tuple(bin_Nmult(Nmult_obs)))
        N_pl_runs[str(j)].append(np.sum(Mtot_obs))

Nmult_runs['all'] = np.array(Nmult_runs['all'], dtype=[(str(n), 'i8') for n in Nmult_bins])
for j in range(samples):
    Nmult_runs[str(j)] = np.array(Nmult_runs[str(j)], dtype=[(str(n), 'i8') for n in Nmult_bins])





def pad_zero_beg_and_end(x):
    return np.array([0] + list(x) + [0])

##### To plot observed multiplicity distributions for each sample, with model uncertainties:
fig = plt.figure(figsize=(6,8))
plot = GridSpec(samples+1,1,left=0.2,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,0])
    Nmult_qtls = [np.quantile(Nmult_runs[sample][m], q=[0.16,0.5,0.84]) for m in Nmult_bins]
    Nmult_med = pad_zero_beg_and_end([qtl[1] for qtl in Nmult_qtls])
    Nmult_errlow = pad_zero_beg_and_end([qtl[1]-qtl[0] for qtl in Nmult_qtls])
    Nmult_errhigh = pad_zero_beg_and_end([qtl[2]-qtl[1] for qtl in Nmult_qtls])
    Nmult_err = np.array([Nmult_errlow, Nmult_errhigh])
    print(sample, ' --- ', Nmult_med)
    
    plt.errorbar(range(0,6+1), Nmult_med, yerr=Nmult_err, drawstyle='steps-mid', ls='--', color=sample_colors[i])
    #plt.scatter(range(1,6), Nmult_Kep[sample], marker='x', color=sample_colors[i])
    
    N_pl_qtls = np.quantile(N_pl_runs[sample], q=[0.16,0.5,0.84])
    plt.text(0.05, 0.1, r'${:0.0f}_{{-{:0.0f} }}^{{+{:0.0f} }}$ planets'.format(N_pl_qtls[1], N_pl_qtls[1]-N_pl_qtls[0], N_pl_qtls[2]-N_pl_qtls[1]), ha='left', va='bottom', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    plt.text(0.98, 0.95, sample_labels[i], ha='right', va='top', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    
    plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=16)
    #plt.yticks([0.5,1,1.5])
    plt.xlim([0.5,5.5])
    plt.ylim([3,1500])
    a = np.array(ax.get_xticks(), dtype='int').tolist()
    a[5] = '5+'
    ax.set_xticklabels(a)
    if i==1:
        plt.ylabel(r'$N_{\rm sim}$', fontsize=tfs)
    if i==2:
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_split_const_credible.pdf')
    plt.close()

##### To plot observed multiplicity distributions for each sample for the Kepler data:
fig = plt.figure(figsize=(6,8))
plot = GridSpec(samples+1,1,left=0.2,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,0])
    
    plt.plot(range(0,6+1), pad_zero_beg_and_end(Nmult_Kep[sample]), drawstyle='steps-mid', ls='-', color=sample_colors[i])
    plt.scatter(range(1,6), Nmult_Kep[sample], marker='x', color=sample_colors[i])
    plt.text(0.05, 0.1, r'${:0.0f}$ planets'.format(N_pl_Kep[sample]), ha='left', va='bottom', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    plt.text(0.98, 0.95, sample_labels[i], ha='right', va='top', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    
    plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=16)
    #plt.yticks([0.5,1,1.5])
    plt.xlim([0.5,5.5])
    plt.ylim([3,1500])
    a = np.array(ax.get_xticks(), dtype='int').tolist()
    a[5] = '5+'
    ax.set_xticklabels(a)
    if i==1:
        plt.ylabel(r'$N_{\rm Kep}$', fontsize=tfs)
    if i==2:
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_split_Kepler.pdf')
    plt.close()
plt.show()





##### Hardcoded version using Table 4 from https://arxiv.org/pdf/2003.04348.pdf:
# 0 = bluer, 1 = redder
#Nmult_med_const_table4 = {'all': [1239, 270, 91, 29, 11], '0': [639, 142, 47, 15, 5], '1': [599, 128, 44, 14, 5]}
#Nmult_errlow_const_table4 = {'all': [132, 30, 14, 6, 4], '0': [69, 17, 8, 4, 2], '1': [65, 17, 8, 4, 3]}
#Nmult_errhigh_const_table4 = {'all': [128, 30, 16, 8, 4], '0': [70, 18, 10, 5, 2], '1': [63, 17, 9, 4, 3]}

Nmult_med_linf_table4 = {'all': [1252, 269, 93, 29, 9], '0': [525, 116, 39, 13, 4], '1': [726, 152, 53, 17, 5]}
Nmult_errlow_linf_table4 = {'all': [109, 29, 14, 6, 4], '0': [69, 18, 8, 4, 2], '1': [78, 18, 8, 5, 2]}
Nmult_errhigh_linf_table4 = {'all': [110, 29, 14, 9, 4], '0': [68, 18, 9, 4, 3], '1': [80, 21, 10, 5, 3]}


fig = plt.figure(figsize=(6,8))
plot = GridSpec(samples+1,1,left=0.2,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,0])
    plt.axhline(y=1., ls=':', c='k') #label='Exact match'

    for j,bin in enumerate(Nmult_bins):
        nsim_q16, nsim_med, nsim_q84 = np.quantile(Nmult_runs[sample][bin], [0.16,0.5,0.84])
        ratio_q16, ratio_med, ratio_q84 = np.quantile(Nmult_runs[sample][bin]/float(Nmult_Kep[sample][j]), [0.16,0.5,0.84])
        #ratio_q16 = (Nmult_med_const_table4[sample][j] - Nmult_errlow_const_table4[sample][j])/Nmult_Kep[sample][j]
        #ratio_med = Nmult_med_const_table4[sample][j]/Nmult_Kep[sample][j]
        #ratio_q84 = (Nmult_med_const_table4[sample][j] + Nmult_errhigh_const_table4[sample][j])/Nmult_Kep[sample][j]
        plt.plot((j+0.5, j+1.5), (ratio_med, ratio_med), color=sample_colors[i], ls='--', lw=2, label='Constant model' if j==0 else None)
        plt.plot((j+1.05,j+1.05), (ratio_q16, ratio_q84), color=sample_colors[i], ls='--', lw=1)
        
        #plt.text(j+1, 1.3, r'${:0.0f}_{{-{:0.0f} }}^{{+{:0.0f} }}$'.format(nsim_med, nsim_med-nsim_q16, nsim_q84-nsim_med), ha='center')
        #plt.text(j+1, 0.7, str(Nmult_Kep[sample][j]), ha='center')
        
        # To also plot another model:
        ratio_q16 = (Nmult_med_linf_table4[sample][j] - Nmult_errlow_linf_table4[sample][j])/Nmult_Kep[sample][j]
        ratio_med = Nmult_med_linf_table4[sample][j]/Nmult_Kep[sample][j]
        ratio_q84 = (Nmult_med_linf_table4[sample][j] + Nmult_errhigh_linf_table4[sample][j])/Nmult_Kep[sample][j]
        plt.plot((j+0.5, j+1.5), (ratio_med, ratio_med), color=sample_colors[i], ls='-', lw=2, label=r'Linear $f_{\rm swpa}$ model' if j==0 else None)
        plt.plot((j+0.95,j+0.95), (ratio_q16, ratio_q84), color=sample_colors[i], ls='-', lw=1)

    plt.text(0.98, 0.95, sample_labels[i], ha='right', va='top', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    ax.tick_params(axis='both', labelsize=16)
    plt.yticks([0.5,1,1.5])
    plt.xlim([0.5,5.5])
    plt.ylim([0.4,1.6])
    a = np.array(ax.get_xticks(), dtype='int').tolist()
    a[5] = '5+'
    ax.set_xticklabels(a)
    if i==0:
        plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
    if i==1:
        plt.ylabel(r'$N_{\rm sim}/N_{\rm Kep}$', fontsize=tfs)
    if i==2:
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_split_normed_const_linear_fswpa_credible.pdf') #'Nmult_normed_const_credible.pdf', 'Nmult_split_normed_const_linear_fswpa_credible.pdf'
    plt.close()
plt.show()
