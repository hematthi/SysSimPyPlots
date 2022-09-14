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

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/Split_stars/Distances_compare/'

model_name = 'Clustered_P_R_Model'





# To load and process the observed Kepler catalog and compare with our simulated catalog:
stars_cleaned = load_Kepler_stars_cleaned()
Rstar_med = np.nanmedian(stars_cleaned['radius'])
Mstar_med = np.nanmedian(stars_cleaned['mass'])
teff_med = np.nanmedian(stars_cleaned['teff'])
#bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
bp_rp_corr_med = np.nanmedian(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'])

# To load the table at: 'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt':
EEM_table = np.genfromtxt("/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Miscellaneous_data/EEM_dwarf_UBVIJHK_colors_Teff.txt", skip_header=21, names=True)

# To define interpolation functions for getting various stellar parameters from bp_rp:
Teff_interp = scipy.interpolate.interp1d(EEM_table['BpRp'], EEM_table['Teff'])
Rstar_interp = scipy.interpolate.interp1d(EEM_table['BpRp'], EEM_table['R_Rsun'])
Mstar_interp = scipy.interpolate.interp1d(EEM_table['BpRp'], EEM_table['Msun'])





#'''

def load_split_stars_model_evaluations_weighted(file_name, dtot_max_keep=np.inf, max_keep=np.inf):
    sample_names = ['all', 'bluer', 'redder']

    Nmult_max = 8
    params_evals = []
    Nmult_evals = {key: [] for key in sample_names}
    d_used_keys_evals = {key: [] for key in sample_names}
    d_used_vals_w_evals = {key: [] for key in sample_names}
    d_used_vals_tot_w_evals = []

    with open(file_name, 'r') as file:
        for line in file:
            if line[0:20] == '# Active parameters:':
                param_names = np.array(line[23:-2].split('", "'))

            if line[0:14] == 'Active_params:':
                params = [float(x) for x in line[16:-2].split(', ')]
                params_evals.append(params)

            for key in sample_names:
                n = len(key)
                if line[0:n+2] == '[%s]' % key:
                    if line[n+3:n+3+6] == 'Counts':
                        Nmult_str, counts_str = line[n+3+9:-2].split('][')
                        Nmult = tuple([int(x) for x in Nmult_str.split(', ')])
                        Nmult_evals[key].append(Nmult)

                    if line[n+3:n+3+12] == 'd_used_keys:':
                        d_used_keys = line[n+3+15:-3].split('", "')
                        d_used_keys_evals[key].append(d_used_keys)

                    if line[n+3:n+3+12] == 'd_used_vals:':
                        d_used_vals_str, d_used_vals_tot_str = line[n+3+14:-2].split('][')
                        d_used_vals = tuple([float(x) for x in d_used_vals_str.split(', ')])
                        #d_used_vals_evals[key].append(d_used_vals)

                    elif line[n+3:n+3+13] == 'd_used_vals_w':
                        d_used_vals_w_str, d_used_vals_tot_w_str = line[n+3+16:-2].split('][')
                        d_used_vals_w = tuple([float(x) for x in d_used_vals_w_str.split(', ')])
                        d_used_vals_tot_w = float(d_used_vals_tot_w_str)
                        d_used_vals_w_evals[key].append(d_used_vals_w)

    d_used_vals_tot_w_evals = [np.sum(d_used_vals_w_evals['all'][i]) + np.sum(d_used_vals_w_evals['bluer'][i]) + np.sum(d_used_vals_w_evals['redder'][i]) for i in range(len(d_used_vals_w_evals['all']))]

    # Keep only the evals that pass a distance threshold:
    params_keep = []
    Nmult_keep = {key: [] for key in sample_names}
    d_used_keys_keep = {key: [] for key in sample_names}
    d_used_vals_w_keep = {key: [] for key in sample_names}
    d_used_vals_tot_w_keep = []
    for i,dtot_w in enumerate(d_used_vals_tot_w_evals):
        if (dtot_w <= dtot_max_keep) and (len(d_used_vals_tot_w_keep) < max_keep):
            params_keep.append(params_evals[i])
            for key in sample_names:
                Nmult_keep[key].append(Nmult_evals[key][i])
                d_used_keys_keep[key].append(d_used_keys_evals[key][i])
                d_used_vals_w_keep[key].append(d_used_vals_w_evals[key][i])
            d_used_vals_tot_w_keep.append(dtot_w)
    params_keep = np.array(params_keep)
    d_used_vals_tot_w_keep = np.array(d_used_vals_tot_w_keep)
    print('Evals passing threshold: ', len(d_used_vals_tot_w_keep))

    for key in sample_names:
        Nmult_evals[key] = np.array(Nmult_evals[key], dtype=[(str(n), 'i8') for n in range(1,Nmult_max+1)])
        d_used_keys_evals[key] = np.array(d_used_keys_evals[key])
        d_used_vals_w_evals[key] = np.array(d_used_vals_w_evals[key], dtype=[(dist_key, 'f8') for dist_key in d_used_keys_evals[key][0]])

        Nmult_keep[key] = np.array(Nmult_keep[key], dtype=[(str(n), 'i8') for n in range(1,Nmult_max+1)])
        d_used_keys_keep[key] = np.array(d_used_keys_keep[key])
        d_used_vals_w_keep[key] = np.array(d_used_vals_w_keep[key], dtype=[(dist_key, 'f8') for dist_key in d_used_keys_keep[key][0]])

    return param_names, params_keep, Nmult_keep, d_used_keys_keep, d_used_vals_w_keep

def compute_fswp_bprp_quantiles(bp_rp_axis, bp_rp_corr_med, param_names, params_evals, qtl=[0.16, 0.5, 0.84]):
    i_med = np.where(param_names == 'f_stars_with_planets_attempted_at_med_color')[0][0]
    i_slope = np.where(param_names == 'f_stars_with_planets_attempted_color_slope')[0][0]
    fswp_bprp_all = np.zeros((len(params_evals), len(bp_rp_axis)))
    for n in range(len(params_evals)):
        fswp_bprp_all[n,:] = linear_fswp_bprp(bp_rp_axis, bp_rp_corr_med, fswp_med=params_evals[n,i_med], slope=params_evals[n,i_slope])
    return np.quantile(fswp_bprp_all, qtl, axis=0)

def compute_alphaP_bprp_quantiles(bp_rp_axis, bp_rp_corr_med, param_names, params_evals, qtl=[0.16, 0.5, 0.84]):
    i_med = np.where(param_names == 'power_law_P_at_med_color')[0][0]
    i_slope = np.where(param_names == 'power_law_P_color_slope')[0][0]
    alphaP_bprp_all = np.zeros((len(params_evals), len(bp_rp_axis)))
    for n in range(len(params_evals)):
        alphaP_bprp_all[n,:] = linear_alphaP_bprp(bp_rp_axis, bp_rp_corr_med, alphaP_med=params_evals[n,i_med], slope=params_evals[n,i_slope])
    return np.quantile(alphaP_bprp_all, qtl, axis=0)



param_names_KS_1, params_evals_KS_1, Nmult_keep_KS_1, d_used_keys_keep_KS_1, d_used_vals_w_keep_1_KS = load_split_stars_model_evaluations_weighted('/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/Clustered_P_R_pass_GP_meanf75.0_thres47.0_pass1000_targs88912.txt', dtot_max_keep=47., max_keep=1000)

param_names_KS_2, params_evals_KS_2, Nmult_keep_KS_2, d_used_keys_keep_KS_2, d_used_vals_w_keep_2_KS = load_split_stars_model_evaluations_weighted('/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_alphaP_bprp/Params13_KS/durations_KS/GP_best_models/Clustered_P_R_pass_GP_meanf75.0_thres47.0_pass1000_targs88912.txt', dtot_max_keep=47., max_keep=1000)



##### To plot the f_swpa(bprp) vs bprp:

runs = 100

fswp_med_1, fswp_slope_1 = 0.6, 0.9
alphaP_med_2, alphaP_slope_2 = 0.7, -1.3

# 1 for linear fswp(bp-rp) model
loadfiles_directory1_KS = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
loadfiles_directory1_AD = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_AD/durations_AD/GP_best_models/'

# 2 for linear alphaP(bp-rp) model
loadfiles_directory2_KS = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_alphaP_bprp/Params13_KS/durations_KS/GP_best_models/' #catalogs_separate/
loadfiles_directory2_AD = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_alphaP_bprp/Params13_AD/durations_AD/GP_best_models/'

param_vals_all_KS_1 = []
param_vals_all_AD_1 = []
param_vals_all_KS_2 = []
param_vals_all_AD_2 = []
for i in range(1,runs+1): #range(1,runs+1)
    run_number = i

    param_vals_i = read_sim_params(loadfiles_directory1_KS+'periods%s.out' % run_number)
    param_vals_all_KS_1.append(param_vals_i)

    param_vals_i = read_sim_params(loadfiles_directory1_AD+'clusterids_all%s.out' % run_number)
    param_vals_all_AD_1.append(param_vals_i)

    param_vals_i = read_sim_params(loadfiles_directory2_KS+'periods%s.out' % run_number)
    param_vals_all_KS_2.append(param_vals_i)

    param_vals_i = read_sim_params(loadfiles_directory2_AD+'clusterids_all%s.out' % run_number)
    param_vals_all_AD_2.append(param_vals_i)

    print(i)





# Compute the fswp as a function of bprp for the simulated catalog, and the quantiles:

def compute_fswp_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all, qtl=[0.16, 0.5, 0.84]):
    fswps_at_bprp = []
    for pvs in param_vals_all:
        fswp_at_bprp = linear_fswp_bprp([bp_rp], bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"])[0]
        fswps_at_bprp.append(fswp_at_bprp)
    return np.quantile(fswps_at_bprp, qtl)

def compute_alphaP_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all, qtl=[0.16, 0.5, 0.84]):
    alphaPs_at_bprp = []
    for pvs in param_vals_all:
        alphaP_at_bprp = linear_alphaP_bprp([bp_rp], bp_rp_corr_med, alphaP_med=pvs["power_law_P_at_med_color"], slope=pvs["power_law_P_color_slope"])[0]
        alphaPs_at_bprp.append(alphaP_at_bprp)
    return np.quantile(alphaPs_at_bprp, qtl)

bp_rp_all = stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp']
bp_rp_10, bp_rp_90 = np.quantile(bp_rp_all, [0.1, 0.9])
pts = 101
bp_rp_array = np.linspace(np.min(bp_rp_all), np.max(bp_rp_all), pts)

fswp_qtls_KS_1 = compute_fswp_bprp_quantiles(bp_rp_array, bp_rp_corr_med, param_names_KS_1, params_evals_KS_1, qtl=[0.16, 0.5, 0.84]) ##### NEW
alphaP_qtls_KS_2 = compute_alphaP_bprp_quantiles(bp_rp_array, bp_rp_corr_med, param_names_KS_2, params_evals_KS_2, qtl=[0.16, 0.5, 0.84]) ##### NEW

fswp_sim_1 = linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=fswp_med_1, slope=fswp_slope_1)

alphaP_sim_2 = linear_alphaP_bprp(bp_rp_array, bp_rp_corr_med, alphaP_med=alphaP_med_2, slope=alphaP_slope_2)

fswp_16_KS_1, fswp_84_KS_1 = np.zeros(pts), np.zeros(pts)
fswp_16_AD_1, fswp_84_AD_1 = np.zeros(pts), np.zeros(pts)
alphaP_16_KS_2, alphaP_84_KS_2 = np.zeros(pts), np.zeros(pts)
alphaP_16_AD_2, alphaP_84_AD_2 = np.zeros(pts), np.zeros(pts)
for i,bp_rp in enumerate(bp_rp_array):
    fswp_16_KS_1[i], fswp_84_KS_1[i] = compute_fswp_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all_KS_1, qtl=[0.16, 0.84])
    fswp_16_AD_1[i], fswp_84_AD_1[i] = compute_fswp_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all_AD_1, qtl=[0.16, 0.84])
    alphaP_16_KS_2[i], alphaP_84_KS_2[i] = compute_alphaP_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all_KS_2, qtl=[0.16, 0.84])
    alphaP_16_AD_2[i], alphaP_84_AD_2[i] = compute_alphaP_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all_AD_2, qtl=[0.16, 0.84])

# Plot the results:
afs = 16 #axes labels font size
tfs = 20 #text labels font size
lfs = 14 #legend labels font size

c0, c1, c2 = 'k', 'b', 'r'

fig = plt.figure(figsize=(12,14)) #(10,8), (7,8)

plot = GridSpec(1,1,left=0.1,bottom=0.88,right=0.95,top=0.98,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.hist(bp_rp_all, bins=bp_rp_array, label='FGK stars in our sample')
plt.axvline(x=bp_rp_corr_med, c='k', ls='--', label=r'Median $b_p - r_p - E^*$')
plt.annotate(r'10%', xy=(bp_rp_10, 0), xytext=(bp_rp_10, 2000), arrowprops=dict(arrowstyle="->"), ha='center', fontsize=lfs)
plt.annotate(r'90%', xy=(bp_rp_90, 0), xytext=(bp_rp_90, 1000), arrowprops=dict(arrowstyle="->"), ha='center', fontsize=lfs)
plt.xlim([np.min(bp_rp_all), np.max(bp_rp_all)])
ax.tick_params(labelbottom=False)
plt.yticks([])
#plt.ylabel('Number', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)

plot = GridSpec(2,1,left=0.1,bottom=0.07,right=0.95,top=0.85,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0])
plt.plot(bp_rp_array, fswp_sim_1, lw=3, c=c1, label=r'Linear $f_{\rm swpa}(b_p - r_p - E^*)$,' '\n' r'$\frac{df_{\rm swpa}}{d(b_p - r_p - E^*)} = %s$, $f_{\rm swpa,med} = %s$' % (fswp_slope_1, fswp_med_1))
#plt.fill_between(bp_rp_array, fswp_16_KS_1, fswp_84_KS_1, color=c1, alpha=0.4, label='') # label=r'$1\sigma$ region using KS'
plt.fill_between(bp_rp_array, fswp_qtls_KS_1[0,:], fswp_qtls_KS_1[2,:], color=c1, alpha=0.4, label='') ##### NEW
for pvs in param_vals_all_KS_1:
    plt.plot(bp_rp_array, linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"]), lw=1, c=c1, alpha=0.2, label=None)
#for pvs in param_vals_all_AD_1:
#    plt.plot(bp_rp_array, linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"]), lw=1, c='g', alpha=0.2, label=None)
plt.axhline(y=0.52, c=c0, lw=3, label=r'Constant $f_{\rm swpa}$ + $\alpha_P$')
plt.errorbar(1.48, 0.52, yerr=[[0.11], [0.17]], fmt='.', color=c0, lw=3, capsize=5, label='')
plt.axhline(y=0.55, c=c2, lw=3, label='') #r'Linear $\alpha_P(b_p - r_p)$ model'
plt.errorbar(1.52, 0.55, yerr=[[0.11], [0.14]], fmt='.', color=c2, lw=3, capsize=5, label='')
plt.axvline(x=bp_rp_corr_med, c='k', ls='--')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([np.min(bp_rp_all), np.max(bp_rp_all)])
plt.ylim([0., 1.])
plt.ylabel(r'$f_{\rm swpa}$', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(1.,0.), ncol=1, frameon=False, fontsize=lfs)
ax2, ax1_x = ax.twiny(), ax.get_xticks()
ax2_x = Teff_interp(ax1_x)
ax2.set_xticks(ax1_x)
ax2.set_xbound(ax.get_xbound())
ax2.set_xticklabels(['{:0.0f}'.format(x) for x in ax2_x], fontsize=12)
plt.title(r'$T_{\rm eff}$ (K)', x=1., y=0.999, ha='center', fontsize=12)

ax = plt.subplot(plot[1,0])
plt.plot(bp_rp_array, alphaP_sim_2, lw=3, c=c2, label=r'Linear $\alpha_P(b_p - r_p - E^*)$,' '\n' r'$\frac{d\alpha_P}{d(b_p - r_p - E^*)} = %s$, $\alpha_{P,\rm med} = %s$' % (alphaP_slope_2, alphaP_med_2))
#plt.fill_between(bp_rp_array, alphaP_16_KS_2, alphaP_84_KS_2, color=c2, alpha=0.4, label='') # label=r'$1\sigma$ region using KS'
plt.fill_between(bp_rp_array, alphaP_qtls_KS_2[0,:], alphaP_qtls_KS_2[2,:], color=c2, alpha=0.4, label='') ##### NEW
for pvs in param_vals_all_KS_2:
    plt.plot(bp_rp_array, linear_alphaP_bprp(bp_rp_array, bp_rp_corr_med, alphaP_med=pvs["power_law_P_at_med_color"], slope=pvs["power_law_P_color_slope"]), lw=1, c=c2, alpha=0.2, label=None)
#for pvs in param_vals_all_AD_2:
#    plt.plot(bp_rp_array, linear_alphaP_bprp(bp_rp_array, bp_rp_corr_med, alphaP_med=pvs["power_law_P_at_med_color"], slope=pvs["power_law_P_color_slope"]), lw=1, c='g', alpha=0.2, label=None)
plt.axhline(y=0.56, c=c0, lw=3, label='')
plt.errorbar(1.48, 0.56, yerr=[[0.52], [0.56]], fmt='.', color=c0, lw=3, capsize=5, label='')
plt.axhline(y=0.64, c=c1, lw=3, label='')
plt.errorbar(1.52, 0.64, yerr=[[0.58], [0.56]], fmt='.', color=c1, lw=3, capsize=5, label='')
plt.axvline(x=bp_rp_corr_med, c='k', ls='--')
plt.annotate('G2V (Solar)\n $1.0 M_\odot$', xy=(0.823, -1), xytext=(0.823, -0.8), arrowprops=dict(arrowstyle="->"), ha='center', color='r', fontsize=lfs)
plt.annotate('K5V\n $0.68 M_\odot$', xy=(1.45, -1), xytext=(1.45, -0.8), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
plt.annotate('K2V\n $0.78 M_\odot$', xy=(1.10, -1), xytext=(1.10, -0.8), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
plt.annotate('F5V\n $1.33 M_\odot$', xy=(0.587, -1), xytext=(0.587, -0.8), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([np.min(bp_rp_all), np.max(bp_rp_all)])
plt.ylim([-1., 2.2])
plt.xlabel(r'$b_p - r_p - E^*$ (mag)', fontsize=tfs)
plt.ylabel(r'$\alpha_P$', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_fswp_alphaP_bprp_lines_shaded_1sigma_KS.pdf')
    plt.close()
plt.show()
#'''





##### To calculate a weighted average f_swpa for each half:

def compute_weighted_average_fswp_quantiles(bp_rp_sample, bp_rp_corr_med, param_vals_all, qtl=[0.16, 0.5, 0.84]):
    wavg_fswps = []
    for pvs in param_vals_all:
        fswp_stars = linear_fswp_bprp(bp_rp_sample, bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"])
        wavg_fswp = np.sum(bp_rp_sample*fswp_stars) / np.sum(bp_rp_sample)
        wavg_fswps.append(wavg_fswp)
    return np.quantile(wavg_fswps, qtl)

bp_rp_bluer = bp_rp_all[bp_rp_all < bp_rp_corr_med]
bp_rp_redder = bp_rp_all[bp_rp_all > bp_rp_corr_med]
fswp_bluer_qtls = compute_weighted_average_fswp_quantiles(bp_rp_bluer, bp_rp_corr_med, param_vals_all_KS_1)
fswp_redder_qtls = compute_weighted_average_fswp_quantiles(bp_rp_redder, bp_rp_corr_med, param_vals_all_KS_1)





##### To remake f_swpa vs bprp plot again (without alpha_P panel):
'''
fig = plt.figure(figsize=(10,8)) #(10,8), (7,8)

plot = GridSpec(1,1,left=0.1,bottom=0.85,right=0.95,top=0.98,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.hist(bp_rp_all, bins=bp_rp_array, label='FGK stars in our sample')
plt.axvline(x=bp_rp_corr_med, c='k', ls='--', label=r'Median $b_p - r_p - E^*$')
plt.annotate(r'10%', xy=(bp_rp_10, 0), xytext=(bp_rp_10, 2000), arrowprops=dict(arrowstyle="->"), ha='center', fontsize=lfs)
plt.annotate(r'90%', xy=(bp_rp_90, 0), xytext=(bp_rp_90, 1000), arrowprops=dict(arrowstyle="->"), ha='center', fontsize=lfs)
plt.xlim([np.min(bp_rp_all), np.max(bp_rp_all)])
ax.tick_params(labelbottom=False)
plt.yticks([])
#plt.ylabel('Number', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)

plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.8,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0])
plt.plot(bp_rp_array, fswp_sim_1, lw=3, c=c1, label=r'$\frac{df_{\rm swpa}}{d(b_p - r_p - E^*)} = %s$, $f_{\rm swpa,med} = %s$' % (fswp_slope_1, fswp_med_1))
#plt.fill_between(bp_rp_array, fswp_16_KS_1, fswp_84_KS_1, color=c1, alpha=0.4, label='') # label=r'$1\sigma$ region using KS'
plt.fill_between(bp_rp_array, fswp_qtls_KS_1[0,:], fswp_qtls_KS_1[2,:], color=c1, alpha=0.4, label=r'1$\sigma$ region using KS') ##### NEW
for pvs in param_vals_all_KS_1:
    plt.plot(bp_rp_array, linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"]), lw=1, c=c1, alpha=0.2, label=None)
#for pvs in param_vals_all_AD_1:
#    plt.plot(bp_rp_array, linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"]), lw=1, c='g', alpha=0.2, label=None)
#plt.axhline(y=0.52, c=c0, lw=3, label=r'Constant $f_{\rm swpa}$ + $\alpha_P$')
#plt.errorbar(1.48, 0.52, yerr=[[0.11], [0.17]], fmt='.', color=c0, lw=3, capsize=5, label='')
#plt.axhline(y=0.55, c=c2, lw=3, label='') #r'Linear $\alpha_P(b_p - r_p)$ model'
#plt.errorbar(1.52, 0.55, yerr=[[0.11], [0.14]], fmt='.', color=c2, lw=3, capsize=5, label='')
plt.axvline(x=bp_rp_corr_med, c='k', ls='--')
plt.annotate('G2V (Solar)\n $1.0 M_\odot$', xy=(0.823, 0), xytext=(0.823, 0.05), arrowprops=dict(arrowstyle="->"), ha='center', color='r', fontsize=lfs)
plt.annotate('K5V\n $0.68 M_\odot$', xy=(1.45, 0), xytext=(1.45, 0.05), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
plt.annotate('K2V\n $0.78 M_\odot$', xy=(1.10, 0), xytext=(1.10, 0.05), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
plt.annotate('F5V\n $1.33 M_\odot$', xy=(0.587, 0), xytext=(0.587, 0.05), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([np.min(bp_rp_all), np.max(bp_rp_all)])
plt.ylim([0., 1.])
plt.xlabel(r'$b_p - r_p - E^*(b_p - r_p)$ (mag)', fontsize=tfs)
plt.ylabel(r'Fraction of stars with planets, $f_{\rm swpa}$', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(1.,0.15), ncol=1, frameon=False, fontsize=lfs)
ax2, ax1_x = ax.twiny(), ax.get_xticks()
ax2_x = Teff_interp(ax1_x)
ax2.set_xticks(ax1_x)
ax2.set_xbound(ax.get_xbound())
ax2.set_xticklabels(['{:0.0f}'.format(x) for x in ax2_x], fontsize=12)
plt.title(r'$T_{\rm eff}$ (K)', x=1., y=0.999, ha='center', fontsize=12)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_fswp_bprp_lines_shaded_1sigma_KS.pdf')
    plt.close()
plt.show()
'''
