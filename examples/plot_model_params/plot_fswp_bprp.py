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
#savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/Best_models/GP_best_models/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/'
model_name = 'Clustered_P_R_Model'





# To load and process the observed Kepler catalog and compare with our simulated catalog:
stars_cleaned = load_Kepler_stars_cleaned()
Rstar_med = np.nanmedian(stars_cleaned['radius'])
Mstar_med = np.nanmedian(stars_cleaned['mass'])
teff_med = np.nanmedian(stars_cleaned['teff'])
#bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
bp_rp_corr_med = np.nanmedian(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'])

# To load the table at: 'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt':
EEM_table = np.genfromtxt("../../data/EEM_dwarf_UBVIJHK_colors_Teff.txt", skip_header=21, names=True)

# To define interpolation functions for getting various stellar parameters from bp_rp:
Teff_interp = scipy.interpolate.interp1d(EEM_table['BpRp'], EEM_table['Teff'])
Rstar_interp = scipy.interpolate.interp1d(EEM_table['BpRp'], EEM_table['R_Rsun'])
Mstar_interp = scipy.interpolate.interp1d(EEM_table['BpRp'], EEM_table['Msun'])





#'''
##### To plot the f_swpa(bprp) vs bprp:

runs = 100

fswp_med_1, slope_1 = 0.6, 0.9 # Paper II
fswp_med_2, slope_2 = 0.88, 0.9 # Paper III

loadfiles_directory1_KS = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
#loadfiles_directory1_AD = '../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_AD/durations_AD/GP_best_models/'

loadfiles_directory2_KS = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
#loadfiles_directory2_AD = '../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_norm_circ_singles_multis_GF2020_AD/GP_best_models/'

param_vals_all_KS_1 = []
#param_vals_all_AD_1 = []
param_vals_all_KS_2 = []
#param_vals_all_AD_2 = []
for i in range(1,runs+1): #range(1,runs+1)
    run_number = i
    
    param_vals_i = read_sim_params(loadfiles_directory1_KS+'observed_catalog_stars%s.csv' % run_number)
    param_vals_all_KS_1.append(param_vals_i)

    #param_vals_i = read_sim_params(loadfiles_directory1_AD+'periods%s.out' % run_number)
    #param_vals_all_AD_1.append(param_vals_i)

    param_vals_i = read_sim_params(loadfiles_directory2_KS+'observed_catalog_stars%s.csv' % run_number)
    param_vals_all_KS_2.append(param_vals_i)
    
    #param_vals_i = read_sim_params(loadfiles_directory2_AD+'periods%s.out' % run_number)
    #param_vals_all_AD_2.append(param_vals_i)





# Compute the fswp as a function of bprp for the simulated catalog, and the quantiles:

def compute_fswp_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all, qtl=[0.16, 0.5, 0.84]):
    fswps_at_bprp = []
    for pvs in param_vals_all:
        fswp_at_bprp = linear_fswp_bprp([bp_rp], bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"])[0]
        fswps_at_bprp.append(fswp_at_bprp)
    return np.quantile(fswps_at_bprp, qtl)

bp_rp_all = stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp']
bp_rp_10, bp_rp_90 = np.quantile(bp_rp_all, [0.1, 0.9])
pts = 101
bp_rp_array = np.linspace(np.min(bp_rp_all), np.max(bp_rp_all), pts)

fswp_sim_1 = linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=fswp_med_1, slope=slope_1)
fswp_sim_2 = linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=fswp_med_2, slope=slope_2)

fswp_16_KS_1, fswp_84_KS_1 = np.zeros(pts), np.zeros(pts)
#fswp_16_AD_1, fswp_84_AD_1 = np.zeros(pts), np.zeros(pts)
fswp_16_KS_2, fswp_84_KS_2 = np.zeros(pts), np.zeros(pts)
#fswp_16_AD_2, fswp_84_AD_2 = np.zeros(pts), np.zeros(pts)
for i,bp_rp in enumerate(bp_rp_array):
    fswp_16_KS_1[i], fswp_84_KS_1[i] = compute_fswp_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all_KS_1, qtl=[0.16, 0.84])
    #fswp_16_AD_1[i], fswp_84_AD_1[i] = compute_fswp_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all_AD_1, qtl=[0.16, 0.84])
    fswp_16_KS_2[i], fswp_84_KS_2[i] = compute_fswp_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all_KS_2, qtl=[0.16, 0.84])
    #fswp_16_AD_2[i], fswp_84_AD_2[i] = compute_fswp_at_bprp_quantiles(bp_rp, bp_rp_corr_med, param_vals_all_AD_2, qtl=[0.16, 0.84])

# Plot the results:
afs = 16 #axes labels font size
tfs = 20 #text labels font size
lfs = 14 #legend labels font size

c1, c2 = 'b', 'g'

fig = plt.figure(figsize=(10,8)) #(10,8), (7,8)

plot = GridSpec(1,1,left=0.12,bottom=0.85,right=0.95,top=0.98,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.hist(bp_rp_all, bins=bp_rp_array, label='FGK stars')
plt.axvline(x=bp_rp_corr_med, c='k', ls='--', label='Median') #label=r'Median $b_p - r_p - E^*(b_p - r_p)$'
#plt.annotate(r'10%', xy=(bp_rp_10, 0), xytext=(bp_rp_10, 2000), arrowprops=dict(arrowstyle="->"), ha='center')
#plt.annotate(r'90%', xy=(bp_rp_90, 0), xytext=(bp_rp_90, 1000), arrowprops=dict(arrowstyle="->"), ha='center')
plt.xlim([np.min(bp_rp_all), np.max(bp_rp_all)])
ax.tick_params(labelbottom=False)
plt.yticks([])
#plt.ylabel('Number', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)

plot = GridSpec(1,1,left=0.12,bottom=0.1,right=0.95,top=0.8,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#plt.plot(bp_rp_array, fswp_sim_2, lw=3, c=c2, label=r'$\frac{df_{\rm swpa}}{d(b_p-r_p-E^*)} = %s$, $f_{\rm swpa,med} = %s$' % (slope_2, fswp_med_2))
plt.plot(bp_rp_array, fswp_sim_1, lw=3, c=c1, label=r'$\frac{df_{\rm swpa}}{d(b_p-r_p-E^*)} = %s$, $f_{\rm swpa,med} = %s$' % (slope_1, fswp_med_1))
#plt.text(x=0.98, y=0.7, s='Maximum AMD model', ha='right', fontsize=lfs, color=c2, transform=ax.transAxes)
#plt.text(x=0.98, y=0.65, s='Two-Rayleigh model', ha='right', fontsize=lfs, color=c1, transform=ax.transAxes)
#plt.fill_between(bp_rp_array, fswp_16_KS_2, fswp_84_KS_2, color=c2, alpha=0.4)
plt.fill_between(bp_rp_array, fswp_16_KS_1, fswp_84_KS_1, color=c1, alpha=0.4, label='$1\sigma$ region using KS') # label='$1\sigma$ region using KS'
#for pvs in param_vals_all_KS_2:
#    plt.plot(bp_rp_array, linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"]), lw=1, c=c2, alpha=0.2, label=None)
for pvs in param_vals_all_KS_1:
    plt.plot(bp_rp_array, linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"]), lw=1, c=c1, alpha=0.2, label=None)
#for pvs in param_vals_all_AD_1:
#    plt.plot(bp_rp_array, linear_fswp_bprp(bp_rp_array, bp_rp_corr_med, fswp_med=pvs["f_stars_with_planets_attempted_at_med_color"], slope=pvs["f_stars_with_planets_attempted_color_slope"]), lw=1, c='g', alpha=0.2, label=None)
plt.axvline(x=bp_rp_corr_med, c='k', ls='--')
plt.annotate('G2V (Solar)\n $1.0 M_\odot$', xy=(0.823, 0), xytext=(0.823, 0.05), arrowprops=dict(arrowstyle="->"), ha='center', color='r', fontsize=lfs)
plt.annotate('K5V\n $0.68 M_\odot$', xy=(1.45, 0), xytext=(1.45, 0.05), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
plt.annotate('K2V\n $0.78 M_\odot$', xy=(1.10, 0), xytext=(1.10, 0.05), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
plt.annotate('F5V\n $1.33 M_\odot$', xy=(0.587, 0), xytext=(0.587, 0.05), arrowprops=dict(arrowstyle="->"), ha='center', color='m', fontsize=lfs)
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([np.min(bp_rp_all), np.max(bp_rp_all)])
plt.ylim([0., 1.])
plt.xlabel(r'$b_p - r_p - E^*(b_p - r_p)$', fontsize=tfs)
plt.ylabel(r'$f_{\rm swpa}$', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(1.,0.15), ncol=1, frameon=False, fontsize=lfs)
ax2, ax1_x = ax.twiny(), ax.get_xticks()
ax2_x = Teff_interp(ax1_x)
ax2.set_xticks(ax1_x)
ax2.set_xbound(ax.get_xbound())
ax2.set_xticklabels(['{:0.0f}'.format(x) for x in ax2_x], fontsize=12)
plt.title(r'$T_{\rm eff}$ (K)', x=1., y=0.999, ha='center', fontsize=12)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_fswp_bprp_lines_shaded_1sigma_KS.pdf') #'_fswp_bprp_lines.pdf' #'_fswp_bprp_lines_1sigma_KS.pdf' #'_fswp_bprp_lines_shaded_1sigma_KS.pdf'
    plt.close()
plt.show()
#'''
