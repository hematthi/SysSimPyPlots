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





##### To load the stellar tables:

savefigures = False
savefigures_directory = ''
subdirectory = 'Stellar_samples/Kepler_data/E_BP_MIN_RP/Interpolate_e_bp_min_rp/e_bp_min_rp_dist/CKS_Berger2020_compare/' #'Paper_Figures/'; 'Talk_Figures/'

stars_cleaned = np.genfromtxt('../../../SysSimExClusters/plotting/q1_q17_dr25_gaia_berger_fgk_HFR2020b_cleaned.csv', dtype={'names': ('kepid', 'mass', 'radius', 'teff', 'bp_rp', 'e_bp_rp_interp', 'e_bp_rp_true', 'rrmscdpp04p5'), 'formats': ('i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')}, delimiter=',')
stars_cleaned = stars_cleaned[1:]

bprp_corr = stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_true']
#bprp_corr = stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp']
bprp_corr_med = np.nanmedian(bprp_corr)

cks_table = np.genfromtxt('/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Miscellaneous_data/cks_physical_merged.csv', names=True, delimiter=',')

# To match CKS and CKS+isochrone T_eff's to colors:
bprp_corr_match = []
teff_cks_match = []
teff_cks_iso_match = []
for i,kepid in enumerate(stars_cleaned['kepid']):
    bprp_corr_i = bprp_corr[i]
    idx = np.where(cks_table['id_kic'] == kepid)[0]
    if len(idx) > 0: # found a match in CKS table
        bprp_corr_match.append(bprp_corr_i)
        teff_cks_match.append(cks_table['cks_steff'][idx[0]])
        teff_cks_iso_match.append(cks_table['iso_steff'][idx[0]])
bprp_corr_match = np.array(bprp_corr_match)
teff_cks_match = np.array(teff_cks_match)
teff_cks_iso_match = np.array(teff_cks_iso_match)




##### To plot b_p-r_p-E*(b_p-r_p) vs. T_eff from various sources (CKS, CKS+isochrone, and Berger isochrone) for Paper II revision:

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
plt.plot(bprp_corr, stars_cleaned['teff'], '.', c='k', alpha=0.1, markersize=1, label='Berger et al. (2020) isochrone')
plt.plot(bprp_corr_match, teff_cks_match, '.', c='b', markersize=3, label='CKS spectra')
plt.plot(bprp_corr_match, teff_cks_iso_match, '.', c='r', markersize=3, label='CKS spectra+isochrone')
#plt.axvline(x=bprp_corr_med, ls='--')
ax.tick_params(axis='both', labelsize=16)
plt.xlim([0.5, 1.7])
plt.ylim([4000, 7000])
plt.xlabel(r'$b_p-r_p-E^*$ (mag)', fontsize=20)
plt.ylabel(r'$T_{\rm eff}$ (K)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=16)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'bprp_ebprp_vs_teff_cks_berger.pdf')
    plt.close()
plt.show()

'''
fig = plt.figure(figsize=(16,7))
plot = GridSpec(5,3,left=0.08,bottom=0.15,right=0.97,top=0.95,wspace=0,hspace=0)

x_min, x_max = 10**-4.8, 10**-2.7
y_min, y_max = 0, 2500

# CDPP vs. bprp
ax = plt.subplot(plot[0,0]) # histograms
x_all = stars_cleaned['rrmscdpp04p5']/1e6
#x_min, x_max = np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=16)

ax = plt.subplot(plot[1:,0]) # scatter contours
corner.hist2d(np.log10(stars_cleaned['rrmscdpp04p5']/1e6), bprp_corr, bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']})
plt.scatter(np.log10(ssk_per_sys['cdpp4p5_obs']/1e6), ssk_per_sys['bp_rp_obs'] - ssk_per_sys['e_bp_rp_obs'], s=1, marker='o', c='tab:orange', label='Planet hosting stars')
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks([-5,-4.5,-4,-3.5,-3])
plt.xlim([np.log10(x_min), np.log10(x_max)])
plt.ylim([np.min(bprp_corr), np.max(bprp_corr)])
plt.xlabel(r'$\log_{10}{(\sigma_{\rm CDPP, 4.5hr})}$', fontsize=20)
plt.ylabel(r'$b_p - r_p - E^*$ (mag)', fontsize=20)
#plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=16)
'''
