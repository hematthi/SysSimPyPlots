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
savefigures_directory = ''





##### To load the files with the systems with observed planets:

# To first read the number of simulated targets and bounds for the periods and radii:
P_min, P_max, radii_min, radii_max = 3., 300., 0.5, 10.

# To load and process the observed Kepler catalog and compare with our simulated catalog:

stars_cleaned = load_Kepler_stars_cleaned()
Rstar_med = np.nanmedian(stars_cleaned['radius'])
Mstar_med = np.nanmedian(stars_cleaned['mass'])
teff_med = np.nanmedian(stars_cleaned['teff'])
bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])

bins, i_blue_per_bin, i_red_per_bin = split_colors_per_cdpp_bin(stars_cleaned, nbins=10)

stellar_prop = 'bprp_cdpp4p5'

ssk_per_sys0, ssk0 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max)
ssk_per_sys1, ssk1 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, i_stars_custom=i_blue_per_bin) #_max=_med
ssk_per_sys2, ssk2 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, i_stars_custom=i_red_per_bin) #_min=_med

#label1, label2 = '$R_\star < %s R_\odot$' % np.round(Rstar_med, 3), '$R_\star > %s R_\odot$' % np.round(Rstar_med, 3)
#label1, label2 = '$M_\star < %s M_\odot$' % np.round(Mstar_med, 3), '$M_\star > %s M_\odot$' % np.round(Mstar_med, 3)
#label1, label2 = r'$T_{\rm eff} < %s K$' % np.round(teff_med, 3), r'$T_{\rm eff} > %s K$' % np.round(teff_med, 3)
#label1, label2 = '$b_p - r_p < %s$ mag' % np.round(bp_rp_med, 3), '$b_p - r_p > %s$ mag' % np.round(bp_rp_med, 3)
label1, label2 = 'Bluer half CDPP bins', 'Redder half CDPP bins'

split_ssk = [ssk1, ssk2]
split_ssk_per_sys = [ssk_per_sys1, ssk_per_sys2]
split_names = [label1, label2]
split_linestyles = ['-', '--']
split_colors = ['b', 'r']





#'''
##### To plot the Kepler catalog as marginal distributions:

subdirectory = 'Stellar_samples/Kepler_data/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 1 #linewidth
#alpha = 0.2 #transparency of histograms

afs = 16 #axes labels font size
tfs = 16 #text labels font size
lfs = 12 #legend labels font size





fig = plt.figure(figsize=(16,8))
plot = GridSpec(4,3,left=0.075,bottom=0.1,right=0.975,top=0.95,wspace=0.3,hspace=0.6)

# Multiplicities:
ax = plt.subplot(plot[0,0])
plot_panel_counts_hist_simple(ax, [], [ssk_per_sys['Mtot_obs'] for ssk_per_sys in split_ssk_per_sys], x_min=0, y_max=5e4, x_llim=0.5, log_y=True, c_Kep=split_colors, ms_Kep=['x','x'], lines_Kep=True, lw=lw, labels_Kep=split_names, xlabel_text='Observed planets per system', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_Kep=True)

# Periods:
ax = plt.subplot(plot[0,1])
plot_panel_pdf_simple(ax, [], [ssk['P_obs'] for ssk in split_ssk], x_min=P_min, x_max=P_max, n_bins=n_bins, normalize=False, log_x=True, log_y=True, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True)

# Period ratios (all, with some upper cut-off):
ax = plt.subplot(plot[0,2])
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_panel_pdf_simple(ax, [], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut] for ssk in split_ssk], x_min=1., x_max=R_max_cut, n_bins=n_bins, normalize=False, log_x=True, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Transit durations:
ax = plt.subplot(plot[1,1])
plot_panel_pdf_simple(ax, [], [ssk['tdur_tcirc_obs'] for ssk in split_ssk], x_min=0., x_max=1.5, n_bins=n_bins, normalize=False, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs) #r'$t_{\rm dur}$ (mins)'

# Log(xi):
ax = plt.subplot(plot[1,2])
plot_panel_pdf_simple(ax, [], [np.log10(ssk['xi_obs']) for ssk in split_ssk], x_min=-0.5, x_max=0.5, n_bins=n_bins, normalize=False, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$\log{\xi}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Transit depths:
ax = plt.subplot(plot[2,1])
plot_panel_pdf_simple(ax, [], [ssk['D_obs'] for ssk in split_ssk], x_min=1e-5, x_max=10.**-1.5, n_bins=n_bins, normalize=False, log_x=True, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$\delta$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Transit depth ratios:
ax = plt.subplot(plot[2,2])
plot_panel_pdf_simple(ax, [], [ssk['D_ratio_obs'] for ssk in split_ssk], x_min=10.**-1.5, x_max=10.**1.5, n_bins=n_bins, normalize=False, log_x=True, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Planet radii:
ax = plt.subplot(plot[3,1])
plot_panel_pdf_simple(ax, [], [ssk['radii_obs'] for ssk in split_ssk], x_min=radii_min, x_max=radii_max, n_bins=n_bins, normalize=False, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$R_p (R_\oplus)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Stellar radii:
ax = plt.subplot(plot[1,0])
plot_panel_pdf_simple(ax, [], [ssk['Rstar_obs'] for ssk in split_ssk], n_bins=n_bins, normalize=False, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$R_\star (R_\odot)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Stellar masses:
ax = plt.subplot(plot[2,0])
plot_panel_pdf_simple(ax, [], [ssk['Mstar_obs'] for ssk in split_ssk], n_bins=n_bins, normalize=False, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$M_\star (M_\odot)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Stellar teff:
ax = plt.subplot(plot[3,0])
plot_panel_pdf_simple(ax, [], [ssk['teff_obs'] for ssk in split_ssk], n_bins=n_bins, normalize=False, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$t_{\rm eff} (K)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Stellar colors:
ax = plt.subplot(plot[3,2])
plot_panel_pdf_simple(ax, [], [ssk['bp_rp_obs'] for ssk in split_ssk], n_bins=n_bins, normalize=False, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xlabel_text=r'$b_p - r_p$ (mag)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

# Periods of inner-most planet:
#ax = plt.subplot(plot[3,2])
#plot_panel_pdf_simple(ax, [], [ssk_per_sys['P_obs'][:,0] for ssk_per_sys in split_ssk_per_sys], x_min=P_min, x_max=P_max, n_bins=n_bins, normalize=False, log_x=True, log_y=True, c_Kep=split_colors, ls_Kep=split_linestyles, lw=lw, labels_Kep=split_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P_1$ (days)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_high_low_%s_summary.pdf' % stellar_prop)
else:
    plt.show()
#'''





'''
##### To plot the stellar properties of the planet-hosting stars, with our divisions:

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['radius'], stars_cleaned['teff'], s=1, marker='.', label='All stars')
plt.scatter(ssk_per_sys0['Rstar_obs'], ssk_per_sys0['teff_obs'], s=1, label='Planet hosting stars')
plt.axvline(x=Rstar_med)
plt.axhline(y=teff_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$R_\star (R_\odot)$', fontsize=20)
plt.ylabel(r'$T_{\rm eff} (K)$', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Rstar_teff.png')
else:
    plt.show()

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['mass'], stars_cleaned['teff'], s=1, marker='.', label='All stars')
plt.scatter(ssk_per_sys0['Mstar_obs'], ssk_per_sys0['teff_obs'], s=1, label='Planet hosting stars')
plt.axvline(x=Mstar_med)
plt.axhline(y=teff_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$M_\star (M_\odot)$', fontsize=20)
plt.ylabel(r'$T_{\rm eff} (K)$', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Mstar_teff.png')
else:
    plt.show()

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['radius'], stars_cleaned['mass'], s=1, marker='.', label='All stars')
plt.scatter(ssk_per_sys0['Rstar_obs'], ssk_per_sys0['Mstar_obs'], s=1, label='Planet hosting stars')
plt.axvline(x=Rstar_med)
plt.axhline(y=Mstar_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$R_\star (R_\odot)$', fontsize=20)
plt.ylabel(r'$M_\star (M_\odot)$', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Rstar_Mstar.png')
else:
    plt.show()



fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['radius'], stars_cleaned['bp_rp'], s=1, marker='.', label='All stars')
plt.scatter(ssk0['Rstar_obs'], ssk0['bp_rp_obs'], s=1, label='Planet hosting stars')
plt.axvline(x=Rstar_med)
plt.axhline(y=bp_rp_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$R_\star (R_\odot)$', fontsize=20)
plt.ylabel(r'$b_p - r_p$ (mag)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Rstar_bprp.png')
else:
    plt.show()

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['mass'], stars_cleaned['bp_rp'], s=1, marker='.', label='All stars')
plt.scatter(ssk0['Mstar_obs'], ssk0['bp_rp_obs'], s=1, label='Planet hosting stars')
plt.axvline(x=Mstar_med)
plt.axhline(y=bp_rp_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$M_\star (M_\odot)$', fontsize=20)
plt.ylabel(r'$b_p - r_p$ (mag)', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Mstar_bprp.png')
else:
    plt.show()

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['teff'], stars_cleaned['bp_rp'], s=1, marker='.', label='All stars')
plt.scatter(ssk0['teff_obs'], ssk0['bp_rp_obs'], s=1, label='Planet hosting stars')
plt.axvline(x=teff_med)
plt.axhline(y=bp_rp_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$T_{\rm eff} (K)$', fontsize=20)
plt.ylabel(r'$b_p - r_p$ (mag)', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_teff_bprp.png')
else:
    plt.show()
'''
