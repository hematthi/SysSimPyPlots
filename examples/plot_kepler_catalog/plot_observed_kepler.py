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
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios_adjacent)





#'''
##### To plot the Kepler catalog as marginal distributions:

subdirectory = 'Stellar_samples/Kepler_data/E_BP_MIN_RP/Interpolate_e_bp_min_rp/e_bp_min_rp_dist/' #'Paper_Figures/'; 'Talk_Figures/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 1 #linewidth
#alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size



# Multiplicities:
plot_fig_counts_hist_simple(fig_size, [], [ssk_per_sys['Mtot_obs']], x_min=0, y_max=1e4, x_llim=0.5, log_y=True, lines_Kep=True, lw=lw, xlabel_text='Observed planets per system', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_Kep=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_multiplicities.pdf', save_fig=savefigures)

# Periods:
plot_fig_pdf_simple(fig_size, [], [ssk['P_obs']], x_min=P_min, x_max=P_max, n_bins=n_bins, normalize=False, log_x=True, log_y=True, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_periods.pdf', save_fig=savefigures)

# Period ratios (all, with some upper cut-off):
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple(fig_size, [], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, normalize=False, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_periodratios.pdf', save_fig=savefigures)

# Period ratios (< 5):
plot_fig_pdf_simple(fig_size, [], [ssk['Rm_obs'][ssk['Rm_obs'] < 5.]], x_min=1., x_max=5., n_bins=n_bins, normalize=False, lw=lw, xlabel_text=r'$P_{i+1}/P_i$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_periodratios_less5.pdf', save_fig=savefigures)

# Transit durations:
plot_fig_pdf_simple(fig_size, [], [ssk['tdur_obs']], x_max=15., n_bins=n_bins, normalize=False, lw=lw, xlabel_text=r'$t_{\rm dur}$ (hrs)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_durations.pdf', save_fig=savefigures)

# Transit durations (for singles and multis):
plot_fig_pdf_simple(fig_size, [], [ssk['tdur_tcirc_1_obs'], ssk['tdur_tcirc_2p_obs']], x_max=1.5, n_bins=n_bins, normalize=False, c_Kep=['b','r'], ls_Kep=['-','-'], lw=lw, labels_Kep=['Singles', 'Multis'], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_durations_norm_circ_singles_multis.pdf', save_fig=savefigures)

# Transit depths:
plot_fig_pdf_simple(fig_size, [], [ssk['D_obs']], x_min=1e-5, x_max=10.**-1.5, n_bins=n_bins, normalize=False, log_x=True, lw=lw, xlabel_text=r'$\delta$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_depths.pdf', save_fig=savefigures)

# Transit depths (above and below the photoevaporation boundary):
plot_fig_pdf_simple(fig_size, [], [ssk['D_above_obs'], ssk['D_below_obs']], x_min=1e-5, x_max=10.**-1.5, n_bins=n_bins, normalize=False, log_x=True, c_Kep=['b','r'], ls_Kep=['-','-'], lw=lw, labels_Kep=['Above', 'Below'], xlabel_text=r'$\delta$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_depths_photoevap.pdf', save_fig=savefigures)

# Planet radii:
plot_fig_pdf_simple(fig_size, [], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, n_bins=n_bins, normalize=False, lw=lw, xlabel_text=r'$R_p (R_\oplus)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_radii.pdf', save_fig=savefigures)

# Stellar radii:
plot_fig_pdf_simple(fig_size, [], [ssk['Rstar_obs']], n_bins=n_bins, normalize=False, lw=lw, xlabel_text=r'$R_\star (R_\odot)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_stellar_radii.pdf', save_fig=savefigures)

# Transit depth ratios:
plot_fig_pdf_simple(fig_size, [], [ssk['D_ratio_obs']], x_min=10.**-1.5, x_max=10.**1.5, n_bins=n_bins, normalize=False, log_x=True, lw=lw, xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_depthratios.pdf', save_fig=savefigures)

# Transit depth ratios (above, below, and across the photoevaporation boundary):
plot_fig_pdf_simple(fig_size, [], [ssk['D_ratio_above_obs'], ssk['D_ratio_below_obs'], ssk['D_ratio_across_obs']], x_min=10.**-1.5, x_max=10.**1.5, n_bins=n_bins, normalize=False, log_x=True, c_Kep=['b','r','k'], ls_Kep=['-','-','-'], lw=lw, labels_Kep=['Above', 'Below', 'Across'], xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_depthratios_photoevap.pdf', save_fig=savefigures)

# Log(xi):
plot_fig_pdf_simple(fig_size, [], [np.log10(ssk['xi_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, normalize=False, lw=lw, xlabel_text=r'$\log{\xi}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_logxi_all.pdf', save_fig=savefigures)

# Log(xi) by res/non-res:
plot_fig_pdf_simple(fig_size, [], [np.log10(ssk['xi_res_obs']), np.log10(ssk['xi_nonres_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, normalize=False, c_Kep=['m','g'], ls_Kep=['-','-'], lw=lw, labels_Kep=['Near MMR', 'Not near MMR'], xlabel_text=r'$\log{\xi}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_logxi.pdf', save_fig=savefigures)

# Log(xi) within res:
plot_fig_pdf_simple(fig_size, [], [np.log10(ssk['xi_res32_obs']), np.log10(ssk['xi_res21_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, normalize=False, c_Kep=['r','b'], ls_Kep=['-','-'], lw=lw, labels_Kep=['Near 3:2 MMR', 'Near 2:1 MMR'], xlabel_text=r'$\log{\xi}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_logxi_res.pdf', save_fig=savefigures)

plt.show()
#plt.close()
#'''





##### To plot a histogram of the zeta statistic:

zeta1_all = zeta1(ssk['Rm_obs'])
pratios_small = ssk['Rm_obs'][ssk['Rm_obs'] < 2.5]
zeta1_small = zeta1(pratios_small)
plot_fig_pdf_simple(fig_size, [], [zeta1_all, zeta1_small], x_min=-1., x_max=1., n_bins=30, normalize=True, c_Kep=['b','r'], ls_Kep=['-','-'], lw=lw, labels_Kep=['All period ratios', 'Period ratios < 2.5'], xlabel_text=r'$\zeta_1$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_zeta1.pdf', save_fig=savefigures)
#plot_fig_pdf_simple(fig_size, [], [zeta1_all], x_min=-1., x_max=1., n_bins=30, normalize=True, c_Kep=['b'], lw=lw, labels_Kep=['All period ratios'], xlabel_text=r'$\zeta_1$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_zeta1_all.pdf', save_fig=savefigures)
#plot_fig_pdf_simple(fig_size, [], [zeta1_small], x_min=-1., x_max=1., n_bins=30, normalize=True, c_Kep=['r'], lw=lw, labels_Kep=['Period ratios < 2.5'], xlabel_text=r'$\zeta_1$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_zeta1_small.pdf', save_fig=savefigures)
plt.show()





##### To make various plots involving CDPP (4.5hr):
'''
# Histograms of CDPP (all stars and planet hosting stars):
plot_fig_pdf_simple((10,4), [stars_cleaned['rrmscdpp04p5']], [ssk_per_sys['cdpp4p5_obs']], log_x=True, n_bins=100, normalize=True, lw=lw, labels_sim=['All stars'], labels_Kep=['Planet hosting stars'], xlabel_text=r'$\sigma_{\rm CDPP, 4.5hr}$ (ppm)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_cdpp4p5.pdf', save_fig=savefigures)

# CDPP vs. bp-rp color:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['rrmscdpp04p5'], stars_cleaned['bp_rp'], s=1, marker='.', label='All stars')
plt.scatter(ssk_per_sys['cdpp4p5_obs'], ssk_per_sys['bp_rp_obs'], s=1, marker='.', label='Planet hosting stars')
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\sigma_{\rm CDPP, 4.5hr}$ (ppm)', fontsize=20)
plt.ylabel(r'$b_p - r_p$ (mag)', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_cdpp4p5_bprp.pdf')
else:
    plt.show()

# Rstar*sqrt(CDPP) vs. bp-rp color:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['radius']*np.sqrt(stars_cleaned['rrmscdpp04p5']/1e6), stars_cleaned['bp_rp'], s=1, marker='.', label='All stars')
plt.scatter(ssk_per_sys['Rstar_obs']*np.sqrt(ssk_per_sys['cdpp4p5_obs']/1e6), ssk_per_sys['bp_rp_obs'], s=1, marker='.', label='Planet hosting stars')
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$R_\star \sqrt{\sigma_{\rm CDPP, 4.5hr}}$ ($R_\odot$)', fontsize=20)
plt.ylabel(r'$b_p - r_p$ (mag)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Rstar_root_cdpp4p5_bprp.pdf')
else:
    plt.show()

# CDPP vs. transit depths:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
plt.scatter(ssk['cdpp4p5_obs'], 1e6*ssk['D_obs'], s=5, marker='.', label='Planet hosting stars')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\sigma_{\rm CDPP, 4.5hr}$ (ppm)', fontsize=20)
plt.ylabel(r'$\delta$ (ppm)', fontsize=20)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_cdpp4p5_depths.pdf')
else:
    plt.show()

# Histogram of CDPP with 10 bins, dividing each bin into red and blue halves based on bp-rp color:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
nbins = 10
bins, i_blue_per_bin, i_red_per_bin = split_colors_per_cdpp_bin(stars_cleaned, nbins=nbins)
plt.hist(stars_cleaned['rrmscdpp04p5'], bins=bins, alpha=0.2, label='All stars')
plt.hist([stars_cleaned['rrmscdpp04p5'][i_blue_per_bin], stars_cleaned['rrmscdpp04p5'][i_red_per_bin]], bins=bins, histtype='step', color=['b','r'], stacked=True, label=['Bluer half', 'Redder half'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\sigma_{\rm CDPP, 4.5hr}$ (ppm)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=True, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_cdpp4p5_split_bprp.pdf')
else:
    plt.show()
'''





##### To plot R_star*CDPP vs. bprp, with histograms of R_star*CDPP on top for the bluer and redder halves:
'''
fig = plt.figure(figsize=(8,8))

plot = GridSpec(1,1,left=0.15,bottom=0.85,right=0.95,top=0.98,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
x_all = stars_cleaned['radius']*stars_cleaned['rrmscdpp04p5']/1e6
x_min, x_max = 1e-5, 1e-3 #np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[stars_cleaned['bp_rp'] < 0.95], x_all[stars_cleaned['bp_rp'] > 0.95]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.xlim([x_min, x_max])
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=16)

plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.85,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['radius']*stars_cleaned['rrmscdpp04p5']/1e6, stars_cleaned['bp_rp'], s=1, marker='.', label='All stars')
plt.scatter(ssk_per_sys['Rstar_obs']*ssk_per_sys['cdpp4p5_obs']/1e6, ssk_per_sys['bp_rp_obs'], s=1, marker='.', label='Planet hosting stars')
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlim([x_min, x_max])
plt.ylim([np.min(stars_cleaned['bp_rp']), np.max(stars_cleaned['bp_rp'])])
plt.xlabel(r'$R_\star \sigma_{\rm CDPP, 4.5hr}$ ($R_\odot$)', fontsize=20)
plt.ylabel(r'$b_p - r_p$ (mag)', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=True, fontsize=16)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Rstar_cdpp4p5_bprp.pdf')
    plt.close()
plt.show()

##### To plot the above (R_star*CDPP vs. bprp, with histograms of R_star*CDPP on top) along with CDPP vs. bprp (also with histograms on top) for the bluer and redder halves:

fig = plt.figure(figsize=(16,8))

# CDPP vs. bprp
plot = GridSpec(1,1,left=0.08,bottom=0.85,right=0.5,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
x_all = stars_cleaned['rrmscdpp04p5']/1e6
x_min, x_max = 1e-5, 1e-3 #np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[stars_cleaned['bp_rp'] < 0.95], x_all[stars_cleaned['bp_rp'] > 0.95]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.xlim([x_min, x_max])
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=16)

plot = GridSpec(1,1,left=0.08,bottom=0.1,right=0.5,top=0.85,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['rrmscdpp04p5']/1e6, stars_cleaned['bp_rp'], s=1, marker='.', label='All stars')
plt.scatter(ssk_per_sys['cdpp4p5_obs']/1e6, ssk_per_sys['bp_rp_obs'], s=1, marker='.', label='Planet hosting stars')
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlim([x_min, x_max])
plt.ylim([np.min(stars_cleaned['bp_rp']), np.max(stars_cleaned['bp_rp'])])
plt.xlabel(r'$\sigma_{\rm CDPP, 4.5hr}$', fontsize=20)
plt.ylabel(r'$b_p - r_p$ (mag)', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=True, fontsize=16)

# R_star*CDPP vs. bprp
plot = GridSpec(1,1,left=0.55,bottom=0.85,right=0.97,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
x_all = stars_cleaned['radius']*stars_cleaned['rrmscdpp04p5']/1e6
x_min, x_max = 1e-5, 1e-3 #np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[stars_cleaned['bp_rp'] < 0.95], x_all[stars_cleaned['bp_rp'] > 0.95]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.xlim([x_min, x_max])

plot = GridSpec(1,1,left=0.55,bottom=0.1,right=0.97,top=0.85,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(stars_cleaned['radius']*stars_cleaned['rrmscdpp04p5']/1e6, stars_cleaned['bp_rp'], s=1, marker='.', label='All stars')
plt.scatter(ssk_per_sys['Rstar_obs']*ssk_per_sys['cdpp4p5_obs']/1e6, ssk_per_sys['bp_rp_obs'], s=1, marker='.', label='Planet hosting stars')
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlim([x_min, x_max])
plt.ylim([np.min(stars_cleaned['bp_rp']), np.max(stars_cleaned['bp_rp'])])
plt.xlabel(r'$R_\star \sigma_{\rm CDPP, 4.5hr}$ ($R_\odot$)', fontsize=20)
plt.ylabel('')

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Rstar_cdpp4p5_bprp_two.pdf')
    plt.close()
plt.show()
'''

##### To plot the above but with contours (2d histograms using corner.hist2d) instead of all the points:

fig = plt.figure(figsize=(16,8))

# CDPP vs. bprp
plot = GridSpec(1,1,left=0.08,bottom=0.85,right=0.5,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
x_all = stars_cleaned['rrmscdpp04p5']/1e6
x_min, x_max = 1e-5, 1e-3 #np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[stars_cleaned['bp_rp'] < 0.95], x_all[stars_cleaned['bp_rp'] > 0.95]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.xlim([x_min, x_max])
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=16)

plot = GridSpec(1,1,left=0.08,bottom=0.1,right=0.5,top=0.85,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
corner.hist2d(np.log10(stars_cleaned['rrmscdpp04p5']/1e6), stars_cleaned['bp_rp'], bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']})
plt.scatter(np.log10(ssk_per_sys['cdpp4p5_obs']/1e6), ssk_per_sys['bp_rp_obs'], s=1, marker='o', c='tab:orange', label='Planet hosting stars')
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks([-5,-4.5,-4,-3.5,-3])
plt.xlim([np.log10(x_min), np.log10(x_max)])
plt.ylim([np.min(stars_cleaned['bp_rp']), np.max(stars_cleaned['bp_rp'])])
plt.xlabel(r'$\log_{10}{(\sigma_{\rm CDPP, 4.5hr})}$', fontsize=20)
plt.ylabel(r'$b_p - r_p$ (mag)', fontsize=20)
#plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=16)

# R_star*CDPP vs. bprp
plot = GridSpec(1,1,left=0.55,bottom=0.85,right=0.97,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
x_all = stars_cleaned['radius']*stars_cleaned['rrmscdpp04p5']/1e6
x_min, x_max = 1e-5, 1e-3 #np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[stars_cleaned['bp_rp'] < 0.95], x_all[stars_cleaned['bp_rp'] > 0.95]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.xlim([x_min, x_max])

plot = GridSpec(1,1,left=0.55,bottom=0.1,right=0.97,top=0.85,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
corner.hist2d(np.log10(stars_cleaned['radius']*stars_cleaned['rrmscdpp04p5']/1e6), stars_cleaned['bp_rp'], bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']})
plt.scatter(np.log10(ssk_per_sys['Rstar_obs']*ssk_per_sys['cdpp4p5_obs']/1e6), ssk_per_sys['bp_rp_obs'], s=1, marker='o', c='tab:orange', label='Planet hosting stars')
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks([-5,-4.5,-4,-3.5,-3])
plt.xlim([np.log10(x_min), np.log10(x_max)])
plt.ylim([np.min(stars_cleaned['bp_rp']), np.max(stars_cleaned['bp_rp'])])
plt.xlabel(r'$\log_{10}{((R_\star/R_\odot)\sigma_{\rm CDPP, 4.5hr})}$', fontsize=20)
plt.ylabel('')

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Rstar_cdpp4p5_bprp_two_contours.pdf')
    plt.close()
plt.show()





##### RE-DO THE ABOVE BUT WITH CORRECTED COLORS:

bprp_corr = stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp']
bprp_corr_med = np.nanmedian(bprp_corr)

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

# R_star^2*CDPP vs. bprp
ax = plt.subplot(plot[0,1]) # histograms
x_all = (stars_cleaned['radius']**2.)*stars_cleaned['rrmscdpp04p5']/1e6
#x_min, x_max = np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.yticks([])
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])

ax = plt.subplot(plot[1:,1])
corner.hist2d(np.log10((stars_cleaned['radius']**2.)*stars_cleaned['rrmscdpp04p5']/1e6), bprp_corr, bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']})
plt.scatter(np.log10((ssk_per_sys['Rstar_obs']**2.)*ssk_per_sys['cdpp4p5_obs']/1e6), ssk_per_sys['bp_rp_obs'] - ssk_per_sys['e_bp_rp_obs'], s=1, marker='o', c='tab:orange', label='Planet hosting stars')
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks([-5,-4.5,-4,-3.5,-3])
plt.yticks([])
plt.xlim([np.log10(x_min), np.log10(x_max)])
plt.ylim([np.min(bprp_corr), np.max(bprp_corr)])
plt.xlabel(r'$\log_{10}{[(R_\star/R_\odot)^2 \sigma_{\rm CDPP, 4.5hr}]}$', fontsize=20)
plt.ylabel('')

# R_star^(2 + alpha_R1)*CDPP vs. bprp
'''
alpha_R1 = -1.4
ax = plt.subplot(plot[0,2]) # histograms
x_all = (stars_cleaned['radius']**(2.+alpha_R1))*stars_cleaned['rrmscdpp04p5']/1e6
#x_min, x_max = np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.yticks([])
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])

ax = plt.subplot(plot[1:,2])
corner.hist2d(np.log10((stars_cleaned['radius']**(2.+alpha_R1))*stars_cleaned['rrmscdpp04p5']/1e6), bprp_corr, bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']})
plt.scatter(np.log10((ssk_per_sys['Rstar_obs']**(2.+alpha_R1))*ssk_per_sys['cdpp4p5_obs']/1e6), ssk_per_sys['bp_rp_obs'] - ssk_per_sys['e_bp_rp_obs'], s=1, marker='o', c='tab:orange', label='Planet hosting stars')
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks([-5,-4.5,-4,-3.5,-3])
plt.yticks([])
plt.xlim([np.log10(x_min), np.log10(x_max)])
plt.ylim([np.min(bprp_corr), np.max(bprp_corr)])
plt.xlabel(r'$\log_{10}{[(R_\star/R_\odot)^{%.1f} \sigma_{\rm CDPP, 4.5hr}]}$' % (2.+alpha_R1), fontsize=20)
plt.ylabel('')
'''

# R_star*CDPP vs. bprp
#'''
ax = plt.subplot(plot[0,2]) # histograms
x_all = (stars_cleaned['radius']**0.5)*stars_cleaned['rrmscdpp04p5']/1e6
#x_min, x_max = np.min(x_all), np.max(x_all)
bins = 100
plt.hist([x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], bins=np.logspace(np.log10(x_min), np.log10(x_max), bins+1), histtype='step', color=['b','r'], label=['Bluer', 'Redder'])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16, labelbottom=False)
plt.yticks([])
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])

ax = plt.subplot(plot[1:,2])
corner.hist2d(np.log10((stars_cleaned['radius']**0.5)*stars_cleaned['rrmscdpp04p5']/1e6), bprp_corr, bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']})
plt.scatter(np.log10((ssk_per_sys['Rstar_obs']**0.5)*ssk_per_sys['cdpp4p5_obs']/1e6), ssk_per_sys['bp_rp_obs'] - ssk_per_sys['e_bp_rp_obs'], s=1, marker='o', c='tab:orange', label='Planet hosting stars')
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks([-5,-4.5,-4,-3.5,-3])
plt.yticks([])
plt.xlim([np.log10(x_min), np.log10(x_max)])
plt.ylim([np.min(bprp_corr), np.max(bprp_corr)])
plt.xlabel(r'$\log_{10}{[(R_\star/R_\odot)^{0.5} \sigma_{\rm CDPP, 4.5hr}]}$', fontsize=20)
plt.ylabel('')
#'''

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'Kepler_Rstar_cdpp4p5_bprp_corrected_three_contours.pdf')
    plt.close()
plt.show()





##### Histograms only:

# CDPP:
x_all = stars_cleaned['rrmscdpp04p5']/1e6
plot_fig_pdf_simple(fig_size, [x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], [], x_min=1e-5, x_max=1e-3, n_bins=n_bins, normalize=False, log_x=True, c_sim=['b','r'], ls_sim=['-','-'], lw=lw, labels_sim=['Bluer', 'Redder'], xlabel_text=r'$\sigma_{\rm CDPP, 4.5hr}$', ylabel_text='Counts', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, legend=True, save_name=savefigures_directory + subdirectory + 'Kepler_cdpp4p5_split_hists.pdf', save_fig=savefigures)

# R_star:
x_all = stars_cleaned['radius']
plot_fig_pdf_simple(fig_size, [x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], [], x_min=0.4, x_max=2.5, n_bins=n_bins, normalize=False, log_x=False, c_sim=['b','r'], ls_sim=['-','-'], lw=lw, labels_sim=['Bluer', 'Redder'], xlabel_text=r'$R_\star$ $(R_\odot)$', ylabel_text='Counts', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_Rstar_split_hists.pdf', save_fig=savefigures)

# (R_star^2)*CDPP
x_all = (stars_cleaned['radius']**2.)*stars_cleaned['rrmscdpp04p5']/1e6
plot_fig_pdf_simple(fig_size, [x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], [], x_min=1e-5, x_max=3e-3, n_bins=n_bins, normalize=False, log_x=True, c_sim=['b','r'], ls_sim=['-','-'], lw=lw, labels_sim=['Bluer', 'Redder'], xlabel_text=r'$(R_\star/R_\odot)^2 \sigma_{\rm CDPP, 4.5hr}$', ylabel_text='Counts', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_Rstar_sq_cdpp4p5_split_hists.pdf', save_fig=savefigures)

# R_star*CDPP
x_all = stars_cleaned['radius']*stars_cleaned['rrmscdpp04p5']/1e6
plot_fig_pdf_simple(fig_size, [x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], [], x_min=1e-5, x_max=1e-3, n_bins=n_bins, normalize=False, log_x=True, c_sim=['b','r'], ls_sim=['-','-'], lw=lw, labels_sim=['Bluer', 'Redder'], xlabel_text=r'$(R_\star/R_\odot) \sigma_{\rm CDPP, 4.5hr}$', ylabel_text='Counts', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_Rstar_cdpp4p5_split_hists.pdf', save_fig=savefigures)

# (R_star^0.5)*CDPP
x_all = np.sqrt(stars_cleaned['radius'])*stars_cleaned['rrmscdpp04p5']/1e6
plot_fig_pdf_simple(fig_size, [x_all[bprp_corr < bprp_corr_med], x_all[bprp_corr > bprp_corr_med]], [], x_min=1e-5, x_max=1e-3, n_bins=n_bins, normalize=False, log_x=True, c_sim=['b','r'], ls_sim=['-','-'], lw=lw, labels_sim=['Bluer', 'Redder'], xlabel_text=r'$(R_\star/R_\odot)^{0.5} \sigma_{\rm CDPP, 4.5hr}$', ylabel_text='Counts', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + 'Kepler_Rstar_root_cdpp4p5_split_hists.pdf', save_fig=savefigures)





##### To plot a period-radius diagram showing the radius valley:

P_all = ssk['P_obs']
R_all = ssk['radii_obs']

P_l, P_u = 3., 300.
R_l, R_u = 1., 5.
P_plot = P_all[(P_all > P_l) & (P_all < P_u) & (R_all > R_l) & (R_all < R_u)]
R_plot = R_all[(P_all > P_l) & (P_all < P_u) & (R_all > R_l) & (R_all < R_u)]

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5,5,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[1:,:4])
corner.hist2d(np.log10(P_plot), np.log10(R_plot), bins=20, smooth=1., plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']})
plt.scatter(np.log10(P_plot), np.log10(R_plot), s=1, marker='o', c='k')
P_axis = np.linspace(np.log10(P_l), np.log10(P_u), 101)
plt.plot(P_axis, -0.09*P_axis + np.log10(2.4), '--', lw=2., c='r')
ax.tick_params(axis='both', labelsize=16)
xtick_vals = np.array([3,10,30,100])
ytick_vals = np.array([1,1.5,2,3,4])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlabel(r'Orbital Period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)
#plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=16)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist(P_plot, bins=np.logspace(np.log10(P_l), np.log10(P_u), 51), weights=np.ones(len(P_plot))/len(P_plot), histtype='step', color='k', ls='-')
plt.gca().set_xscale("log")
plt.xlim([P_l, P_u])
plt.xticks([])
plt.yticks([])
plt.minorticks_off()

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(R_plot, bins=np.logspace(np.log10(R_l), np.log10(R_u), 51), weights=np.ones(len(R_plot))/len(R_plot), histtype='step', orientation='horizontal', color='k', ls='-')
plt.gca().set_yscale("log")
plt.ylim([R_l, R_u])
plt.xticks([])
plt.yticks([])
plt.minorticks_off()

plt.show()
