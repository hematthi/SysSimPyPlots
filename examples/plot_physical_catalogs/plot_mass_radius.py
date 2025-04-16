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

from mass_radius_models import *





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/'





##### To load the underlying populations:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #Lognormal_mass_Earthlike_rocky/
run_number = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)





##### To make mass-radius plots:

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

bins = 100



# Density vs. radius for new model based on Li Zeng's Earth-like rocky:

fig = plt.figure(figsize=(8,8))
plot = GridSpec(4, 1, left=0.15, bottom=0.1, right=0.98, top=0.98, wspace=0, hspace=0)

ax = plt.subplot(plot[0,:]) # sigma_rho vs. radius
plt.plot(MR_earthlike_rocky['radius'], sigma_rho, color='orange', ls='-', lw=3, label=r'Linear $\log(\sigma_\rho)$ vs $R_p$')
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xticks([])
plt.yticks([1., 2., 3., 4., 5.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)
plt.xlim([radii_min, radii_switch])
plt.ylim([0.9, 4.])
plt.ylabel(r'$\sigma_\rho$ ($g/cm^3$)', fontsize=tfs)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,:]) # rho vs. radius
plt.plot(MR_pure_iron['radius'], rho_pure_iron, color='r', ls='--', lw=3, label='Pure iron')
plt.plot(MR_earthlike_rocky['radius'], rho_earthlike_rocky, color='orange', ls='--', lw=3, label='Earth-like rocky')
plt.fill_between(MR_earthlike_rocky['radius'], rho_earthlike_rocky - sigma_rho, rho_earthlike_rocky + sigma_rho, color='orange', alpha=0.5, label=r'Earth-like rocky $\pm \sigma_\rho$')
plt.fill_between(MR_earthlike_rocky['radius'], rho_earthlike_rocky - 2.*sigma_rho, rho_earthlike_rocky + 2.*sigma_rho, color='orange', alpha=0.3, label=r'Earth-like rocky $\pm 2\sigma_\rho$')
plt.fill_between(MR_earthlike_rocky['radius'], rho_earthlike_rocky - 3.*sigma_rho, rho_earthlike_rocky + 3.*sigma_rho, color='orange', alpha=0.1, label=r'Earth-like rocky $\pm 3\sigma_\rho$')
plt.axhline(y=1., color='c', lw=3, label='Water density (1 g/cm^3)')
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.minorticks_off()
plt.yticks([1., 2., 3., 4., 5., 7., 10., 15.])
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)
plt.xlim([radii_min, radii_switch])
plt.ylim([0.9, 20.])
plt.xlabel(r'$R_p$ ($R_\oplus$)', fontsize=tfs)
plt.ylabel(r'$\rho$ ($g/cm^3$)', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Density_radius.pdf')
    plt.close()
plt.show()



# Mass vs. radius:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(5, 5, left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4])
masses_all = sssp_per_sys['mass_all'][sssp_per_sys['mass_all'] > 0.]
radii_all = sssp_per_sys['radii_all'][sssp_per_sys['radii_all'] > 0.]
corner.hist2d(np.log10(radii_all), np.log10(masses_all), bins=50, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.plot(MR_table['log_R'], MR_table['05'], '-', color='g', label='Mean prediction (NWG2018)')
plt.fill_between(MR_table['log_R'], MR_table['016'], MR_table['084'], color='g', alpha=0.5, label=r'16%-84% (NWG2018)')

plt.plot(MR_table['log_R'], np.log10(M_from_R_rho(10.**MR_table['log_R'], rho=5.51)), color='b', label='Earth density (5.51 g/cm^3)')
plt.plot(MR_table['log_R'], np.log10(M_from_R_rho(10.**MR_table['log_R'], rho=3.9)), color='m', label='Mars density (3.9 g/cm^3)')
plt.plot(MR_table['log_R'], np.log10(M_from_R_rho(10.**MR_table['log_R'], rho=1.)), color='c', label='Water density (1 g/cm^3)')
plt.plot(MR_table['log_R'], np.log10(M_from_R_rho(10.**MR_table['log_R'], rho=7.9)), color='r', label='Iron density (7.9 g/cm^3)')
plt.plot(MR_table['log_R'], np.log10(M_from_R_rho(10.**MR_table['log_R'], rho=100.)), color='k', label='100 g/cm^3')

plt.plot(np.log10(MR_earthlike_rocky['radius']), np.log10(MR_earthlike_rocky['mass']), color='orange', ls='--', lw=3, label='Earth-like rocky')
#plt.fill_between(np.log10(MR_earthlike_rocky['radius']), np.log10(M_from_R_rho(MR_earthlike_rocky['radius'], rho=rho_earthlike_rocky-sigma_rho)), np.log10(M_from_R_rho(MR_earthlike_rocky['radius'], rho=rho_earthlike_rocky+sigma_rho)), color='orange', alpha=0.5, label=r'16%-84% ($\rho \sim \mathcal{N}(\rho_{\rm Earthlike\:rocky}, \sigma_\rho(R_p))$)') #label=r'$\rho \sim \mathcal{N}(\rho_{\rm Earthlike\:rocky}, 10^{[\frac{d\log\rho}{dR_p}(R_p - 0.5) + \log{\rho_0}]})$'
plt.fill_between(np.log10(MR_earthlike_rocky['radius']), np.log10(MR_earthlike_rocky['mass']) - sigma_log_M, np.log10(MR_earthlike_rocky['mass']) + sigma_log_M, color='orange', alpha=0.5, label=r'16%-84% ($\log{M_p} \sim \mathcal{N}(M_{p,\rm Earthlike\:rocky}, \sigma_{\log{M_p}})$)')
plt.plot(np.log10(MR_pure_iron['radius']), np.log10(MR_pure_iron['mass']), color='r', ls='--', lw=3, label='Pure iron')
#plt.axvline(x=np.log10(0.7), color='k', ls='--', lw=3)
plt.axvline(x=np.log10(radii_switch), color='k', ls='--', lw=3)
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([0.5, 1., 2., 4., 10.])
ytick_vals = np.array([1e-1, 1., 10., 1e2])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(radii_min), np.log10(radii_max)])
plt.ylim([np.log10(0.07), 2.])
plt.xlabel(r'$R_p$ ($R_\oplus$)', fontsize=tfs)
plt.ylabel(r'$M_p$ ($M_\oplus$)', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist(radii_all, bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), histtype='step', color='k', ls='-', label=r'All')
#plt.axvline(x=0.7, color='k', ls='--', lw=3)
plt.axvline(x=radii_switch, color='k', ls='--', lw=3)
plt.gca().set_xscale("log")
plt.xlim([radii_min, radii_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(masses_all, bins=np.logspace(np.log10(0.07), 2., bins+1), histtype='step', orientation='horizontal', color='k', ls='-', label='All')
radii_cut = radii_switch
plt.hist(masses_all[radii_all > radii_cut], bins=np.logspace(np.log10(0.07), 2., bins+1), histtype='step', orientation='horizontal', color='b', ls='-', label=r'$R_p > %s R_\oplus$' % radii_cut)
plt.hist(masses_all[radii_all < radii_cut], bins=np.logspace(np.log10(0.07), 2., bins+1), histtype='step', orientation='horizontal', color='r', ls='-', label=r'$R_p < %s R_\oplus$' % radii_cut)
plt.gca().set_yscale("log")
plt.ylim([0.07, 1e2])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + 'MR_diagram.pdf')
    plt.close()
plt.show()
