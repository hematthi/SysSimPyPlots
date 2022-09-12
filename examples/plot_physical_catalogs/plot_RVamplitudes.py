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

from syssimpyplots.compute_RVs import *





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/RV/'
save_name = 'Maximum_AMD_model'





##### To load the underlying populations:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)





##### To compute RV semi-amplitudes:

# For multi-planet systems where at least one planet is in the period range:
det_inrange_all = []
P_inrange_all = []
Rp_inrange_all = []
Mp_inrange_all = []
K_K_max_inrange_all = []
K_max_or_2ndmax_all = [] # either K_max (when planet in range is not the max) or the 2nd K_max (when planet in range is the max); also includes repeats, for systems where more than one planet is in the range
P_K_max_or_2ndmax_all = [] # period of either the planet with K_max (when planet in range is not the max) or of the planet with the 2nd K_max (when planet in range is the max)

# For single systems where the planet is in the period range:
P_inrange_singles = []
Rp_inrange_singles = []
Mp_inrange_singles = []
K_max_singles = []

for i,P_sys in enumerate(sssp_per_sys['P_all']):
    det_sys = sssp_per_sys['det_all'][i]
    Mp_sys = sssp_per_sys['mass_all'][i][P_sys > 0]
    Rp_sys = sssp_per_sys['radii_all'][i][P_sys > 0]
    e_sys = sssp_per_sys['e_all'][i][P_sys > 0]
    incl_sys = sssp_per_sys['incl_all'][i][P_sys > 0]
    P_sys = P_sys[P_sys > 0]
    n_pl = len(P_sys)
    if any((P_sys > 5.) & (P_sys < 10.)):
        if n_pl == 1: # singles
            K_single = rv_K(Mp_sys, P_sys, Mstar=sssp['Mstar_all'][i])
            P_inrange_singles.append(P_sys[0])
            Rp_inrange_singles.append(Rp_sys[0])
            Mp_inrange_singles.append(Mp_sys[0])
            K_max_singles.append(K_single[0])
        else: # multi-planet systems
            j_inrange = np.arange(len(P_sys))[(P_sys > 5.) & (P_sys < 10.)]
            #print(i, ': planets in [5,10]d = ', len(j_inrange))

            #K_sys = rv_K(Mp_sys, P_sys, Mstar=sssp['Mstar_all'][i])
            K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=sssp['Mstar_all'][i])
            idsort_K_sys = np.argsort(K_sys)
            K_max, K_2ndmax = K_sys[idsort_K_sys[-1]], K_sys[idsort_K_sys[-2]]
            P_K_max, P_K_2ndmax = P_sys[idsort_K_sys[-1]], P_sys[idsort_K_sys[-2]]
            K_K_max_sys = K_sys/K_max
            for j in j_inrange:
                det_inrange_all.append(det_sys[j])
                P_inrange_all.append(P_sys[j])
                Rp_inrange_all.append(Rp_sys[j])
                Mp_inrange_all.append(Mp_sys[j])
                #K_max_all.append(K_max)
                if K_K_max_sys[j] == 1:
                    K_K_max_inrange_all.append(K_max/K_2ndmax)
                    K_max_or_2ndmax_all.append(K_2ndmax)
                    P_K_max_or_2ndmax_all.append(P_K_2ndmax)
                else:
                    K_K_max_inrange_all.append(K_K_max_sys[j])
                    K_max_or_2ndmax_all.append(K_max)
                    P_K_max_or_2ndmax_all.append(P_K_max)

det_inrange_all = np.array(det_inrange_all)
P_inrange_all = np.array(P_inrange_all)
Rp_inrange_all = np.array(Rp_inrange_all)
Mp_inrange_all = np.array(Mp_inrange_all)
K_K_max_inrange_all = np.array(K_K_max_inrange_all)
K_max_or_2ndmax_all = np.array(K_max_or_2ndmax_all)
P_K_max_or_2ndmax_all = np.array(P_K_max_or_2ndmax_all)

P_inrange_singles = np.array(P_inrange_singles)
Rp_inrange_singles = np.array(Rp_inrange_singles)
Mp_inrange_singles = np.array(Mp_inrange_singles)
K_max_singles = np.array(K_max_singles)





##### To plot RV semi-amplitudes:

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

bins = 100



# K/K_max vs. planet radii (all planets):

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5, 5, left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4])
corner.hist2d(np.log10(Rp_inrange_all), np.log10(K_K_max_inrange_all), bins=50, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.axhline(y=0., color='g', ls='--', lw=3)
#plt.scatter(np.log10(Rp_inrange_all[det_inrange_all == 1]), np.log10(K_K_max_inrange_all[det_inrange_all == 1]), marker='.', label='Observed planets')
#plt.scatter(Rp_inrange_all, K_K_max_inrange_all, marker='.')
#plt.scatter(Rp_inrange_singles, np.ones(len(Rp_inrange_singles)), marker='x')
#plt.gca().set_xscale("log")
#plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
#ax.set_xticks([0.5,1.,2.,4.,8.])
#ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
xtick_vals = np.array([0.5,1.,2.,4.,8.])
#ytick_vals = np.array([1e-3, 1e-2, 0.1, 1., 10., 1e2, 1e3])
plt.xticks(np.log10(xtick_vals), xtick_vals)
#plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(radii_min), np.log10(radii_max)]) #[radii_min, radii_max]
plt.ylim([-3., 3.]) #[1e-3, 1e3]
plt.xlabel(r'$R_p$ ($R_\oplus$)', fontsize=tfs)
plt.ylabel(r'$\log_{10}(K/K_{\rm max})$', fontsize=tfs)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist(Rp_inrange_all, bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), histtype='step', color='k', ls='-', label=r'All')
plt.hist(Rp_inrange_all[K_K_max_inrange_all > 1.], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), histtype='step', color='b', ls='-', label=r'$K/K_{\rm max} > 1$')
plt.hist(Rp_inrange_all[K_K_max_inrange_all < 1.], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), histtype='step', color='r', ls='-', label=r'$K/K_{\rm max} < 1$')
plt.gca().set_xscale("log")
plt.xlim([radii_min, radii_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(K_K_max_inrange_all, bins=np.logspace(-3., 3., bins+1), histtype='step', orientation='horizontal', color='k', ls='-')
plt.axhline(y=1., color='g', ls='--', lw=3)
plt.gca().set_yscale("log")
plt.ylim([1e-3, 1e3])
plt.xticks([])
plt.yticks([])
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_RVratio_all_5-10d_planet_radii.pdf')
    plt.close()



# K/K_max vs. planet mass (all planets):

mass_min, mass_max = 0.08, 1e2

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5, 5, left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4])
corner.hist2d(np.log10(Mp_inrange_all), np.log10(K_K_max_inrange_all), bins=50, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.axhline(y=0., color='g', ls='--', lw=3)
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([0.1, 1., 10., 1e2])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.xlim([np.log10(mass_min), np.log10(mass_max)]) #[radii_min, radii_max]
plt.ylim([-3., 3.]) #[1e-3, 1e3]
plt.xlabel(r'$M_p$ ($M_\oplus$)', fontsize=tfs)
plt.ylabel(r'$\log_{10}(K/K_{\rm max})$', fontsize=tfs)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist(Mp_inrange_all, bins=np.logspace(np.log10(mass_min), np.log10(mass_max), bins+1), histtype='step', color='k', ls='-', label=r'All')
plt.hist(Mp_inrange_all[K_K_max_inrange_all > 1.], bins=np.logspace(np.log10(mass_min), np.log10(mass_max), bins+1), histtype='step', color='b', ls='-', label=r'$K/K_{\rm max} > 1$')
plt.hist(Mp_inrange_all[K_K_max_inrange_all < 1.], bins=np.logspace(np.log10(mass_min), np.log10(mass_max), bins+1), histtype='step', color='r', ls='-', label=r'$K/K_{\rm max} < 1$')
plt.gca().set_xscale("log")
plt.xlim([mass_min, mass_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(K_K_max_inrange_all, bins=np.logspace(-3., 3., bins+1), histtype='step', orientation='horizontal', color='k', ls='-')
plt.axhline(y=1., color='g', ls='--', lw=3)
plt.gca().set_yscale("log")
plt.ylim([1e-3, 1e3])
plt.xticks([])
plt.yticks([])
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_RVratio_all_5-10d_planet_mass.pdf')
    plt.close()



# K_max/K vs. period ratio (all planets):

pr_min, pr_max = 3./10., 300./5.

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5, 5, left=0.18, bottom=0.13, right=0.98, top=0.98, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4])
x = 1./K_K_max_inrange_all
corner.hist2d(np.log10(P_K_max_or_2ndmax_all/P_inrange_all), np.log10(x), bins=50, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.scatter(np.log10(P_K_max_or_2ndmax_all/P_inrange_all)[det_inrange_all == 1], np.log10(x[det_inrange_all == 1]), marker='.', s=2, c='m')
plt.axhline(y=0., color='g', ls='--', lw=3)
plt.axvline(x=0., color='g', ls='--', lw=3)
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([0.5, 1., 2., 4., 8., 16., 32.])
ytick_vals = np.array([0.01, 0.1, 1., 10., 100.])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(pr_min), np.log10(pr_max)])
plt.ylim([-2., 2.]) #[1e-3, 1e3]
plt.xlabel(r'$P_{K_{\rm max\;or\;2nd\;max}}/P_{[5,10]d}$', fontsize=tfs)
plt.ylabel(r'$K_{\rm 2nd\;max}/K_{[5,10]d}$         $K_{\rm max}/K_{[5,10]d}$     ', fontsize=tfs)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist((P_K_max_or_2ndmax_all/P_inrange_all)[(x > 1.) & (det_inrange_all == 1)], bins=np.logspace(np.log10(pr_min), np.log10(pr_max), bins+1), histtype='step', color='b', ls='-', label=r'$\frac{K_{\rm max}}{K_{[5,10]d}} > 1$')
plt.hist((P_K_max_or_2ndmax_all/P_inrange_all)[(x < 1.) & (det_inrange_all == 1)], bins=np.logspace(np.log10(pr_min), np.log10(pr_max), bins+1), histtype='step', color='r', ls='-', label=r'$\frac{K_{\rm 2nd\;max}}{K_{[5,10]d}} < 1$')
plt.hist((P_K_max_or_2ndmax_all/P_inrange_all)[det_inrange_all == 1], bins=np.logspace(np.log10(pr_min), np.log10(pr_max), bins+1), histtype='step', color='m', ls='-', label='')
plt.axvline(x=1., color='g', ls='--', lw=3)
plt.gca().set_xscale("log")
plt.xlim([pr_min, pr_max])
plt.ylim([0,80])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(x, bins=np.logspace(-3., 3., bins+1), weights=np.ones(len(x))/len(x), histtype='step', orientation='horizontal', color='k', ls='-', label='All')
plt.hist(x[det_inrange_all == 1], bins=np.logspace(-3., 3., bins+1), weights=np.ones(np.sum(det_inrange_all == 1))/np.sum(det_inrange_all == 1), histtype='step', orientation='horizontal', color='m', ls='-', label=r'Observed $P_{[5,10]d}$')
plt.axhline(y=1., color='g', ls='--', lw=3)
plt.gca().set_yscale("log")
plt.ylim([1e-2, 1e2])
plt.xticks([])
plt.yticks([])
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_RVratio_all_5-10d_pratio_hists_obs.pdf')
    plt.close()





# K/K_max vs. planet radii (observed planets):
#'''
fig = plt.figure(figsize=(8,8))
plot = GridSpec(5, 5, left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4])
corner.hist2d(np.log10(Rp_inrange_all[det_inrange_all == 1]), np.log10(K_K_max_inrange_all[det_inrange_all == 1]), bins=20, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.axhline(y=0., color='g', ls='--', lw=3)
#plt.scatter(Rp_inrange_all[det_inrange_all == 1], K_K_max_inrange_all[det_inrange_all == 1], marker='.')
#plt.gca().set_xscale("log")
#plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
#ax.set_xticks([0.5,1.,2.,4.,8.])
#ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
xtick_vals = np.array([0.5,1.,2.,4.,8.])
#ytick_vals = np.array([1e-3, 1e-2, 0.1, 1., 10., 1e2, 1e3])
plt.xticks(np.log10(xtick_vals), xtick_vals)
#plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(radii_min), np.log10(radii_max)]) #[radii_min, radii_max]
plt.ylim([-3., 3.]) #[1e-3, 1e3]
plt.xlabel(r'$R_p$ ($R_\oplus$)', fontsize=tfs)
plt.ylabel(r'$\log_{10}(K/K_{\rm max})$', fontsize=tfs)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist(Rp_inrange_all[det_inrange_all == 1], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), histtype='step', color='k', ls='-', label=r'Observed')
plt.hist(Rp_inrange_all[(K_K_max_inrange_all > 1.) & (det_inrange_all == 1)], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), histtype='step', color='b', ls='-', label=r'$K/K_{\rm max} > 1$')
plt.hist(Rp_inrange_all[(K_K_max_inrange_all < 1.) & (det_inrange_all == 1)], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), histtype='step', color='r', ls='-', label=r'$K/K_{\rm max} < 1$')
plt.gca().set_xscale("log")
plt.xlim([radii_min, radii_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(K_K_max_inrange_all[det_inrange_all == 1], bins=np.logspace(-3., 3., bins+1), histtype='step', orientation='horizontal', color='k', ls='-')
plt.axhline(y=1., color='g', ls='--', lw=3)
plt.gca().set_yscale("log")
plt.ylim([1e-3, 1e3])
plt.xticks([])
plt.yticks([])
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_RVratio_obs_5-10d_planet_radii.pdf')
    plt.close()
#'''



# K_max/K vs. period ratio (observed planets):

pr_min, pr_max = 3./10., 300./5.

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5, 5, left=0.18, bottom=0.13, right=0.98, top=0.98, wspace=0, hspace=0)

ax = plt.subplot(plot[1:3,:4]) # upper half, for K_max/K > 1
x = 1./K_K_max_inrange_all
corner.hist2d(np.log10(P_K_max_or_2ndmax_all/P_inrange_all)[(x > 1) & (det_inrange_all == 1)], np.log10(x[(x > 1) & (det_inrange_all == 1)]), bins=15, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.axhline(y=0., color='g', ls='--', lw=3)
plt.axvline(x=0., color='g', ls='--', lw=3)
ax.tick_params(axis='both', labelsize=afs)
#xtick_vals = np.array([0.5, 1., 2., 4., 8., 16., 32.])
ytick_vals = np.array([1., 10., 100.])
plt.xticks([])
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(pr_min), np.log10(pr_max)])
plt.ylim([0., 2.]) #[1e-3, 1e3]
#plt.xlabel(r'$P_{K_{\rm max}}/P_{[5,10]d}$', fontsize=tfs)
plt.ylabel(r'$K_{\rm max}/K_{[5,10]d}$', fontsize=tfs)

ax = plt.subplot(plot[3:,:4]) # lower half, for K_2ndmax/K < 1
corner.hist2d(np.log10(P_K_max_or_2ndmax_all/P_inrange_all)[(x < 1) & (det_inrange_all == 1)], np.log10(x[(x < 1) & (det_inrange_all == 1)]), bins=15, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.axhline(y=0., color='g', ls='--', lw=3)
plt.axvline(x=0., color='g', ls='--', lw=3)
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([0.5, 1., 2., 4., 8., 16., 32.])
ytick_vals = np.array([0.01, 0.1])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(pr_min), np.log10(pr_max)])
plt.ylim([-2., 0.]) #[1e-3, 1e3]
plt.xlabel(r'$P_{K_{\rm max\;or\;2nd\;max}}/P_{[5,10]d}$', fontsize=tfs)
plt.ylabel(r'$K_{\rm 2nd\;max}/K_{[5,10]d}$', fontsize=tfs)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist((P_K_max_or_2ndmax_all/P_inrange_all)[det_inrange_all == 1], bins=np.logspace(np.log10(pr_min), np.log10(pr_max), bins+1), histtype='step', color='k', ls='-', label=r'All')
plt.hist((P_K_max_or_2ndmax_all/P_inrange_all)[(x > 1.) & (det_inrange_all == 1)], bins=np.logspace(np.log10(pr_min), np.log10(pr_max), bins+1), histtype='step', color='b', ls='-', label=r'$K_{\rm max}/K_{[5,10]d} > 1$')
plt.hist((P_K_max_or_2ndmax_all/P_inrange_all)[(x < 1.) & (det_inrange_all == 1)], bins=np.logspace(np.log10(pr_min), np.log10(pr_max), bins+1), histtype='step', color='r', ls='-', label=r'$K_{\rm 2nd\;max}/K_{[5,10]d} < 1$')
plt.axvline(x=1., color='g', ls='--', lw=3)
plt.gca().set_xscale("log")
plt.xlim([pr_min, pr_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(x[det_inrange_all == 1], bins=np.logspace(-3., 3., bins+1), weights=np.ones(np.sum(det_inrange_all == 1))/np.sum(det_inrange_all == 1), histtype='step', orientation='horizontal', color='k', ls='-')
#plt.hist(x, bins=np.logspace(-3., 3., bins+1), weights=np.ones(len(x))/len(x), histtype='step', orientation='horizontal', color='r', ls='-')
plt.axhline(y=1., color='g', ls='--', lw=3)
plt.gca().set_yscale("log")
plt.ylim([1e-2, 1e2])
plt.xticks([])
plt.yticks([])
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_RVratio_obs_5-10d_pratio.pdf')
    plt.close()

plt.show()
