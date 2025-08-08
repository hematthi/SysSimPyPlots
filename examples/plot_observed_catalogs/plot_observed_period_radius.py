# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm # for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for specifying plot attributes
from matplotlib import ticker # for setting contour plots to log scale
from matplotlib.colors import LogNorm # for log color scales
import scipy.integrate # for numerical integration
import scipy.misc # for factorial function
from scipy.special import erf # error function, used in computing CDF of normal distribution
import scipy.interpolate # for interpolation functions
from scipy.stats import gaussian_kde
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *

from gapfit import gap_line, bootstrap_fit_gap, uncertainty_from_boots





savefigures = False
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8_KS/Params9_fix_highM/GP_best_models_100/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Observed/'
run_number = '89'
model_name = 'Hybrid_model' + run_number

compute_ratios = compute_ratios_adjacent
weights_all = load_split_stars_weights_only()
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 'period_ratios_KS',
                 'durations_KS',
                 #'durations_norm_circ_KS',
                 #'durations_norm_circ_singles_KS',
                 #'durations_norm_circ_multis_KS',
                 'duration_ratios_KS',
                 #'duration_ratios_nonmmr_KS',
                 #'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radii_KS',
                 'radius_ratios_KS',
                 'radii_partitioning_KS',
                 'radii_monotonicity_KS',
                 'gap_complexity_KS',
                 ]





##### To load the files with the systems with observed planets:

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated observed catalog of stars and planets:
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor)





##### To perform a gaussian KDE for the log(period)-log(radius) distributions:

logP_min, logP_max = np.log10(P_min), np.log10(P_max)
logR_min, logR_max = np.log10(radii_min), np.log10(radii_max)
logP_grid, logR_grid = np.mgrid[logP_min:logP_max:100j, logR_min:logR_max:100j] # complex step size '100j' to include the upper bound
positions = np.vstack([logP_grid.ravel(), logR_grid.ravel()])

values_sim = np.vstack([np.log10(sss['P_obs']), np.log10(sss['radii_obs'])])
values_Kep = np.vstack([np.log10(ssk['P_obs']), np.log10(ssk['radii_obs'])])
kde_sim = gaussian_kde(values_sim)
kde_Kep = gaussian_kde(values_Kep)
f_sim = np.reshape(kde_sim(positions).T, np.shape(logP_grid))
f_Kep = np.reshape(kde_Kep(positions).T, np.shape(logP_grid))

##### To fit a line for the radius valley using gapfit (https://github.com/parkus/gapfit):

x0 = 0. # offset location, log10(P)
y0_guess = 0.5 # guess for the offset, log10(Rgap_offset)
m_guess = -0.1 # guess for the slope
sig = 0.15 # kernel width
y0_rng = 0.5 # maximum allowed deviation from y0 in search

# Fitting the Kepler catalog:
boots_Kep = bootstrap_fit_gap(np.log10(ssk['P_obs']), np.log10(ssk['radii_obs']), x0, y0_guess, m_guess, sig, y0_rng, nboots=500)
# Fitting the simulated catalog:
boots_sim = bootstrap_fit_gap(np.log10(sss['P_obs']), np.log10(sss['radii_obs']), x0, y0_guess, m_guess, sig, y0_rng, nboots=500)

triplets_Kep = uncertainty_from_boots(boots_Kep)
y0trip_Kep, mtrip_Kep = triplets_Kep # [best-fit, +1sigma, -1sigma] for each of y0, m
triplets_sim = uncertainty_from_boots(boots_sim)
y0trip_sim, mtrip_sim = triplets_sim # [best-fit, +1sigma, -1sigma] for each of y0, m
print('Best-fit gap line for the Kepler catalog:')
print('m = {:.2f} +{:.2f}/-{:.2f}'.format(*mtrip_Kep))
print('y0 = {:.2f} +{:.2f}/-{:.2f}'.format(*y0trip_Kep))
print('Best-fit gap line for the simulated catalog:')
print('m = {:.2f} +{:.2f}/-{:.2f}'.format(*mtrip_sim))
print('y0 = {:.2f} +{:.2f}/-{:.2f}'.format(*y0trip_sim))

m_Kep = mtrip_Kep[0] # best-fit slope
logRgap_offset_Kep = y0trip_Kep[0] # best-fit intercept

# Best-fit from Van Eylen et al. (2018):
m_VE18 = -0.10 # slope
logRgap_offset_VE18 = 0.38 # intercept at P=1 day

# Carrera et al. (2018):
m_C18 = -0.1467
logRgap_offset_C18 = np.log10(2.6)

# If want to use one of the above:
#m_Kep = m_VE18
#logRgap_offset_Kep = logRgap_offset_VE18

m_sim = mtrip_sim[0] # best-fit slope
logRgap_offset_sim = y0trip_sim[0] # best-fit intercept

# Evaluate the line for the radius valley along an array of log10(P):
logP_array = np.linspace(np.log10(P_min), np.log10(P_max), 101)

logRgap_array_Kep = m_Kep*logP_array + logRgap_offset_Kep
#Rgap_array_Kep = 10.**logRgap_array_Kep
logRgap_array_sim = m_sim*logP_array + logRgap_offset_sim
#Rgap_array_sim = 10.**logRgap_array_sim





#####

n_bins = 100
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size



##### To plot the period-radius distributions of the simulated and Kepler catalogs:

#load_cat_obs_and_plot_fig_period_radius(loadfiles_directory, run_number=run_number, lw=lw, save_name='no_name_fig.pdf', save_fig=False)
#plt.show()

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,2,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0.1,hspace=0)
cmap = 'Blues' #'Blues' #'viridis'

ax = plt.subplot(plot[:,0]) # for the Kepler distribution
# KDE contours:
plt.contourf(logP_grid, logR_grid, f_Kep, cmap=cmap)
# Scatter points:
plt.scatter(np.log10(ssk['P_obs']), np.log10(ssk['radii_obs']), s=5, marker='o', edgecolor='k', facecolor='none', label='Kepler')
# Radius valley fit:
plt.plot(logP_array, logRgap_array_Kep, ls='--', lw=lw, color='k', label=r'gapfit: $R_{\rm gap} = {%s}P^{%s}$' % ('{:0.2f}'.format(10.**logRgap_offset_Kep), '{:0.2f}'.format(m_Kep)))
plt.plot(logP_array, m_VE18*logP_array + logRgap_offset_VE18, ls='--', lw=lw, color='b', label=r'Van Eylen+2018: $R_{\rm gap} = {%s}P^{%s}$' % ('{:0.2f}'.format(10.**logRgap_offset_VE18), '{:0.2f}'.format(m_VE18))) # Van Eylen et al. (2018) best-fit
#plt.plot(logP_array, m_C18*logP_array + logRgap_offset_C18, ls='--', lw=lw, color='r', label=r'Carrera+2018: $R_{\rm gap} = {%s}P^{%s}$' % ('{:0.2f}'.format(10.**logRgap_offset_C18), '{:0.2f}'.format(m_C18))) # Carrera et al. (2018) photoevaporation boundary
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([3,10,30,100,300])
ytick_vals = np.array([0.5,1,2,4,10])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(P_min), np.log10(P_max)])
plt.ylim([np.log10(radii_min), np.log10(radii_max)])
plt.xlabel(r'Orbital period, $P$ [days]', fontsize=tfs)
plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[:,1]) # for the simulated distribution
# KDE contours:
plt.contourf(logP_grid, logR_grid, f_sim, cmap=cmap)
# Scatter points:
plt.scatter(np.log10(sss['P_obs']), np.log10(sss['radii_obs']), s=5, marker='o', edgecolor='b', facecolor='none', label='Simulated')
# Radius valley fit:
plt.plot(logP_array, logRgap_array_sim, ls='--', lw=lw, color='k')
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([3,10,30,100,300])
ytick_vals = np.array([0.5,1,2,4,10])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), [])
plt.xlim([np.log10(P_min), np.log10(P_max)])
plt.ylim([np.log10(radii_min), np.log10(radii_max)])
plt.xlabel(r'Orbital period, $P$ [days]', fontsize=tfs)
#plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_period_radius.pdf')
    plt.close()
plt.show()
