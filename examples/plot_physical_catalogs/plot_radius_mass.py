# To import required modules:
import numpy as np
import time
import os
import sys
import copy
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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Underlying/'





##### To generate a simple underlying population using SysSimExClusters for testing:
##### This will call Julia functions from Python
from generate_test_population_SysSimExClusters import *

mass_min, mass_max = 0.1, 1e3
radii_min, radii_max = 0.5, 10.

##### To load the underlying populations:
##### TODO: figure out how to save the draws of the initial planet properties from the simulations of the Hybrid Model
'''
loadfiles_directory = ''
run_number = ''

N_sim, cos_factor, period_min, period_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)
'''





#####

N_pl_plot = 1000 # number of planets to plot

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 12 #legend labels font size

bins = 100



##### To plot radius vs. mass from several models:

fig = plt.figure(figsize=(10,8))
plot = GridSpec(5, 5, left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4]) # main panel
# Plot the mass-radius/radius-mass relations:
plt.plot(mass_evals_med_H20, radius_evals_H20, '--', color='k') #, label='H20, mean prediction'
plt.fill_betweenx(radius_evals_H20, mass_evals_016_H20, mass_evals_084_H20, color='k', alpha=0.2, label=r'H20 model, 16%-84% region') # 16%-84% region of H20 model
plt.plot(M_array, μ_R_array, '-', color='b') #, label='NR20, Model 2, mean'
plt.fill_between(M_array, μ_R_array*(1+σ_R_array), μ_R_array*(1-σ_R_array), color='b', alpha=0.2, label='NR20, Model 2 (initial)')
plt.plot(M_array, R_S07_silicate_array, color='g') #, label='S07, pure-silicate'
plt.fill_between(M_array, 0.95*R_S07_silicate_array, 1.05*R_S07_silicate_array, color='g', alpha=0.2, label='NR20, 5% scatter around \nS07 pure-silicate (final)') #, label='NR20, 5% scatter around S07 pure-silicate (final)'
# Plot the sample population:
plt.scatter(M_init_all[:N_pl_plot], R_init_all[:N_pl_plot], s=10, marker='o', edgecolors='k', facecolors='none', label='Sample population (initial)')
plt.scatter(M_final_all[:N_pl_plot], R_final_all[:N_pl_plot], s=10, marker='.', c='r', label='Sample population (final)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([1e-1, 1., 10., 1e2, 1e3])
ytick_vals = np.array([0.5, 1., 2., 4., 10.])
plt.xticks(xtick_vals, xtick_vals)
plt.yticks(ytick_vals, ytick_vals)
plt.xlim([mass_min, mass_max])
plt.ylim([radii_min, radii_max])
plt.xlabel(r'Planet mass, $M_p$ [$M_\oplus$]', fontsize=tfs)
plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist([M_init_all, M_final_all], bins=np.logspace(np.log10(mass_min), np.log10(mass_max), bins+1), histtype='step', color=['k','r'], ls='-', label=['Initial','Final'])
plt.gca().set_xscale("log")
plt.xlim([mass_min, mass_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist([R_init_all, R_final_all], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), histtype='step', orientation='horizontal', color=['k','r'], ls='-', label=['Initial','Final'])
plt.gca().set_yscale("log")
plt.ylim([radii_min, radii_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Models_underlying_radius_mass.pdf')
    plt.close()
plt.show()





##### To make a simpler version, for the paper:

fig = plt.figure(figsize=(7,7))
plot = GridSpec(5, 1, left=0.2, bottom=0.15, right=0.9, top=0.95, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:]) # main panel
plt.plot(mass_evals_med_H20, radius_evals_H20, '--', color='k') #, label='H20, mean prediction'
plt.fill_betweenx(radius_evals_H20, mass_evals_016_H20, mass_evals_084_H20, color='k', alpha=0.2, label=r'H20 model, 16%-84% region') # 16%-84% region of H20 model
plt.plot(M_array, μ_R_array, '-', color='c') #, label='NR20, Model 2, mean'
plt.fill_between(M_array, μ_R_array*(1+σ_R_array), μ_R_array*(1-σ_R_array), color='c', alpha=0.2, label='NR20, Model 2 (initial)')
plt.plot(M_array, R_S07_silicate_array, color='tab:brown') #, label='S07, pure-silicate'
plt.fill_between(M_array, 0.95*R_S07_silicate_array, 1.05*R_S07_silicate_array, color='tab:brown', alpha=0.5, label='NR20, 5% scatter around \nS07 pure-silicate (final)') #, label='NR20, 5% scatter around S07 pure-silicate (final)'
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([1e-1, 1., 10., 1e2, 1e3])
ytick_vals = np.array([0.5, 1., 2., 4., 10.])
plt.xticks(xtick_vals, [0.1, 1.0, 10, 100, 1000])
plt.yticks(ytick_vals, ytick_vals)
plt.xlim([mass_min, mass_max])
plt.ylim([radii_min, radii_max])
plt.xlabel(r'Planet mass, $M_p$ [$M_\oplus$]', fontsize=tfs)
plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[0,:]) # top panel
plt.plot(M_array, Minit_pdf_array, '-', color='c', label='NR20, Model 2 \n(initial mass distribution)')
plt.gca().set_xscale("log")
plt.xlim([mass_min, mass_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='lower left', bbox_to_anchor=(0.1,0), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Models_underlying_radius_mass_simple.pdf')
    plt.close()
plt.show()
