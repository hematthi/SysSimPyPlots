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





##### To load some mass-radius tables:

# NWG-2018 model:
MR_table_file = '../../src/syssimpyplots/data/MRpredict_table_weights3025_R1001_Q1001.txt'
with open(MR_table_file, 'r') as file:
    lines = (line for line in file if not line.startswith('#'))
    MR_table = np.genfromtxt(lines, names=True, delimiter=', ')

# Li Zeng models:
# https://www.cfa.harvard.edu/~lzeng/tables/massradiusEarthlikeRocky.txt
# https://www.cfa.harvard.edu/~lzeng/tables/massradiusFe.txt
MR_earthlike_rocky = np.genfromtxt('/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters-select_files/Miscellaneous_data/MR_earthlike_rocky.txt', names=['mass','radius']) # mass and radius are in Earth units
MR_pure_iron = np.genfromtxt('/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters-select_files/Miscellaneous_data/MR_pure_iron.txt', names=['mass','radius']) # mass and radius are in Earth units

# To construct an interpolation function for each MR relation:
MR_NWG2018_interp = scipy.interpolate.interp1d(10.**MR_table['log_R'], 10.**MR_table['05'])
MR_earthlike_rocky_interp = scipy.interpolate.interp1d(MR_earthlike_rocky['radius'], MR_earthlike_rocky['mass'])
MR_pure_iron_interp = scipy.interpolate.interp1d(MR_pure_iron['radius'], MR_pure_iron['mass'])

# To find where the Earth-like rocky relation intersects with the NWG2018 mean relation (between 1.4-1.5 R_earth):
def diff_MR(R):
    M_NWG2018 = MR_NWG2018_interp(R)
    M_earthlike_rocky = MR_earthlike_rocky_interp(R)
    return np.abs(M_NWG2018 - M_earthlike_rocky)
# The intersection is approximately 1.472 R_earth
radii_switch = 1.472

# H20 modification below 'radii_switch: Lognormal distribution for mass centered around Earth-like rocky, with a sigma_log_M that grows with radius
# To define sigma_log_M as a linear function of radius:
sigma_log_M_at_radii_switch = 0.3 # std of log_M (Earth masses) at radii_switch
sigma_log_M_at_radii_min = 0.04 # std of log_M (Earth masses) at radii_min
sigma_log_M_radius_slope = (sigma_log_M_at_radii_switch - sigma_log_M_at_radii_min) / (radii_switch - radii_min)
sigma_log_M = sigma_log_M_radius_slope*(MR_earthlike_rocky['radius'] - radii_min) + sigma_log_M_at_radii_min

# H20 model:
end_ELR = 27-1
start_NWG = 284-1 # index closest to log10(R=1.472)
radius_evals_H20 = np.concatenate((MR_earthlike_rocky['radius'][:end_ELR], 10.**MR_table['log_R'][start_NWG:]))
mass_evals_med_H20 = np.concatenate((MR_earthlike_rocky['mass'][:end_ELR], 10.**MR_table['05'][start_NWG:]))
mass_evals_016_H20 = np.concatenate((10.**(np.log10(MR_earthlike_rocky['mass'])-sigma_log_M)[:end_ELR], 10.**MR_table['016'][start_NWG:]))
mass_evals_084_H20 = np.concatenate((10.**(np.log10(MR_earthlike_rocky['mass'])+sigma_log_M)[:end_ELR], 10.**MR_table['084'][start_NWG:]))





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
    plt.savefig(savefigures_directory + 'radius_mass_models.pdf')
    plt.close()
plt.show()
