# To import required modules:
import numpy as np
import time
#from tqdm import tqdm # for progress bar
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
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Underlying/'
run_number = ''
model_name = 'Hybrid_model' + run_number
model_label, model_color = 'HM-C', 'g' #'Maximum AMD model', 'g' #'Two-Rayleigh model', 'b'





# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated physical catalog of stars and planets:
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number)





##### To plot the simulated catalog as marginal distributions:

fig_size = (8,3) # size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 2
alpha = 0.2

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:
#'''
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/'

runs = 100

sssp_all = []
sssp_per_sys_all = []
params_all = []

M_env_all = []
M_core_all = []

for i in range(runs):
    run_number = i+1
    print(i)
    N_sim_i = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)[0]
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number)

    # Catalogs and parameters:
    sssp_all.append(sssp_i)
    sssp_per_sys_all.append(sssp_per_sys_i)
    params_all.append(params_i)
    
    # Compute the initial planet envelope masses and core masses:
    M_init_i = sssp_i['init_mass_all']
    M_env_i = envelope_mass_smoothed_low_high_neil_rogers2020(M_init_i)
    M_core_i = M_init_i - M_env_i
    
    M_env_all.append(M_env_i)
    M_core_all.append(M_core_i)
#'''
#####





##### First, plot the fractional envelope mass and core mass as a function of planet mass:

M_min, M_max = 0.1, 1e3
M_array = np.logspace(np.log10(M_min), np.log10(M_max), 1000)
M_env_array = envelope_mass_smoothed_low_high_neil_rogers2020(M_array)
M_core_array = M_array - M_env_array

# Plot fractional envelope mass vs. planet mass:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
plt.plot(M_array, M_env_array/M_array)
ax.set_xscale('log')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([M_min,M_max])
plt.ylim([0,1])
plt.xlabel(r'Planet mass, $M_p$ [$M_\oplus$]', fontsize=tfs)
plt.ylabel(r'Envelope mass fraction, $M_{\rm env}/M_p$', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_envelope_mass_fraction_vs_mass.pdf')
    plt.close()

# Plot core mass vs. planet mass:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
plt.plot(M_array, M_core_array)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([M_min,M_max])
plt.ylim([0,30])
plt.xlabel(r'Planet mass, $M_p$ [$M_\oplus$]', fontsize=tfs)
plt.ylabel(r'Core mass, $M_{\rm core}$ [$M_\oplus$]', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_core_mass_vs_mass.pdf')
    plt.close()

plt.show()





##### To plot the core mass distribution:
#'''
# A single catalog:
M_init_sim = sssp['init_mass_all']
M_env_sim = envelope_mass_smoothed_low_high_neil_rogers2020(M_init_sim)
M_core_sim = M_init_sim - M_env_sim

plot_fig_pdf_simple([M_core_sim], [], x_min=M_min, x_max=20., n_bins=n_bins, log_x=True, lw=lw, labels_sim=[model_label], xlabel_text=r'Core mass, $M_{\rm core}$ [$M_\oplus$]', afs=afs, tfs=tfs, lfs=lfs)

# Credible regions from many catalogs:
plot_fig_pdf_credible([M_core_all], [], x_min=1e-1, x_max=20., n_bins=n_bins, step=None, plot_median=True, log_x=True, log_y=False, c_sim_all=[model_color], lw=lw, alpha_all=[alpha], labels_sim_all=[model_label], xlabel_text=r'Core mass, $M_{\rm core}$ [$M_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_core_masses.pdf')
    plt.close()

plt.show()
#'''
