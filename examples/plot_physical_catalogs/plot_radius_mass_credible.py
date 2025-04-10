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

##### For calling Julia functions:
# Import Julia:
from julia.api import Julia
jl = Julia(compiled_modules=False)

# Import Julia modules:
from julia import Main
Main.include("/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/src/models_test.jl") # can now call functions in this script using "jl.eval(f"name_of_function({a},{b})")" or just "Main.name_of_function(a,b)"!
Main.include("/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/src/clusters.jl")
#####





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Underlying/'





##### To load the model parameters for a large number of models drawn from the posterior distribution:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_all_KS/Params12/GP_best_models_100/'

params_all = []

runs = 100
for i in range(runs):
    run_number = i+1
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    params_all.append(params_i)

##### To compute the median radius-mass prediction for each set of model parameters:
##### This will call Julia functions from Python

mass_min, mass_max = 0.1, 1e3
radii_min, radii_max = 0.5, 10.

M_array = np.logspace(np.log10(mass_min), np.log10(mass_max), 1000)
μ_R_array_all = np.zeros((runs, len(M_array)))
σ_R_array_all = np.zeros((runs, len(M_array)))

for i,params in enumerate(params_all):
    C = params['norm_radius (R_earth)']
    M_break1 = params['break_mass (M_earth)']
    γ0, γ1 = params['power_law_γ0'], params['power_law_γ1']
    σ0, σ1 = params['power_law_σ0'], params['power_law_σ1']
    
    μσ_R_array = np.array([Main.mean_radius_and_scatter_given_mass_neil_rogers2020(M, C=C, M_break1=M_break1, M_break2=1e4, γ0=γ0, γ1=γ1, γ2=0., σ0=σ0, σ1=σ1, σ2=0.1) for M in M_array])
    μ_R_array_all[i] = np.array([μσ_R[0] for μσ_R in μσ_R_array])
    σ_R_array_all[i] = np.array([μσ_R[1] for μσ_R in μσ_R_array])

μ_R_array_qtls = np.quantile(μ_R_array_all, [0.16,0.5,0.84], axis=0)

R_S07_silicate_array = np.array([Main.radius_given_mass_pure_silicate_fit_seager2007(M) for M in M_array])





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

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 12 #legend labels font size

bins = 100



##### To plot radius vs. mass from several models:

fig = plt.figure(figsize=(10,8))
plot = GridSpec(1, 1, left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

ax = plt.subplot(plot[:,:])
# Plot the mass-radius/radius-mass relations:
plt.plot(mass_evals_med_H20, radius_evals_H20, '--', color='k') #, label='H20, mean prediction'
plt.fill_betweenx(radius_evals_H20, mass_evals_016_H20, mass_evals_084_H20, color='k', alpha=0.2, label=r'H20 model, 1$\sigma$ scatter around median') # 16%-84% region of H20 model
plt.plot(M_array, μ_R_array_qtls[1], '-', color='b') #, label='NR20, Model 2, mean'
plt.fill_between(M_array, μ_R_array_qtls[2], μ_R_array_qtls[0], color='b', alpha=0.2, label='Hybrid model, 16%-84% region of median')
plt.plot(M_array, R_S07_silicate_array, color='g') #, label='S07, pure-silicate'
plt.fill_between(M_array, 0.95*R_S07_silicate_array, 1.05*R_S07_silicate_array, color='g', alpha=0.2, label='S07 pure-silicate model with 5% scatter') #, label='NR20, 5% scatter around S07 pure-silicate (final)'
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

if savefigures:
    plt.savefig(savefigures_directory + 'Models_underlying_radius_mass_credible.pdf')
    plt.close()
plt.show()
