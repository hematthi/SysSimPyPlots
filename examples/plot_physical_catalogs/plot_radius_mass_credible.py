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





##### To compute the credible regions for the median radius-mass relation from the model posterior distribution:
##### This will call Julia functions from Python

mass_min, mass_max = 0.1, 1e3
radii_min, radii_max = 0.5, 10.

M_array = np.logspace(np.log10(mass_min), np.log10(mass_max), 1000)
qtls = [0.16,0.5,0.84] # for 1-sigma

##### (1) From the model parameters drawn from the posterior distribution:

#run_directory = 'Hybrid_NR20_AMD_model1/Fit_all_KS/Params12/GP_files/'
#loadfiles_directory = '/Users/hematthi/Documents/NotreDame_Postdoc/CRC/Files/SysSim/Model_Optimization/' + run_directory
run_directory = 'Hybrid_NR20_AMD_model1/Fit_all_KS/Params8/GP_files/'
loadfiles_directory = '/Users/hematthi/Documents/NPP_ARC_Modernize_Kepler/Personal_research/SysSim/Model_Optimization/' + run_directory

n_params = 8
#n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 37.65, 1425.6 # 12 params
#n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 67.65, 141134.4 # 13 params
n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 16.05, 14.26 # 8 params
n_points, max_mean, max_std, max_post = 100000, 'Inf', 'Inf', -10.0 #100000, 'Inf', 'Inf', 'Inf'
file_name = 'GP_train%s_meanf%s_sigmaf%s_lscales%s_vol%s_points%s_mean%s_std%s_post%s.csv' % (n_train, mean_f, sigma_f, lscales, vol, n_points, max_mean, max_std, max_post)
xprior_accepted_table = load_GP_table_prior_draws(file_name, file_name_path=loadfiles_directory)
active_params_names = np.array(xprior_accepted_table.dtype.names[:n_params])

# To compute the median radius-mass prediction for each set of model parameters:
points_to_use = min(1000, len(xprior_accepted_table))
μ_R_array_all = np.zeros((points_to_use, len(M_array)))
σ_R_array_all = np.zeros((points_to_use, len(M_array)))
for i in range(points_to_use):
    params = xprior_accepted_table[i]
    
    C = params['norm_radius']
    M_break1 = params['break1_mass']
    γ0, γ1 = params['power_law_γ0'], params['power_law_γ1']
    σ0, σ1 = params['power_law_σ0'], params['power_law_σ1']
    
    μσ_R_array = np.array([Main.mean_radius_and_scatter_given_mass_neil_rogers2020(M, C=C, M_break1=M_break1, M_break2=1e4, γ0=γ0, γ1=γ1, γ2=0., σ0=σ0, σ1=σ1, σ2=0.1) for M in M_array])
    μ_R_array_all[i] = np.array([μσ_R[0] for μσ_R in μσ_R_array])
    σ_R_array_all[i] = np.array([μσ_R[1] for μσ_R in μσ_R_array])

μ_R_array_qtls = np.quantile(μ_R_array_all, qtls, axis=0)

##### (2) From the model parameters drawn from the posterior distribution that have simulated catalogs (that still pass the distance threshold after simulation):

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_all_KS/Params8/GP_best_models_100/'

params_all = []

runs = 100
for i in range(runs):
    run_number = i+1
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    params_all.append(params_i)

# To compute the median radius-mass prediction for each set of model parameters:
μ_R_array_all_sim = np.zeros((runs, len(M_array)))
σ_R_array_all_sim = np.zeros((runs, len(M_array)))
for i,params in enumerate(params_all):
    C = params['norm_radius (R_earth)']
    M_break1 = params['break_mass (M_earth)']
    γ0, γ1 = params['power_law_γ0'], params['power_law_γ1']
    σ0, σ1 = params['power_law_σ0'], params['power_law_σ1']
    
    μσ_R_array = np.array([Main.mean_radius_and_scatter_given_mass_neil_rogers2020(M, C=C, M_break1=M_break1, M_break2=1e4, γ0=γ0, γ1=γ1, γ2=0., σ0=σ0, σ1=σ1, σ2=0.1) for M in M_array])
    μ_R_array_all_sim[i] = np.array([μσ_R[0] for μσ_R in μσ_R_array])
    σ_R_array_all_sim[i] = np.array([μσ_R[1] for μσ_R in μσ_R_array])

μ_R_array_qtls_sim = np.quantile(μ_R_array_all_sim, qtls, axis=0)



R_S07_silicate_array = np.array([Main.radius_given_mass_pure_silicate_fit_seager2007(M) for M in M_array])





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
plt.plot(M_array, μ_R_array_qtls[1], '-', color='m')
plt.fill_between(M_array, μ_R_array_qtls[2], μ_R_array_qtls[0], color='m', alpha=0.2, label='Hybrid model, 16%-84% region of median (posterior draws)')
plt.plot(M_array, μ_R_array_qtls_sim[1], '-', color='b')
plt.fill_between(M_array, μ_R_array_qtls_sim[2], μ_R_array_qtls_sim[0], color='b', alpha=0.2, label='Hybrid model, 16%-84% region of median (posterior draws + simulated)')
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
