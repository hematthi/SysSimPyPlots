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
from scipy.stats import ks_2samp
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Observed_compare/'
save_name = 'Models_Compare'





##### To load and compute the same statistics for a large number of models:

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'

model_loadfiles_dirs = [loadfiles_directory1, loadfiles_directory2]
models = len(model_loadfiles_dirs)
model_names = ['Maximum AMD model', 'Two-Rayleigh model']
model_markers = ['o', 'x']
model_colors = ['g', 'b']

runs = 100
param_vals_all = []
alphaP_all = []
lambdac_all = []
for loadfiles_dir in model_loadfiles_dirs:
    param_vals_all_model = []
    alphaP_all_model = []
    lambdac_all_model = []
    for i in range(1,runs+1): #range(1,runs+1)
        run_number = i
        param_vals_i = read_sim_params(loadfiles_dir + 'periods%s.out' % run_number)
        param_vals_all_model.append(param_vals_i)
        alphaP_all_model.append(param_vals_i['power_law_P'])
        lambdac_all_model.append(param_vals_i['log_rate_clusters'])
        print(i, ': alpha_P = ', param_vals_i['power_law_P'], ', lambda_c = ', param_vals_i['log_rate_clusters'])
    param_vals_all.append(param_vals_all_model)
    alphaP_all.append(np.array(alphaP_all_model))
    lambdac_all.append(np.array(lambdac_all_model))





##### To compute the increase factor in the number of planets if extrapolating period power-law to longer periods:

def norm_factor_power_law(alpha, x0=3., x1=300., x2=3000.):
    assert x0 < x1 < x2
    if alpha == -1:
        A12 = np.log(x2/x1)/np.log(x1/x0)
    else:
        A12 = (x2**(alpha+1.)-x1**(alpha+1.)) / (x1**(alpha+1.)-x0**(alpha+1.))
    return 1. + A12

f_to_1000d_all = [np.array([norm_factor_power_law(alphaP, x2=1000.) for alphaP in alphaP_all[0]]), np.array([norm_factor_power_law(alphaP, x2=1000.) for alphaP in alphaP_all[1]])]
f_to_3000d_all = [np.array([norm_factor_power_law(alphaP, x2=3000.) for alphaP in alphaP_all[0]]), np.array([norm_factor_power_law(alphaP, x2=3000.) for alphaP in alphaP_all[1]])]





# Plot factor vs. alpha_P:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95)
ax = plt.subplot(plot[:,:])
alphaP_array = np.linspace(-1., 1.5, 101)
plt.plot(alphaP_array, [norm_factor_power_law(alphaP, x2=1000.) for alphaP in alphaP_array], c='k', label='Extrapolate to 1000d')
#plt.plot(alphaP_array, [norm_factor_power_law(alphaP, x2=400.) for alphaP in alphaP_array], c='k', label='Extrapolate to 400d')
for m in range(models):
    plt.scatter(alphaP_all[m], f_to_1000d_all[m], s=100., marker='+', c=model_colors[m], label=model_names[m])
    #plt.scatter(alphaP_all[m], f_to_3000d_all[m], marker='x', c=model_colors[m], label='Extrapolate to 3000d')
plt.gca().set_yscale("log")
ax.set_yticks([1.,2.,3.,4.,5.,7.,10.,20.,30.,40.,50.,100.])
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
plt.xlim([-1., 1.5])
plt.ylim([1., 25.])
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\alpha_P$', fontsize=16)
plt.ylabel(r'$f$', fontsize=16)
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=16)
if savefigures:
    fig_name = savefigures_directory + save_name + '_extrapolate_period_powerlaw_alphaP.pdf'
    plt.savefig(fig_name)
    plt.close()

# Plot lambda_c vs. alpha_P:
fig = plt.figure(figsize=(9,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.85,top=0.95)
ax = plt.subplot(plot[:,:])
for m in range(models):
    sc = plt.scatter(alphaP_all[m], lambdac_all[m], s=100., marker=model_markers[m], c=f_to_1000d_all[m], cmap='YlOrRd', norm=MidPointLogNorm(vmin=1.,vmax=20.,midpoint=5.), label=model_names[m])
plt.gca().set_yscale("log")
ax.set_yticks([0.2,0.3,0.5,1.,2.,3.,5.])
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
plt.xlim([-0.8, 1.6])
plt.ylim([0.2, 6.])
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\alpha_P$', fontsize=16)
plt.ylabel(r'$\lambda_c$', fontsize=16)
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=16)
plot = GridSpec(1,1,left=0.86,bottom=0.1,right=0.9,top=0.95) # colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [1.,2.,3.,5.,10.,20.,30.,50.]
cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter())
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$f(1000d)$', fontsize=16)
if savefigures:
    fig_name = savefigures_directory + save_name + '_extrapolate_period_powerlaw_lambdac_vs_alphaP.pdf'
    plt.savefig(fig_name)
    plt.close()

# Plot f*lambda_c vs. lambda_c:
fig = plt.figure(figsize=(9,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.8,top=0.95)
ax = plt.subplot(plot[:,:])
for m in range(models):
    sc = plt.scatter(lambdac_all[m], f_to_1000d_all[m]*lambdac_all[m], s=100., marker=model_markers[m], c=alphaP_all[m], cmap='viridis', label=model_names[m])
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.set_xticks([0.2,0.3,0.5,1.,2.,3.,5.])
ax.set_yticks([0.2,0.3,0.5,1.,2.,3.,5.,10.,20.,30.,50.])
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
plt.xlim([0.2, 6.])
plt.ylim([0.5, 100.])
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\lambda_c$', fontsize=16)
plt.ylabel(r'$f(1000d) \times \lambda_c$', fontsize=16)
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=16)
plot = GridSpec(1,1,left=0.81,bottom=0.1,right=0.85,top=0.95) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(sc, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\alpha_P$', fontsize=16)
if savefigures:
    fig_name = savefigures_directory + save_name + '_extrapolate_period_powerlaw_flambdac_vs_lambdac.pdf'
    plt.savefig(fig_name)
    plt.close()

plt.show()
