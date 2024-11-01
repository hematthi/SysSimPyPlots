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
import scipy.stats # for gaussian KDEs
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Underlying/'
run_number = ''
model_name = 'Hybrid_NR20_AMD_model1' + run_number
model_label, model_color = 'Hybrid model', 'b'





##### To generate a simple underlying population using SysSimExClusters for testing:
##### This will call Julia functions from Python
from generate_test_population_SysSimExClusters import *

##### To load the underlying populations:
##### TODO: figure out how to save the draws of the initial planet properties from the simulations of the Hybrid Model

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, period_min, period_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated physical catalog of stars and planets:
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)





##### To perform a gaussian KDE for the log(period)-log(radius) distribution:

logP_min, logP_max = np.log10(period_min), np.log10(period_max)
logR_min, logR_max = np.log10(radii_min), np.log10(radii_max)
logP_grid, logR_grid = np.mgrid[logP_min:logP_max:100j, logR_min:logR_max:100j] # complex step size '100j' to include the upper bound
positions = np.vstack([logP_grid.ravel(), logR_grid.ravel()])

values = np.vstack([np.log10(sssp['P_all']), np.log10(sssp['radii_all'])])
kde = scipy.stats.gaussian_kde(values)
f = np.reshape(kde(positions).T, np.shape(logP_grid))

values_init = np.vstack([np.log10(P_all), np.log10(R_init_all)])
values_final = np.vstack([np.log10(P_all), np.log10(R_final_all)])
kde_init = scipy.stats.gaussian_kde(values_init)
kde_final = scipy.stats.gaussian_kde(values_final)
f_init = np.reshape(kde_init(positions).T, np.shape(logP_grid))
f_final = np.reshape(kde_final(positions).T, np.shape(logP_grid))







#####

N_pl_plot = 1000
n_bins = 100
lw = 1 # linewidth

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size



##### To plot the period-radius distribution:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,2,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0.1,hspace=0)
cmap = 'Blues' #'Blues' #'viridis'

ax = plt.subplot(plot[:,0]) # for the initial distribution (before photoevaporation)
plt.figtext(0.02, 0.95, 'Initial (before photo-evaporation)', transform=ax.transAxes, fontsize=lfs)
# KDE contours:
plt.contourf(logP_grid, logR_grid, f_init, cmap=cmap)
# Scatter points:
plt.scatter(np.log10(P_all[:N_pl_plot]), np.log10(R_init_all[:N_pl_plot]), s=5, marker='o', edgecolor='k', facecolor='none', label='All planets')
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([3,10,30,100,300])
ytick_vals = np.array([0.5,1,2,4,10])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([logP_min, logP_max])
plt.ylim([logR_min, logR_max])
plt.xlabel(r'Orbital period, $P$ [days]', fontsize=tfs)
plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
#plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[:,1]) # for the final distribution (after photoevaporation)
plt.figtext(0.02, 0.95, 'Final (after photo-evaporation)', transform=ax.transAxes, fontsize=lfs)
# KDE contours:
#plt.contourf(logP_grid, logR_grid, f, cmap=cmap)
plt.contourf(logP_grid, logR_grid, f_final, cmap=cmap)
# Scatter points:
#plt.scatter(np.log10(sssp['P_all'][:N_pl_plot]), np.log10(sssp['radii_all'][:N_pl_plot]), s=5, marker='o', edgecolor='b', facecolor='none', label='Final')
plt.scatter(np.log10(P_all[:N_pl_plot][bools_ret_all[:N_pl_plot]]), np.log10(R_final_all[:N_pl_plot][bools_ret_all[:N_pl_plot]]), s=5, marker='o', edgecolor='b', facecolor='none', label='Retained envelope')
plt.scatter(np.log10(P_all[:N_pl_plot][~bools_ret_all[:N_pl_plot]]), np.log10(R_final_all[:N_pl_plot][~bools_ret_all[:N_pl_plot]]), s=5, marker='o', edgecolor='r', facecolor='none', label='Lost envelope')
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([3,10,30,100,300])
ytick_vals = np.array([0.5,1,2,4,10])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), [])
plt.xlim([logP_min, logP_max])
plt.ylim([logR_min, logR_max])
plt.xlabel(r'Orbital period, $P$ [days]', fontsize=tfs)
#plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
plt.legend(loc='upper left', bbox_to_anchor=(0,0.9), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_period_radius.pdf')
    plt.close()
plt.show()
