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

from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *
from syssimpyplots.optim import *





##### This module will be used to plot results of the optimization runs of our clustered model using bboptimize:

savefigures = False
plt.ioff()

run_directory = 'Hybrid_NR20_AMD_model1/Fit_split_KS/Params12/' #'Hybrid_NR20_AMD_model1/Fit_all_KS/Params13_alpha1_100/'
loadfiles_directory = '/Users/hematthi/Documents/NotreDame_Postdoc/CRC/Files/SysSim/Model_Optimization/' + run_directory
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Model_Optimization/' + run_directory

run_numbers = range(1,51) #np.loadtxt(loadfiles_directory + 'run_numbers.txt', dtype='i4')

sample_names = ['all', 'bluer', 'redder']
sample_colors = ['k', 'b', 'r']

model_name = 'Hybrid_NR20_AMD_model1'





##### To iterate through each of the optimization runs (files), and extract the results:

active_params_symbols = [r'$M_{\rm break,1}$',
                         r'$\ln{(\lambda_c)}$',
                         r'$\ln{(\lambda_p)}$',
                         r'$\mu_M$',
                         r'$R_{p,\rm norm}$',
                         r'$\alpha_P$',
                         r'$\gamma_0$',
                         r'$\gamma_1$',
                         r'$\sigma_0$',
                         r'$\sigma_1$',
                         r'$\sigma_M$',
                         r'$\sigma_P$',
                         ] # this list of parameter symbols must match the order of parameters in 'active_params_names'!

results = analyze_bboptimize_split_stars_runs(loadfiles_directory, run_numbers=run_numbers)

N_best = 100

##### To save the best parameter values for simulated catalog generation:
'''
table_header = 'run_number ' + ' '.join(results['active_params_names_all'][0])
fields_formats = ['%i'] + ['%1.6f']*len(results['active_params_names_all'][0])

i_best_N = np.argsort(dtot_w_all)[:N_best]
active_params_table = np.concatenate((np.array([np.arange(N_best)]).transpose(), results['active_params_all'][i_best_N]), axis=1)
np.savetxt('/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/' + run_directory + 'best_N/Active_params_table.txt', active_params_table, fmt=fields_formats, header=table_header, comments='')
'''

##### To save the best parameter values and the distances for training a GP emulator:
'''
N_best_save, keep_every = 100000, 10
i_best_N = np.argsort(results['dtot_w_all'])[0:N_best_save:keep_every]
active_params_distances_table = np.concatenate((results['active_params_all'][i_best_N], np.array([results['dtot_w_all'][i_best_N]]).transpose()), axis=1)
table_header = ' '.join(results['active_params_names_all'][0]) + ' dist_tot_weighted'
fields_formats = ['%1.6f']*len(results['active_params_names_all'][0]) + ['%1.6f']
np.savetxt(loadfiles_directory + 'Active_params_distances_table_best%s_every%s.txt' % (N_best_save, keep_every), active_params_distances_table, fmt=fields_formats, header=table_header, comments='')
'''

#sys.exit("Error message")





##### To make 2D plots of various pairs of parameters:

active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("log_rate_clusters", "power_law_P"),
                       ("log_rate_planets_per_cluster", "power_law_P"),
                       ("break1_mass", "mean_ln_mass"),
                       ("break1_mass", "norm_radius"),
                       ("break1_mass", "power_law_γ0"),
                       ("break1_mass", "power_law_γ1"),
                       ("mean_ln_mass", "log_rate_clusters"),
                       ("mean_ln_mass", "log_rate_planets_per_cluster"),
                       ("mean_ln_mass", "sigma_ln_mass"),
                       ("norm_radius", "power_law_γ0"),
                       ("norm_radius", "power_law_γ1"),
                       ("power_law_γ0", "power_law_σ0"),
                       ("power_law_γ1", "power_law_σ1"),
                       ("power_law_P", "sigma_logperiod_per_pl_in_cluster"),
                       ] # for Hybrid1 model with 12 active parameters



#'''
#To plot the total weighted distances for all the 2D plots on one figure:
N_panels = len(active_params_pairs)
cols = int(np.ceil(np.sqrt(N_panels))) #number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols #number of rows, such that rows*cols >= N_panels

#To plot the best total weighted distance found by each run:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.95,wspace=0.3,hspace=0.5)
for i,pair in enumerate(active_params_pairs):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])

    i_x, i_y = np.where(np.array(results['active_params_names_all'][0]) == pair[1])[0][0], np.where(np.array(results['active_params_names_all'][0]) == pair[0])[0][0]
    #print(pair, ': (ix,iy) = (%s,%s) ; x=%s, y=%s' % (i_x, i_y, active_params_symbols[i_x], active_params_symbols[i_y]))

    active_params_plot = results['active_params_best_all']
    colors = results['dtot_best_all']
    best_scatter = plt.scatter(active_params_plot[:,i_x], active_params_plot[:,i_y], marker='.', c=colors, s=200, alpha=1) #best values for each run
    plt.colorbar(best_scatter)

    plt.xlim(results['active_params_bounds_all'][0][i_x])
    plt.ylim(results['active_params_bounds_all'][0][i_y])
    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel(active_params_symbols[i_x], fontsize=16)
    plt.ylabel(active_params_symbols[i_y], fontsize=16)
    #plt.legend(loc='center left', bbox_to_anchor=(1.,0.5), ncol=1, fontsize=12)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_2D_summary_best_per_run.pdf')
else:
    plt.show()
#plt.close()

#To plot the best N total weighted distances found by all the runs in aggregate:
N_best = 100

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.95,wspace=0.3,hspace=0.5)
for i,pair in enumerate(active_params_pairs):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])

    i_x, i_y = np.where(np.array(results['active_params_names_all'][0]) == pair[1])[0][0], np.where(np.array(results['active_params_names_all'][0]) == pair[0])[0][0]

    i_best_N = np.argsort(results['dtot_w_all'])[:N_best]
    active_params_plot = results['active_params_all'][i_best_N]
    colors = results['dtot_w_all'][i_best_N]
    best_N_scatter = plt.scatter(active_params_plot[:,i_x], active_params_plot[:,i_y], marker=',', c=colors, s=40, alpha=1)
    plt.colorbar(best_N_scatter)
    plt.xlim(results['active_params_bounds_all'][0][i_x])
    plt.ylim(results['active_params_bounds_all'][0][i_y])
    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel(active_params_symbols[i_x], fontsize=16)
    plt.ylabel(active_params_symbols[i_y], fontsize=16)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_2D_summary_best_N.pdf')
else:
    plt.show()
#plt.close()
#'''





#'''
keep_ranked = [(100000, 10), (10000, 1)]
for i in range(len(keep_ranked)):
    N_best_save, keep_every = keep_ranked[i][0], keep_ranked[i][1]
    i_best_N = np.argsort(results['dtot_w_all'])[0:N_best_save:keep_every]

    plot_cornerpy_wrapper(active_params_symbols, results['active_params_all'][i_best_N], title_kwargs={'fontsize':20}, save_name=savefigures_directory + model_name + '_best%s_every%s_corner.pdf' % (N_best_save, keep_every), save_fig=savefigures)
#'''





##### To plot the individual distance terms (weighted and unweighted) from the model evaluations used to compute the weights:

N_best_save, keep_every = 1000, 1
i_best_N = np.argsort(results['dtot_w_all'])[0:N_best_save:keep_every]

N_panels = len(results['d_used_vals_w_all']['all'][0]) + 1
cols = int(np.ceil(np.sqrt(N_panels))) #number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols #number of rows, such that rows*cols >= N_panels

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)
for i,d_key in enumerate(results['d_used_vals_all']['all'].dtype.names):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])

    plt.hist([results['d_used_vals_all'][sample][d_key][i_best_N] for sample in sample_names], bins=50, histtype='step', color=sample_colors)
    plt.xlabel(d_key, fontsize=12)
    plt.ylabel('')
# For total distance of each sample:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist([results['dtot_samples_all'][sample][i_best_N] for sample in sample_names], bins=50, histtype='step', color=sample_colors)
plt.xlabel('Sum of distance terms', fontsize=12)
plt.ylabel('')
# For total distance overall: (still selecting best N by total weighted distance)
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist(results['dtot_all'][i_best_N], bins=50, histtype='step', linewidth=3, color='k')
plt.xlabel('Total sum of distance terms', fontsize=12)
plt.ylabel('')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_dists_best%s_every%s.pdf' % (N_best_save, keep_every))
else:
    plt.show()

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)
for i,d_key in enumerate(results['d_used_vals_w_all']['all'].dtype.names):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])

    plt.hist([results['d_used_vals_w_all'][sample][d_key][i_best_N] for sample in sample_names], bins=50, histtype='step', color=sample_colors)
    plt.xlabel(d_key, fontsize=12)
    plt.ylabel('')
# For total weighted distance of each sample:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist([results['dtot_w_samples_all'][sample][i_best_N] for sample in sample_names], bins=50, histtype='step', color=sample_colors)
plt.xlabel('Weighted sum of distance terms', fontsize=12)
plt.ylabel('')
# For total weighted distance overall:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist(results['dtot_w_all'][i_best_N], bins=50, histtype='step', linewidth=3, color='k')
plt.xlabel('Total weighted sum of distance terms', fontsize=12)
plt.ylabel('')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_dists_w_best%s_every%s.pdf' % (N_best_save, keep_every))
else:
    plt.show()
