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

#run_directory = 'Hybrid_NR20_AMD_model1/Fit_all_KS/Params12/'
#loadfiles_directory = '/Users/hematthi/Documents/NotreDame_Postdoc/CRC/Files/SysSim/Model_Optimization/' + run_directory
run_directory = 'Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/'
loadfiles_directory = '/Users/hematthi/Documents/NPP_ARC_Modernize_Kepler/Personal_research/SysSim/Model_Optimization/' + run_directory
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Model_Optimization/' + run_directory

run_numbers = range(1,51) #range(51,101) #range(1,51) #np.loadtxt(loadfiles_directory + 'run_numbers.txt', dtype='i4')

model_name = 'Hybrid_NR20_AMD_model1'





##### To iterate through each of the optimization runs (files), and extract the results:

active_params_symbols = [#r'$M_{\rm break,1}$',
                         r'$\ln{(\lambda_c)}$',
                         r'$\ln{(\lambda_p)}$',
                         r'$\ln{(\alpha_{\rm ret})}$',
                         r'$\mu_M$',
                         r'$R_{p,\rm norm}$',
                         r'$\alpha_P$',
                         r'$\gamma_0$',
                         #r'$\gamma_1$',
                         r'$\sigma_0$',
                         #r'$\sigma_1$',
                         r'$\sigma_M$',
                         r'$\sigma_{M,\rm cluster}$',
                         #r'$\sigma_P$',
                         ] # this list of parameter symbols must match the order of parameters in 'active_params_names'!

results = analyze_bboptimize_runs(loadfiles_directory, run_numbers=run_numbers)

N_runs = len(run_numbers)
N_params = len(active_params_symbols)
N_evals = max([len(x) for x in results['dtot_w_runs']])

##### To save the best parameter values for simulated catalog generation:
'''
N_best = 100

table_header = 'run_number ' + ' '.join(results['active_params_names_all'][0])
fields_formats = ['%i'] + ['%1.6f']*len(results['active_params_names_all'][0])

i_best_N = np.argsort(dtot_w_all)[:N_best]
active_params_table = np.concatenate((np.array([np.arange(N_best)]).transpose(), results['active_params_all'][i_best_N]), axis=1)
np.savetxt('/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/' + run_directory + 'best_N/Active_params_table.txt', active_params_table, fmt=fields_formats, header=table_header, comments='')
'''

##### To save the best parameter values and the distances for training a GP emulator:
#'''
#N_best_save, keep_every = 100000, 10
N_best_save, keep_every = 10000, 1 # NOTE: may want to use this if many runs stop early due to reaching the target distance
i_best_N = np.argsort(results['dtot_w_all'])[0:N_best_save:keep_every]
active_params_distances_table = np.concatenate((results['active_params_all'][i_best_N], np.array([results['dtot_w_all'][i_best_N]]).transpose()), axis=1)
table_header = ' '.join(results['active_params_names_all'][0]) + ' dist_tot_weighted'
fields_formats = ['%1.6f']*len(results['active_params_names_all'][0]) + ['%1.6f']
np.savetxt(loadfiles_directory + 'Active_params_distances_table_best%s_every%s.txt' % (N_best_save, keep_every), active_params_distances_table, fmt=fields_formats, header=table_header, comments='')
#'''

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
active_params_pairs = [("break1_mass", "mean_ln_mass"),
                       ("break1_mass", "norm_radius"),
                       ("break1_mass", "power_law_γ0"),
                       ("break1_mass", "power_law_γ1"),
                       ("mean_ln_mass", "sigma_ln_mass"),
                       ("mean_ln_mass", "norm_radius"),
                       ("norm_radius", "power_law_γ0"),
                       ("norm_radius", "power_law_γ1"),
                       ("power_law_γ0", "power_law_σ0"),
                       ("power_law_γ1", "power_law_σ1"),
                       ("power_law_γ0", "power_law_γ1"),
                       ("power_law_σ0", "power_law_σ1"),
                       ] # for Hybrid1 model with 8 active parameters
#'''
#'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_α_pret", "mean_ln_mass"),
                       ("log_α_pret", "sigma_ln_mass"),
                       ("log_α_pret", "norm_radius"),
                       ("log_α_pret", "power_law_γ0"),
                       ("mean_ln_mass", "log_rate_clusters"),
                       ("mean_ln_mass", "log_rate_planets_per_cluster"),
                       ("mean_ln_mass", "sigma_ln_mass"),
                       ("mean_ln_mass", "norm_radius"),
                       ("norm_radius", "power_law_γ0"),
                       ("norm_radius", "power_law_σ0"),
                       ("power_law_γ0", "power_law_σ0"),
                       ("sigma_ln_mass", "power_law_σ0"),
                       ] # for Hybrid1 model with 8 active parameters (fix break mass and power-law above, free lambdas and alpha_pret)
#'''
#'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_clusters", "power_law_P"),
                       ("log_rate_planets_per_cluster", "power_law_P"),
                       ("power_law_P", "log_α_pret"),
                       ("log_α_pret", "mean_ln_mass"),
                       ("log_α_pret", "sigma_ln_mass"),
                       ("log_α_pret", "norm_radius"),
                       ("log_α_pret", "power_law_γ0"),
                       ("mean_ln_mass", "log_rate_clusters"),
                       ("mean_ln_mass", "log_rate_planets_per_cluster"),
                       ("mean_ln_mass", "sigma_ln_mass"),
                       ("mean_ln_mass", "norm_radius"),
                       ("norm_radius", "power_law_γ0"),
                       ("norm_radius", "power_law_σ0"),
                       ("power_law_γ0", "power_law_σ0"),
                       ("sigma_ln_mass", "power_law_σ0"),
                       ] # for Hybrid1 model with 9 active parameters (fix break mass and power-law above, free lambdas, alpha_pret, and alpha_P)
#'''
#'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_clusters", "power_law_P"),
                       ("log_rate_planets_per_cluster", "power_law_P"),
                       ("power_law_P", "log_α_pret"),
                       ("log_α_pret", "mean_ln_mass"),
                       ("log_α_pret", "sigma_ln_mass"),
                       ("log_α_pret", "norm_radius"),
                       ("log_α_pret", "power_law_γ0"),
                       ("mean_ln_mass", "log_rate_clusters"),
                       ("mean_ln_mass", "log_rate_planets_per_cluster"),
                       ("mean_ln_mass", "sigma_ln_mass"),
                       ("mean_ln_mass", "norm_radius"),
                       ("norm_radius", "power_law_γ0"),
                       ("norm_radius", "power_law_σ0"),
                       ("power_law_γ0", "power_law_σ0"),
                       ("sigma_ln_mass", "power_law_σ0"),
                       ("sigma_ln_mass_in_cluster", "log_rate_clusters"),
                       ("sigma_ln_mass_in_cluster", "mean_ln_mass"),
                       ("sigma_ln_mass_in_cluster", "sigma_ln_mass"),
                       ("sigma_ln_mass_in_cluster", "norm_radius"),
                       ] # for Hybrid1 clustered initial masses model with 10 active parameters (fix break mass and power-law above, free lambdas, alpha_pret, and alpha_P)
#'''



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
    plt.savefig(savefigures_directory + model_name + '_params_2D_best_per_run.pdf')
    plt.close()

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
    plt.savefig(savefigures_directory + model_name + '_params_2D_best%s.pdf' % N_best)
    plt.close()
plt.show()
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

N_panels = len(results['d_used_vals_w_all'][0]) + 1
cols = int(np.ceil(np.sqrt(N_panels))) #number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols #number of rows, such that rows*cols >= N_panels

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)
for i,d_key in enumerate(results['d_used_vals_all'].dtype.names):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])

    plt.hist([results['d_used_vals_all'][d_key][i_best_N]], bins=50, histtype='step', color='k')
    plt.xlabel(d_key, fontsize=12)
    plt.ylabel('')
# For total distance overall: (still selecting best N by total weighted distance)
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist(results['dtot_all'][i_best_N], bins=50, histtype='step', linewidth=3, color='k')
plt.xlabel('Sum of distance terms', fontsize=12)
plt.ylabel('')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_dists_best%s_every%s.pdf' % (N_best_save, keep_every))
    plt.close()

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)
for i,d_key in enumerate(results['d_used_vals_w_all'].dtype.names):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])

    plt.hist([results['d_used_vals_w_all'][d_key][i_best_N]], bins=50, histtype='step', color='k')
    plt.xlabel(d_key, fontsize=12)
    plt.ylabel('')
# For total weighted distance overall:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist(results['dtot_w_all'][i_best_N], bins=50, histtype='step', linewidth=3, color='k')
plt.xlabel('Weighted sum of distance terms', fontsize=12)
plt.ylabel('')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_dists_w_best%s_every%s.pdf' % (N_best_save, keep_every))
    plt.close()
plt.show()





##### To plot the weighted distances over time (i.e. as a function of number of evaluations) for each run:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(2, 1, left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0.2)

# Plot the total weighted distance at each evaluation:
ax = plt.subplot(plot[0,0])
for i_run in range(N_runs):
    dtot_w_evals = results['dtot_w_runs'][i_run]
    plt.plot(range(len(dtot_w_evals)), dtot_w_evals, lw=1)
plt.xlim([0, N_evals])
plt.ylim([0., 100.])
ax.tick_params(axis='both', right=True, labelright=True, labelsize=12)
plt.ylabel('Distance function (total weighted distance)', fontsize=12)

# Plot the best total weighted distance found up to each evaluation:
ax = plt.subplot(plot[1,0])
for i_run in range(N_runs):
    dtot_w_evals = results['dtot_w_runs'][i_run]
    dtot_w_best_uptoeval = np.array([np.nanmin(dtot_w_evals[:i+1]) for i in range(len(dtot_w_evals))])
    plt.plot(range(len(dtot_w_evals)), dtot_w_best_uptoeval, lw=1)
plt.xlim([0, N_evals])
plt.ylim([0., 50.])
ax.tick_params(axis='both', right=True, labelright=True, labelsize=12)
plt.xlabel('Evaluation number', fontsize=12)
plt.ylabel('Best total weighted distance found', fontsize=12)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_dtot_w_vs_evaluations_per_run.pdf')
    plt.close()
plt.show()





##### To plot the parameters over time (i.e. as a function of number of evaluations) for each run:

# Parameters at all evaluations:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(N_params, 1, left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)
for i_param in range(N_params):
    ax = plt.subplot(plot[i_param,0])
    for i_run in range(N_runs):
        param_evals = results['active_params_runs'][i_run][:, i_param]
        plt.plot(range(len(param_evals)), param_evals, lw=1)
    plt.xlim([0, N_evals])
    ax.tick_params(axis='both', right=True, labelright=True, labelsize=12)
    if i_param == N_params-1:
        plt.xlabel('Evaluation number', fontsize=12)
    else:
        plt.xticks([])
    plt.ylabel(active_params_symbols[i_param], fontsize=12)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_params_vs_evaluations_per_run.pdf')
    plt.close()
plt.show()

# Parameters at the best distances found up to each evaluation:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(N_params,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
for i_param in range(N_params):
    ax = plt.subplot(plot[i_param,0])
    for i_run in range(N_runs):
        dtot_w_evals = results['dtot_w_runs'][i_run]
        i_evals_best_uptoeval = np.array([np.nanargmin(dtot_w_evals[:i+1]) if not all(np.isnan(dtot_w_evals[:i+1])) else 0 for i in range(len(dtot_w_evals))])
        
        param_best_uptoeval = results['active_params_runs'][i_run][i_evals_best_uptoeval, i_param]
        plt.plot(range(len(param_best_uptoeval)), param_best_uptoeval, lw=1)
    plt.xlim([0, N_evals])
    ax.tick_params(axis='both', right=True, labelright=True, labelsize=12)
    if i_param == N_params-1:
        plt.xlabel('Evaluation number', fontsize=12)
    else:
        plt.xticks([])
    plt.ylabel(active_params_symbols[i_param], fontsize=12)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_params_best_vs_evaluations_per_run.pdf')
    plt.close()
plt.show()



##### To plot histograms of the parameters for distances below various thresholds:

dtot_cuts = [30., 18., 12.] #[20., 15., 10.] #[30., 25., 22.]
i_evals_all_pass_cuts = [results['dtot_w_all'] <= dtot for dtot in dtot_cuts]
for i,dtot in enumerate(dtot_cuts):
    i_evals_all_pass = i_evals_all_pass_cuts[i]
    print(f'Number of evaluations passing distance cut of {dtot}:', np.sum(i_evals_all_pass))

N_panels = N_params
cols = int(np.ceil(np.sqrt(N_panels)))
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows, cols, left=0.05, bottom=0.1, right=0.975, top=0.95, wspace=0.3, hspace=0.5)
for i_param in range(N_params):
    param_bounds = results['active_params_bounds_all'][0, i_param]
    
    i_row, i_col = i_param//cols, i_param%cols
    ax = plt.subplot(plot[i_row,i_col])
    for i,dtot in enumerate(dtot_cuts):
        i_evals_all_pass = i_evals_all_pass_cuts[i]
        params_pass = results['active_params_all'][i_evals_all_pass, i_param]
        
        plt.hist(params_pass, bins=np.linspace(param_bounds[0], param_bounds[1], 101), density=True, histtype='step', label=f'[{len(params_pass)} evals]' + r'  $\mathcal{D}_{\rm tot} \leq$' + f'{dtot:.0f}') # weights=np.ones(len(params_pass))/len(params_pass)
    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel(active_params_symbols[i_param], fontsize=12)
    plt.ylabel('Evals (normalized)', fontsize=12)
    if i_param==0:
        plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, fontsize=12)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_params_dtot_w_cuts.pdf')
    plt.close()
plt.show()

