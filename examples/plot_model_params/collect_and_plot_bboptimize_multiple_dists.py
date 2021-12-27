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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *
from src.functions_plot_catalogs import *
from src.functions_plot_params import *





##### This module will be used to plot results of the optimization runs of our clustered model using bboptimize:

savefigures = False
plt.ioff()

savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Distances_compare/bboptimize/'
save_name = 'Models_Compare_bboptimize'

sample_names = ['all', 'bluer', 'redder']
sample_colors = ['k', 'b', 'r']
samples = len(sample_names)





##### To iterate through each of the optimization runs (files), and extract the results:

def analyze_bboptimize_runs(loadfiles_directory, run_numbers):
    
    sample_names = ['all', 'bluer', 'redder']

    active_params_names_all = [] # list to be filled with arrays of the names of the active parameters for each run (should be the same for all runs)
    active_params_bounds_all = [] # list to be filled with arrays of the search bounds of the active parameters for each run (should be the same for all runs)
    active_params_start_all = []
    steps_best_weighted_all = [] # list to be filled with the number of model iterations to find the best active parameter values (lowest total weighted distance) for each run
    steps_tot_all = [] # list to be filled with the number of total model iterations in the optimization procedure, for each run
    time_optimization_all = [] # list to be filled with the elapsed times (s) for the full optimization procedure, for each run

    active_params_runs = [] # list to be filled with 2D array of the values of all the active parameters at every step for each run
    active_params_all = [] # list to be filled with arrays of the values of all the active parameters at every step (excluding starting values but including best values) for all the runs
    d_used_keys_runs = {key: [] for key in sample_names}
    d_used_vals_runs = {key: [] for key in sample_names}
    d_used_vals_all = {key: [] for key in sample_names}
    d_used_vals_w_runs = {key: [] for key in sample_names}
    d_used_vals_w_all = {key: [] for key in sample_names}

    runs_started = 0
    runs_finished = 0
    for i in run_numbers:
        with open(loadfiles_directory + 'Clustered_P_R_optimization_random%s_targs86760_evals5000.txt' % i, 'r') as file:
            
            optim_lines = False # set to true once we start reading lines in the file that are the outputs of the optimization
            active_params_start = [] # will be replaced by the actual active parameter values if the file is not empty
            
            active_params_run = [] # will be filled with all the active params at each step of the optimization in this run
            d_used_keys_run = {key: [] for key in sample_names}
            d_used_vals_run = {key: [] for key in sample_names} # will be filled with all the distances at each step of the optimization in this run
            d_used_vals_w_run = {key: [] for key in sample_names} # will be filled with all the weighted distances at each step of the optimization in this run
            
            best_fitness = np.inf # will be replaced with the best total weighted distance if the optimization progressed
            steps = 0 # will be a running count of the number of model iterations
            steps_best_weighted = steps # will be replaced by the number of the model iteration at which the best total weighted distance was found
            for line in file:
                # For recording the preliminary runs of the model before optimizations:
                if line[0:19] == '# Active parameters':
                    active_params_names = line[23:-3].split('", "')
                    active_params_names_all.append(active_params_names)

                # For recording the results of the optimizations:
                elif line[0:7] == '# Start':
                    active_params_start = [float(x) for x in line[37:-2].split(', ')]
                elif line[0:7] == '# Optim':
                    runs_started += 1
                    active_params_bounds = [(float(x.split(', ')[0]), float(x.split(', ')[1])) for x in line[72:-3].split('), (')]
                    optim_lines = True
                elif line[0:13] == 'Active_params' and optim_lines:
                    steps += 1
                    active_params = [float(x) for x in line[16:-2].split(', ')]
                    active_params_run.append(active_params)
                    active_params_all.append(active_params)
                elif line[0:12] == 'Total_dist_w' and optim_lines:
                    total_dist_w = float(line[15:-2])
                    if total_dist_w < best_fitness:
                        best_fitness = total_dist_w
                        steps_best_weighted = steps
                
                for sample in sample_names:
                    n = len(sample)
                    if line[0:n+2] == '[%s]' % sample and optim_lines:
                        if line[n+3:n+3+6] == 'Counts':
                            Nmult_str, counts_str = line[n+3+9:-2].split('][')
                            Nmult = [int(x) for x in Nmult_str.split(', ')]
                        
                        elif line[n+3:n+3+12] == 'd_used_keys:':
                            d_used_keys = line[n+3+15:-3].split('", "')
                            d_used_keys_run[sample].append(d_used_keys)
                        
                        elif line[n+3:n+3+12] == 'd_used_vals:':
                            d_used_vals_str, d_used_vals_tot_str = line[n+3+14:-2].split('][')
                            d_used_vals = [float(x) for x in d_used_vals_str.split(', ')]
                            d_used_vals_run[sample].append(d_used_vals)
                            d_used_vals_all[sample].append(tuple(d_used_vals))
                        
                        elif line[n+3:n+3+13] == 'd_used_vals_w':
                            d_used_vals_w_str, d_used_vals_tot_w_str = line[n+3+16:-2].split('][')
                            d_used_vals_w = [float(x) for x in d_used_vals_w_str.split(', ')]
                            d_used_vals_tot_w = float(d_used_vals_tot_w_str)
                            d_used_vals_w_run[sample].append(d_used_vals_w)
                            d_used_vals_w_all[sample].append(tuple(d_used_vals_w))

                if line[0:14] == '# best_fitness':
                    runs_finished += 1
                    best_fitness_end = float(line[16:-1])
                elif line[0:9] == '# elapsed' and optim_lines:
                    time_optimization_all.append(float(line[16:-8]))
        
            print(i, best_fitness, len(active_params_all), [len(d_used_vals_w_all[key]) for key in d_used_vals_w_all])

            active_params_bounds_all.append(active_params_bounds)
            active_params_start_all.append(active_params_start)
            
            active_params_runs.append(active_params_run)
            for sample in sample_names:
                d_used_keys_runs[sample].append(d_used_keys_run[sample])
                d_used_vals_runs[sample].append(d_used_vals_run[sample])
                d_used_vals_w_runs[sample].append(d_used_vals_w_run[sample])

            steps_best_weighted_all.append(steps_best_weighted)
            steps_tot_all.append(steps)

    print('Runs successfully started (and not killed): ', runs_started) # runs killed because of the wall time are not counted here because they have their output files emptied
    print('Runs successfully finished (reached max iterations or target fitness): ', runs_finished) # runs not counted here are ones killed either because of the wall time, or because of bus error

    active_params_names_all = np.array(active_params_names_all)
    active_params_bounds_all = np.array(active_params_bounds_all)
    active_params_bounds_all[0][9], active_params_bounds_all[0][10] = np.array([0., 90.]), np.array([0., 90.]) ##### FOR TRANSFORMED PARAMS
    active_params_start_all = np.array(active_params_start_all)

    steps_best_weighted_all = np.array(steps_best_weighted_all)
    steps_tot_all = np.array(steps_tot_all)
    time_optimization_all = np.array(time_optimization_all)

    active_params_runs = np.array(active_params_runs)
    active_params_all = np.array(active_params_all)

    for sample in sample_names:
        d_used_keys_runs[sample] = np.array(d_used_keys_runs[sample])
        d_used_vals_runs[sample] = np.array(d_used_vals_runs[sample])
        d_used_vals_all[sample] = np.array(d_used_vals_all[sample], dtype=[(d_key, 'f8') for d_key in d_used_keys_runs[sample][0][0]])
        d_used_vals_w_runs[sample] = np.array(d_used_vals_w_runs[sample])
        d_used_vals_w_all[sample] = np.array(d_used_vals_w_all[sample], dtype=[(d_key, 'f8') for d_key in d_used_keys_runs[sample][0][0]])

    # To compute the sums of weighted distances per iteration, for each sample:
    dtot_samples_runs = {}
    dtot_samples_all = {}
    dtot_w_samples_runs = {}
    dtot_w_samples_all = {}
    for sample in sample_names:
        dtot_samples_runs[sample] = np.array([np.sum(run_2d, axis=1) for run_2d in d_used_vals_runs[sample]]) # will be a 2D array of size (runs, steps)
        dtot_samples_all[sample] = np.array([sum(x) for x in d_used_vals_all[sample]]) # will be a 1D array of length runs*steps
        dtot_w_samples_runs[sample] = np.array([np.sum(run_2d, axis=1) for run_2d in d_used_vals_w_runs[sample]]) # will be a 2D array of size (runs, steps)
        dtot_w_samples_all[sample] = np.array([sum(x) for x in d_used_vals_w_all[sample]]) #np.sum(d_used_vals_w_all[sample], axis=1) # will be a 1D array of length runs*steps

    dtot_runs = sum(dtot_samples_runs[sample] for sample in sample_names) # 2D
    dtot_all = sum(dtot_samples_all[sample] for sample in sample_names) # 1D
    dtot_w_runs = sum(dtot_w_samples_runs[sample] for sample in sample_names) # 2D
    dtot_w_all = sum(dtot_w_samples_all[sample] for sample in sample_names) # 1D

    active_params_best_all = np.array([active_params_runs[n][np.argsort(dtot_w_runs[n])[0]] for n in range(len(run_numbers))])
    dtot_best_all = np.array([dtot_w_runs[n][np.argsort(dtot_w_runs[n])[0]] for n in range(len(run_numbers))])

    results = {}
    #results['d_used_keys_runs'] = d_used_keys_runs
    results['d_used_vals_w_all'] = d_used_vals_w_all
    results['dtot_w_samples_all'] = dtot_w_samples_all
    results['dtot_w_all'] = dtot_w_all
    results['i_sort_dtot_w_all'] = np.argsort(dtot_w_all) # indices that would sort all the model evaluations by the total weighted distance
    results['active_params_all'] = active_params_all
    return results




##### To analyze multiple bboptimize runs:

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_KS/Old_mr_relation/'
run_numbers1 = np.loadtxt(loadfiles_directory1 + 'run_numbers.txt', dtype='i4')

loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_KS/Old_mr_relation/'
run_numbers2 = np.loadtxt(loadfiles_directory1 + 'run_numbers2.txt', dtype='i4')

results1 = analyze_bboptimize_runs(loadfiles_directory1, run_numbers1)
results2 = analyze_bboptimize_runs(loadfiles_directory2, run_numbers2)

model_results = [results1, results2]
model_names = ['50 runs', '40 runs'] #[r'Linear $f_{\rm swpa}$', r'Linear $\alpha_P$']
model_savenames = ['50runs', '40runs']
model_linestyles = ['-', '--']
models = len(model_results)





##### To plot the individual distance terms (weighted and unweighted) from the model evaluations used to compute the weights:
##### NOTE: all runs must have the same distance terms for the following plots!

n_bins = 50
lw = 1 #linewidth
alpha = 0.2 #transparency of histograms

afs = 12 #axes labels font size
tfs = 12 #text labels font size
lfs = 12 #legend labels font size

N_best, keep_every = 1000, 1
N_panels = len(results1['d_used_vals_w_all']['all'].dtype.names) + 1
cols = int(np.ceil(np.sqrt(N_panels))) #number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols #number of rows, such that rows*cols >= N_panels


'''
fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)

ls_sim = []
labels_sim = []
for j in range(len(model_results)):
    ls_sim += [model_linestyles[j]]*samples
    labels_sim += [model_names[j],None,None]

for i,d_key in enumerate(results1['d_used_vals_w_all']['all'].dtype.names):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])
    
    dw = []
    for j,results in enumerate(model_results):
        for sample in sample_names:
            isort_keep = results['i_sort_dtot_w_all'][0:N_best:keep_every]
            dw.append(results['d_used_vals_w_all'][sample][d_key][isort_keep])
    plot_panel_pdf_simple(ax, dw, [], n_bins=n_bins, normalize=False, c_sim=sample_colors*models, ls_sim=ls_sim, lw=lw, labels_sim=[None]*models*samples, xlabel_text=d_key, ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

# For total weighted distance of each sample:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
dw = []
for j,results in enumerate(model_results):
    for sample in sample_names:
        isort_keep = results['i_sort_dtot_w_all'][0:N_best:keep_every]
        dw.append(results['dtot_w_samples_all'][sample][isort_keep])
plot_panel_pdf_simple(ax, dw, [], n_bins=n_bins, normalize=False, c_sim=sample_colors*models, ls_sim=ls_sim, lw=lw, labels_sim=labels_sim, xlabel_text='Weighted sum of distance terms', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=False)
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=lfs)

# For total weighted distance overall:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plot_panel_pdf_simple(ax, [results['dtot_w_all'][results['i_sort_dtot_w_all'][0:N_best:keep_every]] for results in model_results], [], n_bins=n_bins, normalize=False, c_sim=['k']*models, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text='Total weighted sum of distance terms', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=False)
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=lfs)

if savefigures == True:
    plt.savefig(savefigures_directory + save_name + '_' + '_'.join(model_savenames) + '_dists_w_best%s_every%s.pdf' % (N_best, keep_every))
else:
    plt.show()
'''





#'''
model_N_best, model_keep_every = [100000, 80000], [10, 8]
N_panels = len(results1['d_used_vals_w_all']['all'].dtype.names) + 1
cols = int(np.ceil(np.sqrt(N_panels))) #number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols #number of rows, such that rows*cols >= N_panels



fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)

ls_sim = []
labels_sim = []
for j in range(len(model_results)):
    ls_sim += [model_linestyles[j]]*samples
    labels_sim += [model_names[j],None,None]

for i,d_key in enumerate(results1['d_used_vals_w_all']['all'].dtype.names):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])
    
    dw = []
    for j,results in enumerate(model_results):
        for sample in sample_names:
            isort_keep = results['i_sort_dtot_w_all'][0:model_N_best[j]:model_keep_every[j]]
            dw.append(results['d_used_vals_w_all'][sample][d_key][isort_keep])
    plot_panel_pdf_simple(ax, dw, [], n_bins=n_bins, normalize=False, c_sim=sample_colors*models, ls_sim=ls_sim, lw=lw, labels_sim=[None]*models*samples, xlabel_text=d_key, ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

# For total weighted distance of each sample:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
dw = []
for j,results in enumerate(model_results):
    for sample in sample_names:
        isort_keep = results['i_sort_dtot_w_all'][0:model_N_best[j]:model_keep_every[j]]
        dw.append(results['dtot_w_samples_all'][sample][isort_keep])
plot_panel_pdf_simple(ax, dw, [], n_bins=n_bins, normalize=False, c_sim=sample_colors*models, ls_sim=ls_sim, lw=lw, labels_sim=labels_sim, xlabel_text='Weighted sum of distance terms', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=False)
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=lfs)

# For total weighted distance overall:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plot_panel_pdf_simple(ax, [model_results[j]['dtot_w_all'][model_results[j]['i_sort_dtot_w_all'][0:model_N_best[j]:model_keep_every[j]]] for j in range(models)], [], n_bins=n_bins, normalize=False, c_sim=['k']*models, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text='Total weighted sum of distance terms', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=False)
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=lfs)

if savefigures == True:
    plt.savefig(savefigures_directory + save_name + '_' + '_'.join(model_savenames) + '_dists_w.pdf')
else:
    plt.show()



active_params_symbols = [#r'$f_{\rm crit}$', #####
                         #r'$f_{\sigma_{i,\rm high}}$',
                         #r'$f_{\rm swpa}$',
                         #r'$f_{\rm swpa,bluer}$',
                         #r'$f_{\rm swpa,redder}$',
                         r'$f_{\rm swpa,med}$',
                         r'$d(f_{\rm swpa})/d(b_p-r_p)$',
                         r'$\ln{(\lambda_c)}$',
                         r'$\ln{(\lambda_p)}$',
                         r'$\Delta_c$',
                         r'$\alpha_P$',
                         #r'$\alpha_{P,\rm med}$',
                         #r'$d\alpha_P/d(b_p-r_p)$',
                         r'$\alpha_{R1}$',
                         r'$\alpha_{R2}$',
                         r'$\sigma_e$',
                         #r'$\sigma_e(0.95)$',
                         #r'$d(\sigma_e)/d(b_p-r_p)$',
                         #r'$\sigma_{i,\rm high}$',
                         #r'$\sigma_{i,\rm low}$',
                         r'$\sigma_R$',
                         r'$\sigma_N$'
                         ] # this list of parameter symbols must match the order of parameters in 'active_params_names'!

for j,results in enumerate(model_results):
    N_best_save, keep_every = model_N_best[j], model_keep_every[j]
    i_best_N = np.argsort(results['dtot_w_all'])[0:N_best_save:keep_every]
    
    plot_cornerpy_wrapper(active_params_symbols, results['active_params_all'][i_best_N], title_kwargs={'fontsize':20}, save_name=savefigures_directory + model_savenames[j] + '_best%s_every%s_corner.pdf' % (N_best_save, keep_every), save_fig=savefigures)
#'''
