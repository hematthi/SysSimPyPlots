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

run_directory = 'AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_equal/durations_norm_circ_singles_multis_GF2020_KS/'
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Model_Optimization/' + run_directory
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/' + run_directory

run_numbers = np.loadtxt(loadfiles_directory + 'run_numbers.txt', dtype='i4')

sample_names = ['all', 'bluer', 'redder']
sample_colors = ['k', 'b', 'r']

model_name = 'ExoplanetsSysSim_Clustered_Model_bboptimize'





##### To iterate through each of the optimization runs (files), and extract the results:

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
                active_params_bounds = [(float(x.split(', ')[0]), float(x.split(', ')[1])) for x in line[50:-3].split('), (')]
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
#active_params_bounds_all[0][9], active_params_bounds_all[0][10] = np.array([0., 90.]), np.array([0., 90.]) ##### FOR TRANSFORMED PARAMS
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

N_best = 100

##### To save the best parameter values for simulated catalog generation:
#'''
table_header = 'run_number ' + ' '.join(active_params_names_all[0])
fields_formats = ['%i'] + ['%1.6f']*len(active_params_names_all[0])

i_best_N = np.argsort(dtot_w_all)[:N_best]
active_params_table = np.concatenate((np.array([np.arange(N_best)]).transpose(), active_params_all[i_best_N]), axis=1)
np.savetxt('/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/' + run_directory + 'best_N/Active_params_table.txt', active_params_table, fmt=fields_formats, header=table_header, comments='')
#'''

##### To save the best parameter values and the distances for training a GP emulator:
#'''
N_best_save, keep_every = 100000, 10
i_best_N = np.argsort(dtot_w_all)[0:N_best_save:keep_every]
active_params_distances_table = np.concatenate((active_params_all[i_best_N], np.array([dtot_w_all[i_best_N]]).transpose()), axis=1)
table_header = ' '.join(active_params_names_all[0]) + ' dist_tot_weighted'
fields_formats = ['%1.6f']*len(active_params_names_all[0]) + ['%1.6f']
np.savetxt(loadfiles_directory + 'Active_params_distances_table_best%s_every%s.txt' % (N_best_save, keep_every), active_params_distances_table, fmt=fields_formats, header=table_header, comments='')
#'''

#sys.exit("Error message")





##### To make 2D plots of various pairs of parameters:

'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("power_law_P", "sigma_incl"),
                       ("sigma_incl", "sigma_hk"),
                       ("sigma_incl", "sigma_incl_near_mmr"),
                       ("f_high_incl", "power_law_P"),
                       ("f_high_incl", "sigma_incl"),
                       ("f_high_incl", "sigma_incl_near_mmr"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted")
                       ] # for some (12) active parameters (clustered model with fraction of stars with planets)
'''
'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("power_law_P", "sigma_incl"),
                       ("sigma_incl", "sigma_hk_at_med_color"),
                       ("sigma_hk_at_med_color", "sigma_hk_color_slope"),
                       ("sigma_incl", "sigma_incl_near_mmr"),
                       ("f_high_incl", "power_law_P"),
                       ("f_high_incl", "sigma_incl"),
                       ("f_high_incl", "sigma_incl_near_mmr"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted"),
                       ] # for some (13) active parameters (clustered model with slope for ecc vs bp_rp colour and normalization)
'''
'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("power_law_P", "sigma_incl"),
                       ("sigma_incl", "sigma_hk"),
                       ("sigma_incl", "sigma_incl_near_mmr"),
                       ("f_high_incl", "power_law_P"),
                       ("f_high_incl", "sigma_incl"),
                       ("f_high_incl", "sigma_incl_near_mmr"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted_bluer"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted_bluer"),
                       ("power_law_P", "f_stars_with_planets_attempted_bluer"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted_redder"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted_redder"),
                       ("power_law_P", "f_stars_with_planets_attempted_redder"),
                       ("f_stars_with_planets_attempted_bluer", "f_stars_with_planets_attempted_redder")
                       ] # for some (13) active parameters (clustered model with step function for the fraction of stars with planets (bluer and redder))
'''
'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("power_law_P", "sigma_incl"),
                       ("sigma_incl", "sigma_hk"),
                       ("sigma_incl", "sigma_incl_near_mmr"),
                       ("f_high_incl", "power_law_P"),
                       ("f_high_incl", "sigma_incl"),
                       ("f_high_incl", "sigma_incl_near_mmr"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted_at_med_color"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted_at_med_color"),
                       ("f_stars_with_planets_attempted_at_med_color", "f_stars_with_planets_attempted_color_slope")
                       ] # for some (13) active parameters (clustered model with slope for fraction of stars with planets vs bp_rp color and normalization)
'''
'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("power_law_P_at_med_color", "sigma_incl"),
                       ("sigma_incl", "sigma_hk"),
                       ("sigma_incl", "sigma_incl_near_mmr"),
                       ("f_high_incl", "power_law_P_at_med_color"),
                       ("f_high_incl", "sigma_incl"),
                       ("f_high_incl", "sigma_incl_near_mmr"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted"),
                       ("log_rate_clusters", "power_law_P_at_med_color"),
                       ("log_rate_planets_per_cluster", "power_law_P_at_med_color"),
                       ("power_law_P_at_med_color", "power_law_P_color_slope")
                       ] # for some (13) active parameters (clustered model with slope for period power-law vs bp_rp color and normalization)
'''
'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("power_law_P_at_med_color", "sigma_incl"),
                       ("sigma_incl", "sigma_hk"),
                       ("sigma_incl", "sigma_incl_near_mmr"),
                       ("f_high_incl", "power_law_P_at_med_color"),
                       ("f_high_incl", "sigma_incl"),
                       ("f_high_incl", "sigma_incl_near_mmr"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted_at_med_color"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted_at_med_color"),
                       ("f_stars_with_planets_attempted_at_med_color", "f_stars_with_planets_attempted_color_slope"),
                       ("log_rate_clusters", "power_law_P_at_med_color"),
                       ("log_rate_planets_per_cluster", "power_law_P_at_med_color"),
                       ("power_law_P_at_med_color", "power_law_P_color_slope"),
                       ("f_stars_with_planets_attempted_at_med_color", "power_law_P_at_med_color"),
                       ("f_stars_with_planets_attempted_color_slope", "power_law_P_color_slope")
                       ] # for some (14) active parameters (clustered model with slopes for fswp and for period power-law, vs bp_rp color and normalizations)
'''
#active_params_pairs = [("f_stars_with_planets_attempted_at_med_color", "f_stars_with_planets_attempted_color_slope")]
#active_params_pairs = [("sigma_hk_at_med_color", "sigma_hk_color_slope")]

'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("log_rate_clusters", "power_law_P"),
                       ("log_rate_planets_per_cluster", "power_law_P"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted_at_med_color"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted_at_med_color"),
                       ("f_stars_with_planets_attempted_at_med_color", "f_stars_with_planets_attempted_color_slope")
                       ] # for some (9) active parameters (AMD system clustered model with slope for fraction of stars with planets vs bp_rp color and normalization)
'''
'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("log_rate_clusters", "power_law_P"),
                       ("log_rate_planets_per_cluster", "power_law_P"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted_at_med_color"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted_at_med_color"),
                       ("f_stars_with_planets_attempted_at_med_color", "f_stars_with_planets_attempted_color_slope"),
                       ("num_mutual_hill_radii", "sigma_logperiod_per_pl_in_cluster"),
                       ("power_law_P", "num_mutual_hill_radii")
                       ] # for some (10) active parameters (AMD system clustered model with slope for fraction of stars with planets vs bp_rp color and normalization, and min mutual Hill separation)
'''
#'''
active_params_pairs = [("log_rate_clusters", "log_rate_planets_per_cluster"),
                       ("log_rate_planets_per_cluster", "sigma_logperiod_per_pl_in_cluster"),
                       ("log_rate_clusters", "power_law_P"),
                       ("log_rate_planets_per_cluster", "power_law_P"),
                       ("power_law_r1", "power_law_r2"),
                       ("sigma_logperiod_per_pl_in_cluster", "sigma_log_radius_in_cluster"),
                       ("log_rate_clusters", "f_stars_with_planets_attempted_at_med_color"),
                       ("log_rate_planets_per_cluster", "f_stars_with_planets_attempted_at_med_color"),
                       ("f_stars_with_planets_attempted_at_med_color", "f_stars_with_planets_attempted_color_slope"),
                       ("num_mutual_hill_radii", "sigma_logperiod_per_pl_in_cluster"),
                       ("power_law_P", "num_mutual_hill_radii"),
                       ("log_rate_clusters", "sigma_hk"),
                       ("log_rate_planets_per_cluster", "sigma_hk"),
                       ("f_stars_with_planets_attempted_at_med_color", "sigma_hk"),
                       #("f_amd_crit", "log_rate_clusters"), #
                       #("f_amd_crit", "log_rate_planets_per_cluster"), #
                       #("num_mutual_hill_radii", "f_amd_crit"), #
                       ] # for some (11) active parameters (AMD system clustered model with separate eccentricity distribution for singles, and with slope for fraction of stars with planets vs bp_rp color and normalization, and min mutual Hill separation) or (12) (f_amd_crit)
#'''



#'''
#To plot the total weighted distances for all the 2D plots on one figure:
N_panels = len(active_params_pairs)
cols = int(np.ceil(np.sqrt(N_panels))) #number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols #number of rows, such that rows*cols >= N_panels

#To plot the best total weighted distance found by each run:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.075,bottom=0.115,right=0.95,top=0.925,wspace=0.3,hspace=0.4)
for i,pair in enumerate(active_params_pairs):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])
    
    i_x, i_y = np.where(np.array(active_params_names) == pair[1])[0][0], np.where(np.array(active_params_names) == pair[0])[0][0]
    #print(pair, ': (ix,iy) = (%s,%s) ; x=%s, y=%s' % (i_x, i_y, active_params_symbols[i_x], active_params_symbols[i_y]))
    
    active_params_plot = active_params_best_all
    colors = dtot_best_all
    best_scatter = plt.scatter(active_params_plot[:,i_x], active_params_plot[:,i_y], marker='.', c=colors, s=200, alpha=1) #best values for each run
    plt.colorbar(best_scatter)
    
    plt.xlim(active_params_bounds_all[0][i_x])
    plt.ylim(active_params_bounds_all[0][i_y])
    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel(active_params_symbols[i_x], fontsize=20)
    plt.ylabel(active_params_symbols[i_y], fontsize=20)
    #plt.legend(loc='center left', bbox_to_anchor=(1.,0.5), ncol=1, fontsize=12)

if savefigures == True:
    plt.savefig(savefigures_directory + model_name + '_2D_summary_best_per_run.pdf')
else:
    plt.show()
#plt.close()

#To plot the best N total weighted distances found by all the runs in aggregate:
N_best = 100

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.075,bottom=0.115,right=0.95,top=0.925,wspace=0.3,hspace=0.4)
for i,pair in enumerate(active_params_pairs):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])
    
    i_x, i_y = np.where(np.array(active_params_names) == pair[1])[0][0], np.where(np.array(active_params_names) == pair[0])[0][0]
    
    i_best_N = np.argsort(dtot_w_all)[:N_best]
    active_params_plot = active_params_all[i_best_N]
    colors = dtot_w_all[i_best_N]
    best_N_scatter = plt.scatter(active_params_plot[:,i_x], active_params_plot[:,i_y], marker=',', c=colors, s=40, alpha=1)
    plt.colorbar(best_N_scatter)
    plt.xlim(active_params_bounds_all[0][i_x])
    plt.ylim(active_params_bounds_all[0][i_y])
    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel(active_params_symbols[i_x], fontsize=20)
    plt.ylabel(active_params_symbols[i_y], fontsize=20)

if savefigures == True:
    plt.savefig(savefigures_directory + model_name + '_2D_summary_best_N.pdf')
else:
    plt.show()
#plt.close()
#'''





#'''
keep_ranked = [(100000, 10), (10000, 1)]
for i in range(len(keep_ranked)):
    N_best_save, keep_every = keep_ranked[i][0], keep_ranked[i][1]
    i_best_N = np.argsort(dtot_w_all)[0:N_best_save:keep_every]

    plot_cornerpy_wrapper(active_params_symbols, active_params_all[i_best_N], title_kwargs={'fontsize':20}, save_name=savefigures_directory + model_name + '_best%s_every%s_corner.pdf' % (N_best_save, keep_every), save_fig=savefigures)
#'''





##### To plot the individual distance terms (weighted and unweighted) from the model evaluations used to compute the weights:

N_best_save, keep_every = 1000, 1
i_best_N = np.argsort(dtot_w_all)[0:N_best_save:keep_every]

N_panels = len(d_used_keys) + 1
cols = int(np.ceil(np.sqrt(N_panels))) #number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols #number of rows, such that rows*cols >= N_panels

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)
for i,d_key in enumerate(d_used_vals_all[sample].dtype.names):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])
    
    plt.hist([d_used_vals_all[sample][d_key][i_best_N] for sample in sample_names], bins=50, histtype='step', color=sample_colors)
    plt.xlabel(d_key, fontsize=12)
    plt.ylabel('')
# For total distance of each sample:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist([dtot_samples_all[sample][i_best_N] for sample in sample_names], bins=50, histtype='step', color=sample_colors)
plt.xlabel('Sum of distance terms', fontsize=12)
plt.ylabel('')
# For total distance overall: (still selecting best N by total weighted distance)
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist(dtot_all[i_best_N], bins=50, histtype='step', linewidth=3, color='k')
plt.xlabel('Total sum of distance terms', fontsize=12)
plt.ylabel('')
if savefigures == True:
    plt.savefig(savefigures_directory + model_name + '_dists_best%s_every%s.pdf' % (N_best_save, keep_every))
else:
    plt.show()

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)
for i,d_key in enumerate(d_used_vals_w_all[sample].dtype.names):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])
    
    plt.hist([d_used_vals_w_all[sample][d_key][i_best_N] for sample in sample_names], bins=50, histtype='step', color=sample_colors)
    plt.xlabel(d_key, fontsize=12)
    plt.ylabel('')
# For total weighted distance of each sample:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist([dtot_w_samples_all[sample][i_best_N] for sample in sample_names], bins=50, histtype='step', color=sample_colors)
plt.xlabel('Weighted sum of distance terms', fontsize=12)
plt.ylabel('')
# For total weighted distance overall:
i += 1
i_row, i_col = i//cols, i%cols
ax = plt.subplot(plot[i_row,i_col])
plt.hist(dtot_w_all[i_best_N], bins=50, histtype='step', linewidth=3, color='k')
plt.xlabel('Total weighted sum of distance terms', fontsize=12)
plt.ylabel('')
if savefigures == True:
    plt.savefig(savefigures_directory + model_name + '_dists_w_best%s_every%s.pdf' % (N_best_save, keep_every))
else:
    plt.show()
