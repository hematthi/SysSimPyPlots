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

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *
from syssimpyplots.compute_RVs import *





##### This module will be used to plot results of the optimization runs of our clustered model using bboptimize:

savefigures = False
plt.ioff()

run_directory = 'AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_equal/durations_norm_circ_singles_multis_GF2020_KS/'
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Model_Optimization/' + run_directory + 'GP_files/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/' + run_directory #+ 'New_terms/'

sample_names = ['all', 'bluer', 'redder']
sample_colors = ['k', 'b', 'r']

model_name = 'Clustered_P_R_fswp_bprp_AMD_Model'





##### To iterate through each of the optimization runs (files), and extract the results:

active_params_symbols = [#r'$f_{\sigma_{i,\rm high}}$',
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
                         #r'$d(\alpha_P)/d(b_p-r_p)$',
                         r'$\alpha_{R1}$',
                         r'$\alpha_{R2}$',
                         r'$\sigma_e$',
                         #r'$\sigma_i$',
                         #r'$\sigma_{i,\rm res}$',
                         r'$\sigma_R$',
                         r'$\sigma_N$'
                         ] # this list of parameter symbols must match the order of parameters in 'active_params_names'!

##### To read a file of recomputed distances and save it as a table format file for training an emulator:

def load_split_stars_recomputed_distances_file(file_name):
    sample_names = ['all', 'bluer', 'redder']

    active_params_evals = []
    d_used_keys_evals = {key: [] for key in sample_names}
    d_used_vals_evals = {key: [] for key in sample_names}
    d_used_vals_w_evals = {key: [] for key in sample_names}
    total_dist_w_evals = []

    with open(file_name, 'r') as file:
        for line in file:
            if line[0:19] == '# Active parameters':
                active_params_names = line[23:-3].split('", "')
            elif line[0:13] == 'Active_params':
                active_params = [float(x) for x in line[16:-2].split(', ')]
                active_params_evals.append(active_params)
            elif line[0:12] == 'Total_dist_w':
                total_dist_w = float(line[15:-2])
                total_dist_w_evals.append(total_dist_w)

            for key in sample_names:
                n = len(key)
                if line[0:n+2] == '[%s]' % key:

                    if line[n+3:n+3+12] == 'd_used_keys:':
                        d_used_keys = line[n+3+15:-3].split('", "')
                        d_used_keys_evals[key].append(d_used_keys)

                    elif line[n+3:n+3+12] == 'd_used_vals:':
                        d_used_vals_str, d_used_vals_tot_str = line[n+3+14:-2].split('][')
                        d_used_vals = [float(x) for x in d_used_vals_str.split(', ')]
                        d_used_vals_evals[key].append(tuple(d_used_vals))

                    elif line[n+3:n+3+13] == 'd_used_vals_w':
                        d_used_vals_w_str, d_used_vals_tot_w_str = line[n+3+16:-2].split('][')
                        d_used_vals_w = [float(x) for x in d_used_vals_w_str.split(', ')]
                        d_used_vals_tot_w = float(d_used_vals_tot_w_str)
                        d_used_vals_w_evals[key].append(tuple(d_used_vals_w))

    active_params_evals = np.array(active_params_evals)
    total_dist_w_evals = np.array(total_dist_w_evals)

    for sample in sample_names:
        d_used_keys_evals[sample] = np.array(d_used_keys_evals[sample])
        d_used_vals_evals[sample] = np.array(d_used_vals_evals[sample], dtype=[(d_key, 'f8') for d_key in d_used_keys_evals[sample][0]])
        d_used_vals_w_evals[sample] = np.array(d_used_vals_w_evals[sample], dtype=[(d_key, 'f8') for d_key in d_used_keys_evals[sample][0]])

    # To compute the sums of weighted distances per iteration, for each sample:
    dtot_samples_evals = {}
    dtot_w_samples_evals = {}
    for sample in sample_names:
        dtot_samples_evals[sample] = np.array([sum(x) for x in d_used_vals_evals[sample]])
        dtot_w_samples_evals[sample] = np.array([sum(x) for x in d_used_vals_w_evals[sample]])
    dtot_w_evals = sum(dtot_w_samples_evals[sample] for sample in sample_names)

    for i in range(len(dtot_w_evals)):
        a, b = dtot_w_evals[i], total_dist_w_evals[i]
        #if np.abs(a - b) > 1e-4:
            #print('{:<5}: {:<8}, {:<8}'.format(i, np.round(a,4), np.round(b,4)))

    return active_params_names, active_params_evals, d_used_vals_w_evals, dtot_w_samples_evals, dtot_w_evals





N_best_save, keep_every = 100000, 10
active_params_names, active_params_evals, d_used_vals_w_evals, dtot_w_samples_evals, dtot_w_evals = load_split_stars_recomputed_distances_file(loadfiles_directory + 'Clustered_P_R_recompute_optim_best%s_every%s_targs86760.txt' % (N_best_save, keep_every))

##### To save the best parameter values and the recomputed distances for training a GP emulator:
'''
active_params_distances_table = np.concatenate((active_params_evals, np.array([dtot_w_samples_evals['all'], dtot_w_samples_evals['bluer'], dtot_w_samples_evals['redder'], dtot_w_evals]).transpose()), axis=1)
table_header = ' '.join(active_params_names) + ' dtot_w_all dtot_w_bluer dtot_w_redder dist_tot_weighted'
fields_formats = ['%1.6f']*len(active_params_names) + ['%1.6f']*4
#####active_params_distances_table = np.concatenate((active_params_evals, np.array([dtot_w_samples_evals['redder'], dtot_w_evals]).transpose()), axis=1)
#####table_header = ' '.join(active_params_names) + ' dtot_w_redder dist_tot_weighted'
#####fields_formats = ['%1.6f']*len(active_params_names) + ['%1.6f']*2
np.savetxt(loadfiles_directory + 'Active_params_recomputed_distances_table_best%s_every%s.txt' % (N_best_save, keep_every), active_params_distances_table, fmt=fields_formats, header=table_header, comments='')
'''





##### To plot corner plots of the same points, with the new distance terms as a colorscale:
#'''
#plot_cornerpy_wrapper(active_params_symbols, active_params_evals, title_kwargs={'fontsize':20}, save_name=savefigures_directory + model_name + '_best%s_every%s_corner.pdf' % (N_best_save, keep_every), save_fig=savefigures)

plot_function_heatmap_averaged_grid_given_irregular_points_corner(active_params_symbols, active_params_evals, dtot_w_evals, flabel=r'$\mathcal{D}_W$', show_points=False, save_name=savefigures_directory + model_name + '_best%s_every%s_corner_dtot_w.pdf' % (N_best_save, keep_every), save_fig=savefigures)

dist_terms = ['radii_partitioning_KS', 'radii_monotonicity_KS', 'gap_complexity_KS']
#dist_terms = ['radii_partitioning_AD', 'radii_monotonicity_AD', 'gap_complexity_AD']
for (i,key) in enumerate(dist_terms):
    plot_function_heatmap_averaged_grid_given_irregular_points_corner(active_params_symbols, active_params_evals, d_used_vals_w_evals['all'][key], flabel=key, show_points=False, save_name=savefigures_directory + model_name + '_best%s_every%s_corner_%s.pdf' % (N_best_save, keep_every, key), save_fig=savefigures)
plt.show()
#'''
