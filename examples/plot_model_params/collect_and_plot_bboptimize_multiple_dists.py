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

savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Distances_compare/bboptimize/'
save_name = 'Models_Compare_bboptimize'

sample_names = ['all', 'bluer', 'redder']
sample_colors = ['k', 'b', 'r']
samples = len(sample_names)





##### To analyze multiple bboptimize runs:

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/'
run_numbers1 = np.loadtxt(loadfiles_directory1 + 'run_numbers.txt', dtype='i4')

loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/'
run_numbers2 = np.loadtxt(loadfiles_directory2 + 'run_numbers.txt', dtype='i4')

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


#'''
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

if savefigures:
    plt.savefig(savefigures_directory + save_name + '_' + '_'.join(model_savenames) + '_dists_w_best%s_every%s.pdf' % (N_best, keep_every))
else:
    plt.show()
#'''





#'''
model_N_best, model_keep_every = [100000, 100000], [10, 10]
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

if savefigures:
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
