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





savefigures = False

savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/Distances/'
save_name = 'Dists'





#####

def load_split_stars_model_evaluations_weighted(file_name, dtot_max_keep=np.inf, max_keep=np.inf):
    sample_names = ['all', 'bluer', 'redder']

    Nmult_max = 8
    Nmult_evals = {key: [] for key in sample_names}
    d_used_keys_evals = {key: [] for key in sample_names}
    d_used_vals_w_evals = {key: [] for key in sample_names}
    d_used_vals_tot_w_evals = []
    
    with open(file_name, 'r') as file:
        for line in file:
            for key in sample_names:
                n = len(key)
                if line[0:n+2] == '[%s]' % key:
                    if line[n+3:n+3+6] == 'Counts':
                        Nmult_str, counts_str = line[n+3+9:-2].split('][')
                        Nmult = tuple([int(x) for x in Nmult_str.split(', ')])
                        Nmult_evals[key].append(Nmult)
                    
                    if line[n+3:n+3+12] == 'd_used_keys:':
                        d_used_keys = line[n+3+15:-3].split('", "')
                        d_used_keys_evals[key].append(d_used_keys)
                    
                    if line[n+3:n+3+12] == 'd_used_vals:':
                        d_used_vals_str, d_used_vals_tot_str = line[n+3+14:-2].split('][')
                        d_used_vals = tuple([float(x) for x in d_used_vals_str.split(', ')])
                        #d_used_vals_evals[key].append(d_used_vals)

                    elif line[n+3:n+3+13] == 'd_used_vals_w':
                        d_used_vals_w_str, d_used_vals_tot_w_str = line[n+3+16:-2].split('][')
                        d_used_vals_w = tuple([float(x) for x in d_used_vals_w_str.split(', ')])
                        d_used_vals_tot_w = float(d_used_vals_tot_w_str)
                        d_used_vals_w_evals[key].append(d_used_vals_w)

    d_used_vals_tot_w_evals = [np.sum(d_used_vals_w_evals['all'][i]) + np.sum(d_used_vals_w_evals['bluer'][i]) + np.sum(d_used_vals_w_evals['redder'][i]) for i in range(len(d_used_vals_w_evals['all']))]

    # Keep only the evals that pass a distance threshold:
    Nmult_keep = {key: [] for key in sample_names}
    d_used_keys_keep = {key: [] for key in sample_names}
    d_used_vals_w_keep = {key: [] for key in sample_names}
    d_used_vals_tot_w_keep = []
    for i,dtot_w in enumerate(d_used_vals_tot_w_evals):
        if (dtot_w <= dtot_max_keep) and (len(d_used_vals_tot_w_keep) < max_keep):
            for key in sample_names:
                Nmult_keep[key].append(Nmult_evals[key][i])
                d_used_keys_keep[key].append(d_used_keys_evals[key][i])
                d_used_vals_w_keep[key].append(d_used_vals_w_evals[key][i])
            d_used_vals_tot_w_keep.append(dtot_w)
    d_used_vals_tot_w_keep = np.array(d_used_vals_tot_w_keep)
    print('Evals passing threshold: ', len(d_used_vals_tot_w_keep))

    for key in sample_names:
        Nmult_evals[key] = np.array(Nmult_evals[key], dtype=[(str(n), 'i8') for n in range(1,Nmult_max+1)])
        d_used_keys_evals[key] = np.array(d_used_keys_evals[key])
        d_used_vals_w_evals[key] = np.array(d_used_vals_w_evals[key], dtype=[(dist_key, 'f8') for dist_key in d_used_keys_evals[key][0]])

        Nmult_keep[key] = np.array(Nmult_keep[key], dtype=[(str(n), 'i8') for n in range(1,Nmult_max+1)])
        d_used_keys_keep[key] = np.array(d_used_keys_keep[key])
        d_used_vals_w_keep[key] = np.array(d_used_vals_w_keep[key], dtype=[(dist_key, 'f8') for dist_key in d_used_keys_keep[key][0]])

    return Nmult_keep, d_used_keys_keep, d_used_vals_w_keep





Nmult_keep_1_KS, d_used_keys_keep_1_KS, d_used_vals_w_keep_1_KS = load_split_stars_model_evaluations_weighted('/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/Clustered_P_R_pass_GP_meanf100.0_thres65.0_pass1000_targs86760.txt', dtot_max_keep=65., max_keep=1000)



#Nmult_keep_1_KS, d_used_keys_keep_1_KS, d_used_vals_w_keep_1_KS = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_KS/GP_best_models/Clustered_P_R_pass_GP_meanf75.0_thres45.0_pass1000_targs86760.txt', dtot_max_keep=45., max_keep=1000)

#Nmult_keep_1_AD, d_used_keys_keep_1_AD, d_used_vals_w_keep_1_AD = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_AD/durations_AD/GP_best_models/Clustered_P_R_pass_GP_meanf150.0_thres80.0_pass1000_targs86760.txt', dtot_max_keep=80.)

#Nmult_keep_2_KS, d_used_keys_keep_2_KS, d_used_vals_w_keep_2_KS = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/Clustered_P_R_pass_GP_meanf75.0_thres47.0_pass1000_targs88912.txt', dtot_max_keep=47.)

#Nmult_keep_2_AD, d_used_keys_keep_2_AD, d_used_vals_w_keep_2_AD = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_AD/durations_AD/GP_best_models/Clustered_P_R_pass_GP_meanf150.0_thres90.0_pass1000_targs88912.txt', dtot_max_keep=90.)



#Nmult_keep_1_KS, d_used_keys_keep_1_KS, d_used_vals_w_keep_1_KS = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp/Params12_KS/durations_KS/GP_best_models/Clustered_P_R_pass_GP_meanf75.0_thres50.0_pass1000_targs88912.txt', dtot_max_keep=50.)

#Nmult_keep_1_AD, d_used_keys_keep_1_AD, d_used_vals_w_keep_1_AD = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp/Params12_AD/durations_AD/GP_best_models/Clustered_P_R_pass_GP_meanf150.0_thres100.0_pass1000_targs88912.txt', dtot_max_keep=100.)

#Nmult_keep_2_KS, d_used_keys_keep_2_KS, d_used_vals_w_keep_2_KS = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/Clustered_P_R_pass_GP_meanf75.0_thres47.0_pass1000_targs88912.txt', dtot_max_keep=47.)

#Nmult_keep_2_AD, d_used_keys_keep_2_AD, d_used_vals_w_keep_2_AD = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_AD/durations_AD/GP_best_models/Clustered_P_R_pass_GP_meanf150.0_thres90.0_pass1000_targs88912.txt', dtot_max_keep=90.)

#Nmult_keep_3_KS, d_used_keys_keep_3_KS, d_used_vals_w_keep_3_KS = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/Split_stars/Clustered_P_R_alphaP_bprp/Params13_KS/durations_KS/GP_best_models/Clustered_P_R_pass_GP_meanf75.0_thres47.0_pass1000_targs88912.txt', dtot_max_keep=47.)

#Nmult_keep_3_AD, d_used_keys_keep_3_AD, d_used_vals_w_keep_3_AD = load_split_stars_model_evaluations_weighted('../../ACI/Simulated_Data/Split_stars/Clustered_P_R_alphaP_bprp/Params13_AD/durations_AD/GP_best_models/Clustered_P_R_pass_GP_meanf150.0_thres90.0_pass1000_targs88912.txt', dtot_max_keep=90.)



model_dw_KS = [d_used_vals_w_keep_1_KS] #[d_used_vals_w_keep_1_KS, d_used_vals_w_keep_2_KS]
#model_dw_AD = [d_used_vals_w_keep_1_AD, d_used_vals_w_keep_2_AD]
model_names = ['Maximum AMD model'] #[r'Constant $f_{\rm swpa}+\alpha_P$', r'Linear $f_{\rm swpa}(b_p - r_p - E^*)$', r'Linear $\alpha_P(b_p - r_p - E^*)$'] #['Maximum AMD model', 'Two-Rayleigh model (Paper II)'] # Make sure this matches the models loaded!
model_linestyles = ['-'] #['-', '--']
model_alphas = [] #[0.3, 0.2, 0.1]
n_models = len(model_names)

sample_names = ['all', 'bluer', 'redder']
sample_labels = ['All', 'Bluer', 'Redder']
sample_colors = ['k','b','r']





##### To plot histograms of the individual distances:

fig_size = (8,4) #size of each panel (figure)
fig_lbrt = [0.125, 0.2, 0.975, 0.95]

n_bins = 50
lw = 2 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 20 #legend labels font size

dist_symbols_KS = {
    "delta_f": r'$w D_f$',
    "mult_CRPD_r": r'$w \rho_{\rm CRPD}$',
    "periods_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{P\}$',
    "period_ratios_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\mathcal{P}\}$',
    "depths_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\delta\}$',
    "radius_ratios_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\delta_{i+1}/\delta_i\}$',
    #"durations_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{t_{\rm dur}\}$',
    #"durations_norm_circ_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{t_{\rm dur}/t_{\rm circ}\}$',
    "durations_norm_circ_singles_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{t_{\rm dur}/t_{\rm circ}\}_{1}$',
    "durations_norm_circ_multis_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{t_{\rm dur}/t_{\rm circ}\}_{2+}$',
    "duration_ratios_mmr_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\xi_{\rm res}\}$',
    "duration_ratios_nonmmr_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\xi_{\rm non-res}\}$',
    "radii_partitioning_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\mathcal{Q}_R\}$',
    "radii_monotonicity_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\mathcal{M}_R\}$',
    "gap_complexity_KS": r'$w \mathcal{D}_{\rm KS}$ for $\{\mathcal{C}\}$',
}
dist_symbols_AD = {
    "delta_f": r'$w D_f$',
    "mult_CRPD_r": r'$w \rho_{\rm CRPD}$',
    "periods_AD": r'$w \mathcal{D}_{\rm AD^\prime}$ for $\{P\}$',
    "period_ratios_AD": r'$w \mathcal{D}_{\rm AD^\prime}$ for $\{\mathcal{P}\}$',
    "depths_AD": r'$w \mathcal{D}_{\rm AD^\prime}$ for $\{\delta\}$',
    "radius_ratios_AD": r'$w \mathcal{D}_{\rm AD^\prime}$ for $\{\delta_{i+1}/\delta_i\}$',
    "durations_AD": r'$w \mathcal{D}_{\rm AD^\prime}$ for $\{t_{\rm dur}\}$',
    "duration_ratios_mmr_AD": r'$w \mathcal{D}_{\rm AD^\prime}$ for $\{\xi_{\rm res}\}$',
    "duration_ratios_nonmmr_AD": r'$w \mathcal{D}_{\rm AD^\prime}$ for $\{\xi_{\rm non-res}\}$',
}



# KS analysis:
#'''
# Total weighted distances:
dtot_w = [np.array([sum(dw_KS['all'][i]) + sum(dw_KS['bluer'][i]) + sum(dw_KS['redder'][i]) for i in range(len(dw_KS['all']))]) for dw_KS in model_dw_KS]
plot_fig_pdf_simple((12,4), dtot_w, [], n_bins=n_bins, normalize=False, c_sim=['k']*n_models, ls_sim=model_linestyles, lw=3, labels_sim=model_names, xlabel_text=r'$\mathcal{D}_{W,3} (\rm KS)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=20)
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_sum_dtot_w.pdf')
    plt.close()

# Total weighted distances (split):
fig = plt.figure(figsize=(12,4))
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)
ax = plt.subplot(plot[0,0])
plot_panel_pdf_simple(ax, [[sum(dw) for dw in dw_KS['all']] for dw_KS in model_dw_KS], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[0]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text='', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s='All', ha='right', c='k', fontsize=20, transform=ax.transAxes)
ax = plt.subplot(plot[0,1])
plot_panel_pdf_simple(ax, [[sum(dw) for dw in dw_KS['bluer']] for dw_KS in model_dw_KS], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[1]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text=r'$\mathcal{D}_{W,3} (\rm KS)$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s='Bluer', ha='right', c='b', fontsize=20, transform=ax.transAxes)
plt.yticks([])
ax = plt.subplot(plot[0,2])
plot_panel_pdf_simple(ax, [[sum(dw) for dw in dw_KS['redder']] for dw_KS in model_dw_KS], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[2]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text='', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s='Redder', ha='right', c='r', fontsize=20, transform=ax.transAxes)
plt.yticks([])
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_dtot_w.pdf')
    plt.close()

# Individual distance terms:
for key in dist_symbols_KS:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)
    plot_panel_pdf_simple(plt.subplot(plot[0,0]), [dw_KS['all'][key] for dw_KS in model_dw_KS], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[0]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text='', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)
    plot_panel_pdf_simple(plt.subplot(plot[0,1]), [dw_KS['bluer'][key] for dw_KS in model_dw_KS], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[1]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text=dist_symbols_KS[key], ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    plt.yticks([])
    plot_panel_pdf_simple(plt.subplot(plot[0,2]), [dw_KS['redder'][key] for dw_KS in model_dw_KS], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[2]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text='', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    plt.yticks([])
    if savefigures:
        plt.savefig(savefigures_directory + save_name + '_%s.pdf' % key)
        plt.close()

plt.show()
#'''



# AD analysis:
'''
# Total weighted distances:
dtot_w = [np.array([sum(dw_AD['all'][i]) + sum(dw_AD['bluer'][i]) + sum(dw_AD['redder'][i]) for i in range(len(dw_AD['all']))]) for dw_AD in model_dw_AD]
plot_fig_pdf_simple((12,4), dtot_w, [], n_bins=n_bins, normalize=False, c_sim=['k']*n_models, ls_sim=model_linestyles, lw=3, labels_sim=model_names, xlabel_text=r'$\mathcal{D}_{W,1} (\rm AD^\prime)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_sum_dtot_w.pdf')
    plt.close()

# Total weighted distances (split):
fig = plt.figure(figsize=(12,4))
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)
ax = plt.subplot(plot[0,0])
plot_panel_pdf_simple(ax, [[sum(dw) for dw in dw_AD['all']] for dw_AD in model_dw_AD], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[0]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text='', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True)
plt.text(x=0.98, y=0.9, s='All', ha='right', c='k', fontsize=20, transform=ax.transAxes)
ax = plt.subplot(plot[0,1])
plot_panel_pdf_simple(ax, [[sum(dw) for dw in dw_AD['bluer']] for dw_AD in model_dw_AD], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[1]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text=r'$\mathcal{D}_{W,1} (\rm AD^\prime)$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=True)
plt.text(x=0.98, y=0.9, s='Bluer', ha='right', c='b', fontsize=20, transform=ax.transAxes)
plt.yticks([])
ax = plt.subplot(plot[0,2])
plot_panel_pdf_simple(ax, [[sum(dw) for dw in dw_AD['redder']] for dw_AD in model_dw_AD], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[2]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text='', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=True)
plt.text(x=0.98, y=0.9, s='Redder', ha='right', c='r', fontsize=20, transform=ax.transAxes)
plt.yticks([])
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_dtot_w.pdf')
    plt.close()

# Individual distance terms:
for key in dist_symbols_AD:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)
    plot_panel_pdf_simple(plt.subplot(plot[0,0]), [dw_AD['all'][key] for dw_AD in model_dw_AD], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[0]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text='', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs)
    plot_panel_pdf_simple(plt.subplot(plot[0,1]), [dw_AD['bluer'][key] for dw_AD in model_dw_AD], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[1]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text=dist_symbols_AD[key], ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    plt.yticks([])
    plot_panel_pdf_simple(plt.subplot(plot[0,2]), [dw_AD['redder'][key] for dw_AD in model_dw_AD], [], n_bins=n_bins, normalize=False, c_sim=[sample_colors[2]]*n_models, ls_sim=model_linestyles, lw=lw, labels_sim=[None]*n_models, xlabel_text='', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    plt.yticks([])
    if savefigures:
        plt.savefig(savefigures_directory + save_name + '_%s.pdf' % key)
        plt.close()

plt.show()
'''





##### To compute the quantiles for the observed multiplicities:

def compute_and_print_quantiles_Nmult_split_stars(Nmult_evals):
    sample_names = ['all', 'bluer', 'redder']
    sample_stars = {'all': N_Kep, 'bluer': 43380, 'redder': 43380}
    
    Nmult_quantiles = {key: {'q16': [], 'qmed': [], 'q84': []} for key in sample_names}
    total_planets = {key: np.zeros(len(Nmult_evals[key])) for key in sample_names}
    for sample in sample_names:
        print(sample)
        for key in Nmult_evals[sample].dtype.names:
            Nmults_i = Nmult_evals[sample][key]
            q16, qmed, q84 = np.quantile(Nmults_i, [0.16, 0.5, 0.84])
            Nmult_quantiles[sample]['q16'].append(q16)
            Nmult_quantiles[sample]['qmed'].append(qmed)
            Nmult_quantiles[sample]['q84'].append(q84)
            #print('%s : %s, %s, %s' % (key, int(np.round(q16)), int(np.round(qmed)), int(np.round(q84))))
            print('%s : %s_{%s}^{+%s}' % (key, int(np.round(qmed)), int(np.round(q16-qmed)), int(np.round(q84-qmed))))
            total_planets[sample] += Nmults_i*int(key)
        
        # Compute the total numbers of planets:
        q16, qmed, q84 = np.quantile(total_planets[sample], [0.16, 0.5, 0.84])
        print('Total planets: %s_{%s}^{+%s}' % (int(np.round(qmed)), int(np.round(q16-qmed)), int(np.round(q84-qmed))))

        # Compute the total numbers of m=0 systems (stars with no detected planets):
        num_0pl = np.zeros(len(Nmult_evals[sample]))
        for (i,x) in enumerate(Nmult_evals[sample]):
            num_0pl[i] = sample_stars[sample] - sum(x)
        q16, qmed, q84 = np.quantile(num_0pl, [0.16, 0.5, 0.84])
        print('Total 0-planet systems: %s_{%s}^{+%s}' % (int(np.round(qmed)), int(np.round(q16-qmed)), int(np.round(q84-qmed))))
        
    return Nmult_quantiles

Nmult_quantiles_1_KS = compute_and_print_quantiles_Nmult_split_stars(Nmult_keep_1_KS)
#Nmult_quantiles_2_KS = compute_and_print_quantiles_Nmult_split_stars(Nmult_keep_2_KS)
#Nmult_quantiles_3_KS = compute_and_print_quantiles_Nmult_split_stars(Nmult_keep_3_KS)

#Nmult_quantiles_1_KS = compute_and_print_quantiles_Nmult_split_stars(Nmult_keep_1_KS)
#Nmult_quantiles_1_AD = compute_and_print_quantiles_Nmult_split_stars(Nmult_keep_1_AD)
#Nmult_quantiles_2_KS = compute_and_print_quantiles_Nmult_split_stars(Nmult_keep_2_KS)
#Nmult_quantiles_2_AD = compute_and_print_quantiles_Nmult_split_stars(Nmult_keep_2_AD)



# To load the Kepler multiplicity distributions:
P_min, P_max, radii_min, radii_max = 3., 300., 0.5, 10.
stars_cleaned = load_Kepler_stars_cleaned()
#bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
bp_rp_corr_med = np.nanmedian(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'])

ssk_per_sys0, ssk0 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
ssk_per_sys1, ssk1 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_max=bp_rp_corr_med) #_max=_med
ssk_per_sys2, ssk2 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=bp_rp_corr_med) #_min=_med

Nmult_Kep = {'all': ssk0['Nmult_obs'], 'bluer': ssk1['Nmult_obs'], 'redder': ssk2['Nmult_obs']}



# To plot the quantiles for the multiplicity distributions normalized by the Kepler multiplicity distributions:
'''
fig = plt.figure(figsize=(8,8))
plot = GridSpec(3,1,left=0.15,bottom=0.1,right=0.95,top=0.98,wspace=0,hspace=0)
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,0])
    plt.axhline(y=1., ls=':', c=sample_colors[i], label='Exact match')
    #plt.plot(range(1,7), np.ones(6), '-x', color=sample_colors[i], label='Exact match')

    for j,Nmult_q in enumerate([Nmult_quantiles_1_KS, Nmult_quantiles_2_KS, Nmult_quantiles_3_KS]):
        for n in range(6):
            ratio_med = Nmult_q[sample]['qmed'][n]/Nmult_Kep[sample][n]
            ratio_q16 = Nmult_q[sample]['q16'][n]/Nmult_Kep[sample][n]
            ratio_q84 = Nmult_q[sample]['q84'][n]/Nmult_Kep[sample][n]
            if n==0:
                plt.plot((n+0.5, n+1.5), (ratio_med, ratio_med), color=sample_colors[i], ls=model_linestyles[j], lw=2, label=model_names[j])
            else:
                plt.plot((n+0.5, n+1.5), (ratio_med, ratio_med), color=sample_colors[i], ls=model_linestyles[j], lw=2)

            # For the credible regions:
            #plt.plot((n+0.5, n+1.5), (ratio_q16, ratio_q16), color=sample_colors[i], ls=model_linestyles[j])
            #plt.plot((n+0.5, n+1.5), (ratio_q84, ratio_q84), color=sample_colors[i], ls=model_linestyles[j])
            #ax.add_patch(matplotlib.patches.Rectangle((n+0.5, ratio_q16), width=1, height=ratio_q84-ratio_q16, alpha=model_alphas[j], color=sample_colors[i], ls=model_linestyles[j]))
            plt.plot((n+1+(j-0.5)/10, n+1+(j-0.5)/10), (ratio_q16, ratio_q84), color=sample_colors[i], ls=model_linestyles[j], lw=1)
            ###plt.errorbar(x=[n+1], y=[ratio_med], yerr=[[ratio_med-ratio_q16], [ratio_q84-ratio_med]], color=sample_colors[i], ls=model_linestyles[j], lw=1)

    plt.text(0.98, 0.95, sample_labels[i], ha='right', va='top', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    ax.tick_params(axis='both', labelsize=afs)
    plt.yticks([0.5,1,1.5])
    plt.xlim([0.5,6.5])
    plt.ylim([0.4,1.6])
    if i==0:
        plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=2, frameon=False, fontsize=14)
    if i==1:
        plt.ylabel(r'$N_{\rm sim}(m)/N_{\rm Kep}(m)$', fontsize=tfs)
    if i==2:
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_normed_bars.pdf')
    plt.close()



fig = plt.figure(figsize=(8,8))
plot = GridSpec(3,1,left=0.15,bottom=0.1,right=0.95,top=0.98,wspace=0,hspace=0)
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,0])
    plt.axhline(y=1., ls=':', c=sample_colors[i], label='Exact match')
    #plt.plot(range(1,7), np.ones(6), '-x', color=sample_colors[i], label='Exact match')
    
    Nmults = [Nmult_keep_1_KS, Nmult_keep_2_KS, Nmult_keep_3_KS]
    for j,Nmult_q in enumerate([Nmult_quantiles_1_KS, Nmult_quantiles_2_KS, Nmult_quantiles_3_KS]):
        for n in range(4): # for bins 1,2,3,4
            ratio_med = Nmult_q[sample]['qmed'][n]/Nmult_Kep[sample][n]
            ratio_q16 = Nmult_q[sample]['q16'][n]/Nmult_Kep[sample][n]
            ratio_q84 = Nmult_q[sample]['q84'][n]/Nmult_Kep[sample][n]
            if n==0:
                plt.plot((n+0.5, n+1.5), (ratio_med, ratio_med), color=sample_colors[i], ls=model_linestyles[j], lw=2, label=model_names[j])
            else:
                plt.plot((n+0.5, n+1.5), (ratio_med, ratio_med), color=sample_colors[i], ls=model_linestyles[j], lw=2)
            
            # For the credible regions:
            #plt.plot((n+0.5, n+1.5), (ratio_q16, ratio_q16), color=sample_colors[i], ls=model_linestyles[j])
            #plt.plot((n+0.5, n+1.5), (ratio_q84, ratio_q84), color=sample_colors[i], ls=model_linestyles[j])
            #ax.add_patch(matplotlib.patches.Rectangle((n+0.5, ratio_q16), width=1, height=ratio_q84-ratio_q16, alpha=model_alphas[j], color=sample_colors[i], ls=model_linestyles[j]))
            plt.plot((n+1+(j-0.5)/10, n+1+(j-0.5)/10), (ratio_q16, ratio_q84), color=sample_colors[i], ls=model_linestyles[j], lw=1)

        # For the binned 5+ bin:
        n = 4
        counts5plus_all = Nmults[j][sample]['5'] + Nmults[j][sample]['6'] + Nmults[j][sample]['7'] + Nmults[j][sample]['8']
        #q16, qmed, q84 = np.quantile(counts5plus_all, [0.16, 0.5, 0.84])
        #print('5+ : %s_{%s}^{+%s}' % (int(np.round(qmed)), int(np.round(q16-qmed)), int(np.round(q84-qmed))))
        ratio_all = counts5plus_all/(Nmult_Kep[sample][4] + Nmult_Kep[sample][5])
        ratio_q16, ratio_med, ratio_q84 = np.quantile(ratio_all, [0.16, 0.5, 0.84])
        plt.plot((n+0.5, n+1.5), (ratio_med, ratio_med), color=sample_colors[i], ls=model_linestyles[j], lw=2)
        #ax.add_patch(matplotlib.patches.Rectangle((n+0.5, ratio_q16), width=1, height=ratio_q84-ratio_q16, alpha=model_alphas[j], color=sample_colors[i], ls=model_linestyles[j]))
        plt.plot((n+1+(j-0.5)/10, n+1+(j-0.5)/10), (ratio_q16, ratio_q84), color=sample_colors[i], ls=model_linestyles[j], lw=1)

    plt.text(0.98, 0.95, sample_labels[i], ha='right', va='top', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    ax.tick_params(axis='both', labelsize=afs)
    plt.yticks([0.5,1,1.5])
    plt.xlim([0.5,5.5])
    plt.ylim([0.4,1.6])
    a = np.array(ax.get_xticks(), dtype='int').tolist()
    a[5] = '5+'
    ax.set_xticklabels(a)
    if i==0:
        plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=2, frameon=False, fontsize=14)
    if i==1:
        plt.ylabel(r'$N_{\rm sim}(m)/N_{\rm Kep}(m)$', fontsize=tfs)
    if i==2:
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_normed_bars_binned5plus.pdf')
    plt.close()
'''





# To plot the quantiles for the multiplicity distributions (log and unlogged y-axes):
'''
fig = plt.figure(figsize=(12,8))
plot = GridSpec(3,2,left=0.1,bottom=0.1,right=0.95,top=0.98,wspace=0.2,hspace=0)

# Linear y-axes:
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,0])
    plt.plot(range(1,7), Nmult_Kep[sample], 'x', color=sample_colors[i], label='Kepler')

    for j,Nmult_q in enumerate([Nmult_quantiles_1_KS, Nmult_quantiles_2_KS]):
        for n in range(6):
            counts_med = Nmult_q[sample]['qmed'][n]
            counts_q16 = Nmult_q[sample]['q16'][n]
            counts_q84 = Nmult_q[sample]['q84'][n]
            if n==0:
                plt.plot((n+0.5, n+1.5), (counts_med, counts_med), color=sample_colors[i], ls=model_linestyles[j], lw=2, label=model_names[j])
            else:
                plt.plot((n+0.5, n+1.5), (counts_med, counts_med), color=sample_colors[i], ls=model_linestyles[j], lw=2)
            ax.add_patch(matplotlib.patches.Rectangle((n+0.5, counts_q16), width=1, height=counts_q84-counts_q16, alpha=model_alphas[j], color=sample_colors[i], ls=model_linestyles[j]))

    ax.tick_params(axis='both', labelsize=afs)
    plt.xticks(range(1,7))
    plt.xlim([0.5,6.5])
    if i==0:
        plt.ylim([0,1400])
        plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)
    if i==1:
        plt.ylim([0,700])
        plt.ylabel(r'$N_{\rm sim}(m)$', fontsize=tfs)
    if i==2:
        plt.ylim([0,700])
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

# Log y-axes:
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,1])
    plt.plot(range(1,7), Nmult_Kep[sample], 'x', color=sample_colors[i], label='Kepler')

    for j,Nmult_q in enumerate([Nmult_quantiles_1_KS, Nmult_quantiles_2_KS]):
        for n in range(6):
            counts_med = Nmult_q[sample]['qmed'][n]
            counts_q16 = Nmult_q[sample]['q16'][n]
            counts_q84 = Nmult_q[sample]['q84'][n]
            if n==0:
                plt.plot((n+0.5, n+1.5), (counts_med, counts_med), color=sample_colors[i], ls=model_linestyles[j], lw=2, label=model_names[j])
            else:
                plt.plot((n+0.5, n+1.5), (counts_med, counts_med), color=sample_colors[i], ls=model_linestyles[j], lw=2)
            ax.add_patch(matplotlib.patches.Rectangle((n+0.5, counts_q16), width=1, height=counts_q84-counts_q16, alpha=model_alphas[j], color=sample_colors[i], ls=model_linestyles[j]))

    plt.text(0.98, 0.95, sample_labels[i], ha='right', va='top', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    plt.xticks(range(1,7))
    plt.xlim([0.5,6.5])
    if i==2:
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_compare.pdf')
    plt.close()

plt.show()
'''
