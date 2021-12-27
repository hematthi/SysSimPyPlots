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

savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/Split_stars/Intrinsic_compare/KS/'
save_name = 'Models_Compare'





##### To load the underlying populations:

# Model 1:
loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Julia_v0.7/Kepler_catalog_optimization/q1q17_dr25_gaia_fgk_stars79935/Clustered_P_R/f_high_incl_low_incl_mmr/Fit_rate_mult_P_Pratios_D_Dratios_dur_durratios_mmr/Some11_params_CRPDr_KS/Fixed_Rbreak3_Ncrit8/lc_0p2_5_lp_0p5_10_alphaP_-2_2_alphaR1_-4_2_alphaR2_-6_0_ecc_0_0p1_incl_inclmmr_0_90_sigmaR_0_0p5_sigmaP_0_0p3/targs79935_maxincl0_maxiters5000/sigma_i_greater_sigma_i_mmr/GP_best_models/'
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory1 + 'clusterids_all1.out')

# Model 2:
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp/Params12_KS/durations_KS/GP_best_models/'

# Model 3:
loadfiles_directory3 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'

# Model 4:
loadfiles_directory4 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_alphaP_bprp/Params13_KS/durations_KS/GP_best_models/'





model_loadfiles_dirs = [loadfiles_directory1, loadfiles_directory2, loadfiles_directory3, loadfiles_directory4]
model_names = ['Clustered P+R (Paper I)', r'Constant $f_{\rm swpa}+\alpha_P$', r'Linear $f_{\rm swpa}(b_p - r_p - E^*)$', r'Linear $\alpha_P(b_p - r_p - E^*)$'] #['Maximum AMD model', r'Linear $f_{\rm swpa}(b_p - r_p)$ model (Paper II)'] # Make sure this matches the models loaded!
model_linestyles = [':', '--', '-', '-.']
model_colors = ['r', 'g', 'b', 'c']
model_stagger_errorbars = [-0.15, -0.05, 0.05, 0.15] # offsets for plotting multiplicity counts in order to stagger errorbars
models = len(model_loadfiles_dirs)

KS_or_AD = 'KS'





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

runs = 100

Mtot_bins = np.arange(23)-1.5 #include a -1 bin with all zeros for plotting purposes
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = []

clustertot_bins = np.arange(12)-0.5 #includes 0 bin but will not count them
clustertot_bins_mid = (clustertot_bins[:-1] + clustertot_bins[1:])/2.
clustertot_counts_all = []

pl_per_cluster_bins = np.arange(12)-0.5 #includes 0 bin but will not count them
pl_per_cluster_bins_mid = (pl_per_cluster_bins[:-1] + pl_per_cluster_bins[1:])/2.
pl_per_cluster_counts_all = []

for loadfiles_dir in model_loadfiles_dirs:
    Mtot_counts = []
    clustertot_counts = []
    pl_per_cluster_counts = []
    
    for i in range(1,runs+1): #range(1,runs+1)
        run_number = i
        print(i)
        N_sim_i = read_targets_period_radius_bounds(loadfiles_dir + 'clusterids_all%s.out' % run_number)[0]
        param_vals_i = read_sim_params(loadfiles_dir + 'clusterids_all%s.out' % run_number)
        sssp_per_sys_basic_i = load_cat_phys_separate_and_compute_basic_summary_stats_per_sys(loadfiles_dir, run_number)
        
        # Multiplicities:
        counts, bins = np.histogram(sssp_per_sys_basic_i['Mtot_all'], bins=Mtot_bins)
        counts[1] = N_sim_i - len(sssp_per_sys_basic_i['Mtot_all'])
        Mtot_counts.append(counts/float(np.sum(counts)))
        
        # Numbers of clusters:
        counts, bins = np.histogram(sssp_per_sys_basic_i['clustertot_all'], bins=clustertot_bins)
        clustertot_counts.append(counts/float(np.sum(counts)))
        
        # Numbers of planets per cluster:
        counts, bins = np.histogram(sssp_per_sys_basic_i['pl_per_cluster_all'], bins=pl_per_cluster_bins)
        pl_per_cluster_counts.append(counts/float(np.sum(counts)))

    Mtot_counts_all.append(np.array(Mtot_counts))
    clustertot_counts_all.append(np.array(clustertot_counts))
    pl_per_cluster_counts_all.append(np.array(pl_per_cluster_counts))

Mtot_counts_all = np.array(Mtot_counts_all)
clustertot_counts_all = np.array(clustertot_counts_all)
pl_per_cluster_counts_all = np.array(pl_per_cluster_counts_all)



Mtot_counts_qtls = [np.zeros((len(Mtot_bins_mid),3)) for m in range(models)]
clustertot_counts_qtls = [np.zeros((len(clustertot_bins_mid),3)) for m in range(models)]
pl_per_cluster_counts_qtls = [np.zeros((len(pl_per_cluster_bins_mid),3)) for m in range(models)]
for m in range(models):
    for b in range(len(Mtot_bins_mid)):
        counts_bin_sorted = np.sort(Mtot_counts_all[m][:,b])
        Mtot_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
    for b in range(len(clustertot_bins_mid)):
        counts_bin_sorted = np.sort(clustertot_counts_all[m][:,b])
        clustertot_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
    for b in range(len(pl_per_cluster_bins_mid)):
        counts_bin_sorted = np.sort(pl_per_cluster_counts_all[m][:,b])
        pl_per_cluster_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])





##### To plot the simulated catalog as marginal distributions:

subdirectory = ''

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.25, 0.975, 0.925]

n_bins = 100
lw = 2 #linewidth

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size



#'''
# Multiplicities:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(Mtot_bins_mid, Mtot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(Mtot_bins_mid + model_stagger_errorbars[m], Mtot_counts_qtls[m][:,1], yerr=[Mtot_counts_qtls[m][:,1]-Mtot_counts_qtls[m][:,0], Mtot_counts_qtls[m][:,2]-Mtot_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='') # capsize=5 #label=r'16% and 84%' if m==0 else ''
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([-0.5, 10.5]) #[0, np.max(x)]
plt.ylim([0., 0.7])
plt.xlabel('Intrinsic planet multiplicity', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1), ncol=1, frameon=False, fontsize=lfs) #show the legend
plt.text(x=0.98, y=0.85, s=KS_or_AD, ha='right', fontsize=lfs, transform = ax.transAxes)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_multiplicities.pdf')
    plt.close()

# Number of clusters:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(clustertot_bins_mid, clustertot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(clustertot_bins_mid + model_stagger_errorbars[m], clustertot_counts_qtls[m][:,1], yerr=[clustertot_counts_qtls[m][:,1]-clustertot_counts_qtls[m][:,0], clustertot_counts_qtls[m][:,2]-clustertot_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 5.5])
plt.ylim([0., 1.])
plt.xlabel(r'Clusters per system $N_c$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
plt.text(x=0.98, y=0.85, s=KS_or_AD, ha='right', fontsize=lfs, transform = ax.transAxes)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_clusters.pdf')
    plt.close()

# Number of planets per cluster:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(pl_per_cluster_bins_mid, pl_per_cluster_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(pl_per_cluster_bins_mid + model_stagger_errorbars[m], pl_per_cluster_counts_qtls[m][:,1], yerr=[pl_per_cluster_counts_qtls[m][:,1]-pl_per_cluster_counts_qtls[m][:,0], pl_per_cluster_counts_qtls[m][:,2]-pl_per_cluster_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 7.5])
plt.ylim([0., 0.6])
plt.xlabel(r'Planets per cluster $N_p$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
plt.text(x=0.98, y=0.85, s=KS_or_AD, ha='right', fontsize=lfs, transform = ax.transAxes)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_planets_per_cluster.pdf')
    plt.close()
#'''

