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
import scipy.stats #for gaussian_kde functions
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *
from src.functions_plot_catalogs import *
from src.functions_plot_params import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
#loadfiles_directory = '../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/Best_models/GP_best_models/'
#savefigures_directory = '../Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/'
run_number = ''
model_name = 'Clustered_P_R_Model' + run_number

compute_ratios = compute_ratios_adjacent
AD_mod = True
weights_all = load_split_stars_weights_only()
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 'period_ratios_KS',
                 'durations_KS',
                 #'durations_norm_circ_KS',
                 #'durations_norm_circ_singles_KS',
                 #'durations_norm_circ_multis_KS',
                 'duration_ratios_nonmmr_KS',
                 'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radius_ratios_KS',
                 #'radii_partitioning_KS',
                 #'radii_monotonicity_KS',
                 #'gap_complexity_KS',
                 ]





##### To load the files with the systems with observed planets:

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
stars_cleaned = load_Kepler_stars_cleaned()
Rstar_med = np.nanmedian(stars_cleaned['radius'])
Mstar_med = np.nanmedian(stars_cleaned['mass'])
teff_med = np.nanmedian(stars_cleaned['teff'])
#bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
bp_rp_corr_med = np.nanmedian(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'])

ssk_per_sys0, ssk0 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
ssk_per_sys1, ssk1 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_max=bp_rp_corr_med) #_max=_med
ssk_per_sys2, ssk2 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=bp_rp_corr_med) #_min=_med

sss_per_sys0, sss0 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios) # combined sample
sss_per_sys1, sss1 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_max=bp_rp_corr_med, compute_ratios=compute_ratios)
sss_per_sys2, sss2 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_min=bp_rp_corr_med, compute_ratios=compute_ratios)



label1, label2 = 'bluer', 'redder'

split_sss = [sss1, sss2]
split_sss_per_sys = [sss_per_sys1, sss_per_sys2]
split_ssk = [ssk1, ssk2]
split_ssk_per_sys = [ssk_per_sys1, ssk_per_sys2]
split_names = [label1, label2]
split_linestyles = ['-', '-']
split_colors = ['b', 'r']



dists0, dists_w0 = compute_distances_sim_Kepler(sss_per_sys0, sss0, ssk_per_sys0, ssk0, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
dists1, dists_w1 = compute_distances_sim_Kepler(sss_per_sys1, sss1, ssk_per_sys1, ssk1, weights_all['bluer'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
dists2, dists_w2 = compute_distances_sim_Kepler(sss_per_sys2, sss2, ssk_per_sys2, ssk2, weights_all['redder'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)





#'''
##### To plot the simulated and Kepler catalogs as marginal distributions:

subdirectory = 'Remake_for_PaperII/hists_step_shaded/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 50
lw = 2 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
#loadfiles_directory = '../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
runs = 100

sample_names = ['all', 'bluer', 'redder']

param_vals_all = []

Mtot_bins = np.arange(10)-0.5
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = {sample: [] for sample in sample_names}
Mtot_normed_counts_all = {sample: [] for sample in sample_names}

P_bins = np.logspace(np.log10(P_min), np.log10(P_max), n_bins+1)
P_bins_mid = (P_bins[:-1] + P_bins[1:])/2.
P_counts_all = {sample: [] for sample in sample_names}

Rm_bins = np.logspace(np.log10(1.), np.log10(30.), n_bins+1)
Rm_bins_mid = (Rm_bins[:-1] + Rm_bins[1:])/2.
Rm_counts_all = {sample: [] for sample in sample_names}

tdur_bins = np.linspace(0., 15., n_bins+1)
tdur_bins_mid = (tdur_bins[:-1] + tdur_bins[1:])/2.
tdur_counts_all = {sample: [] for sample in sample_names}

tdur_tcirc_bins = np.linspace(0., 1.5, n_bins+1)
tdur_tcirc_bins_mid = (tdur_tcirc_bins[:-1] + tdur_tcirc_bins[1:])/2.
tdur_tcirc_1_counts_all = {sample: [] for sample in sample_names}
tdur_tcirc_2p_counts_all = {sample: [] for sample in sample_names}

D_bins = np.logspace(-5., -1.5, n_bins+1)
D_bins_mid = (D_bins[:-1] + D_bins[1:])/2.
D_counts_all = {sample: [] for sample in sample_names}

radii_bins = np.linspace(radii_min, radii_max, n_bins+1)
radii_bins_mid = (radii_bins[:-1] + radii_bins[1:])/2.
radii_counts_all = {sample: [] for sample in sample_names}

Rstar_bins = np.linspace(0.5, 2.5, n_bins+1)
Rstar_bins_mid = (Rstar_bins[:-1] + Rstar_bins[1:])/2.
Rstar_counts_all = {sample: [] for sample in sample_names}

D_ratio_bins = np.logspace(-1.5, 1.5, n_bins+1)
D_ratio_bins_mid = (D_ratio_bins[:-1] + D_ratio_bins[1:])/2.
D_ratio_counts_all = {sample: [] for sample in sample_names}

xi_bins = np.linspace(-0.5, 0.5, n_bins+1)
xi_bins_mid = (xi_bins[:-1] + xi_bins[1:])/2.
xi_counts_all = {sample: [] for sample in sample_names}

xi_res_bins = np.linspace(-0.5, 0.5, n_bins+1)
xi_res_bins_mid = (xi_res_bins[:-1] + xi_res_bins[1:])/2.
xi_res_counts_all = {sample: [] for sample in sample_names}

xi_nonres_bins = np.linspace(-0.5, 0.5, n_bins+1)
xi_nonres_bins_mid = (xi_nonres_bins[:-1] + xi_nonres_bins[1:])/2.
xi_nonres_counts_all = {sample: [] for sample in sample_names}

radii_partitioning_bins = np.logspace(-5., 0., n_bins+1)
radii_partitioning_bins_mid = (radii_partitioning_bins[:-1] + radii_partitioning_bins[1:])/2.
radii_partitioning_counts_all = {sample: [] for sample in sample_names}

radii_monotonicity_bins = np.linspace(-0.5, 0.6, n_bins+1)
radii_monotonicity_bins_mid = (radii_monotonicity_bins[:-1] + radii_monotonicity_bins[1:])/2.
radii_monotonicity_counts_all = {sample: [] for sample in sample_names}

gap_complexity_bins = np.linspace(0., 1., n_bins+1)
gap_complexity_bins_mid = (gap_complexity_bins[:-1] + gap_complexity_bins[1:])/2.
gap_complexity_counts_all = {sample: [] for sample in sample_names}

# Extra things to compute:
f_monotonicity_pos = [] # fraction of systems with monotonicity > 0

for i in range(1,runs+1): #range(1,runs+1)
    run_number = i
    print(i)
    
    param_vals_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    param_vals_all.append(param_vals_i)
    
    sss_per_sys0_i, sss0_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios) # combined sample
    sss_per_sys1_i, sss1_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_max=bp_rp_corr_med, compute_ratios=compute_ratios)
    sss_per_sys2_i, sss2_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_min=bp_rp_corr_med, compute_ratios=compute_ratios)
    
    dists0_i, dists_w0_i = compute_distances_sim_Kepler(sss_per_sys0_i, sss0_i, ssk_per_sys0, ssk0, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
    dists1_i, dists_w1_i = compute_distances_sim_Kepler(sss_per_sys1_i, sss1_i, ssk_per_sys1, ssk1, weights_all['bluer'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
    dists2_i, dists_w2_i = compute_distances_sim_Kepler(sss_per_sys2_i, sss2_i, ssk_per_sys2, ssk2, weights_all['redder'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
    
    samples_sss_per_sys_i = {'all': sss_per_sys0_i, 'bluer': sss_per_sys1_i, 'redder': sss_per_sys2_i}
    samples_sss_i = {'all': sss0_i, 'bluer': sss1_i, 'redder': sss2_i}

    for sample in sample_names:
        # Multiplicities:
        counts, bins = np.histogram(samples_sss_per_sys_i[sample]['Mtot_obs'], bins=Mtot_bins)
        Mtot_counts_all[sample].append(counts)
        Mtot_normed_counts_all[sample].append(counts/float(np.sum(counts)))
    
        # Periods:
        counts, bins = np.histogram(samples_sss_i[sample]['P_obs'], bins=P_bins)
        P_counts_all[sample].append(counts/float(np.sum(counts)))

        # Period ratios:
        counts, bins = np.histogram(samples_sss_i[sample]['Rm_obs'], bins=Rm_bins)
        Rm_counts_all[sample].append(counts/float(np.sum(counts)))

        # Durations:
        counts, bins = np.histogram(samples_sss_i[sample]['tdur_obs'], bins=tdur_bins)
        tdur_counts_all[sample].append(counts/float(np.sum(counts)))

        # Circular normalized durations (singles and multis):
        counts, bins = np.histogram(samples_sss_i[sample]['tdur_tcirc_1_obs'], bins=tdur_tcirc_bins)
        tdur_tcirc_1_counts_all[sample].append(counts/float(np.sum(counts)))

        counts, bins = np.histogram(samples_sss_i[sample]['tdur_tcirc_2p_obs'], bins=tdur_tcirc_bins)
        tdur_tcirc_2p_counts_all[sample].append(counts/float(np.sum(counts)))

        # Depths:
        counts, bins = np.histogram(samples_sss_i[sample]['D_obs'], bins=D_bins)
        D_counts_all[sample].append(counts/float(np.sum(counts)))

        # Planet radii:
        counts, bins = np.histogram(samples_sss_i[sample]['radii_obs'], bins=radii_bins)
        radii_counts_all[sample].append(counts/float(np.sum(counts)))

        # Stellar radii:
        counts, bins = np.histogram(samples_sss_i[sample]['Rstar_obs'], bins=Rstar_bins)
        Rstar_counts_all[sample].append(counts/float(np.sum(counts)))

        # Depth ratios:
        counts, bins = np.histogram(samples_sss_i[sample]['D_ratio_obs'], bins=D_ratio_bins)
        D_ratio_counts_all[sample].append(counts/float(np.sum(counts)))

        # Log(xi):
        counts, bins = np.histogram(np.log10(samples_sss_i[sample]['xi_obs']), bins=xi_bins)
        xi_counts_all[sample].append(counts/float(np.sum(counts)))

        # Log(xi) (res):
        counts, bins = np.histogram(np.log10(samples_sss_i[sample]['xi_res_obs']), bins=xi_res_bins)
        xi_res_counts_all[sample].append(counts/float(np.sum(counts)))

        # Log(xi) (non-res):
        counts, bins = np.histogram(np.log10(samples_sss_i[sample]['xi_nonres_obs']), bins=xi_nonres_bins)
        xi_nonres_counts_all[sample].append(counts/float(np.sum(counts)))

        # Radii partitioning:
        counts, bins = np.histogram(samples_sss_per_sys_i[sample]['radii_partitioning'], bins=radii_partitioning_bins)
        radii_partitioning_counts_all[sample].append(counts/float(np.sum(counts)))

        # Radii monotonicity:
        counts, bins = np.histogram(samples_sss_per_sys_i[sample]['radii_monotonicity'], bins=radii_monotonicity_bins)
        radii_monotonicity_counts_all[sample].append(counts/float(np.sum(counts)))

        # Gap complexity:
        counts, bins = np.histogram(samples_sss_per_sys_i[sample]['gap_complexity'], bins=gap_complexity_bins)
        gap_complexity_counts_all[sample].append(counts/float(np.sum(counts)))

    monotonicity = sss_per_sys0_i['radii_monotonicity']
    f_monotonicity_pos.append(np.sum(monotonicity > 0)/len(monotonicity))

for sample in sample_names:
    Mtot_counts_all[sample] = np.array(Mtot_counts_all[sample])
    P_counts_all[sample] = np.array(P_counts_all[sample])
    Rm_counts_all[sample] = np.array(Rm_counts_all[sample])
    tdur_counts_all[sample] = np.array(tdur_counts_all[sample])
    tdur_tcirc_1_counts_all[sample] = np.array(tdur_tcirc_1_counts_all[sample])
    tdur_tcirc_2p_counts_all[sample] = np.array(tdur_tcirc_2p_counts_all[sample])
    D_counts_all[sample] = np.array(D_counts_all[sample])
    radii_counts_all[sample] = np.array(radii_counts_all[sample])
    Rstar_counts_all[sample] = np.array(Rstar_counts_all[sample])
    D_ratio_counts_all[sample] = np.array(D_ratio_counts_all[sample])
    xi_counts_all[sample] = np.array(xi_counts_all[sample])
    xi_res_counts_all[sample] = np.array(xi_res_counts_all[sample])
    xi_nonres_counts_all[sample] = np.array(xi_nonres_counts_all[sample])
    radii_partitioning_counts_all[sample] = np.array(radii_partitioning_counts_all[sample])
    radii_monotonicity_counts_all[sample] = np.array(radii_monotonicity_counts_all[sample])
    gap_complexity_counts_all[sample] = np.array(gap_complexity_counts_all[sample])



Mtot_counts_16, Mtot_counts_84 = {sample: np.zeros(len(Mtot_bins_mid)) for sample in sample_names}, {sample: np.zeros(len(Mtot_bins_mid)) for sample in sample_names}
for b in range(len(Mtot_bins_mid)):
    for sample in sample_names:
        counts_bin_sorted = np.sort(Mtot_counts_all[sample][:,b])
        Mtot_counts_16[sample][b], Mtot_counts_84[sample][b] = counts_bin_sorted[16], counts_bin_sorted[84]

P_counts_16, P_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
Rm_counts_16, Rm_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
tdur_counts_16, tdur_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
tdur_tcirc_1_counts_16, tdur_tcirc_1_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
tdur_tcirc_2p_counts_16, tdur_tcirc_2p_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
D_counts_16, D_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
radii_counts_16, radii_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
Rstar_counts_16, Rstar_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
D_ratio_counts_16, D_ratio_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
xi_counts_16, xi_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
xi_res_counts_16, xi_res_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
xi_nonres_counts_16, xi_nonres_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
radii_partitioning_counts_16, radii_partitioning_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
radii_monotonicity_counts_16, radii_monotonicity_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
gap_complexity_counts_16, gap_complexity_counts_84 = {sample: np.zeros(n_bins) for sample in sample_names}, {sample: np.zeros(n_bins) for sample in sample_names}
for b in range(n_bins):
    for sample in sample_names:
        # Periods:
        P_counts_16[sample][b], P_counts_84[sample][b] = np.quantile(P_counts_all[sample][:,b], [0.16, 0.84])

        # Period ratios:
        Rm_counts_16[sample][b], Rm_counts_84[sample][b] = np.quantile(Rm_counts_all[sample][:,b], [0.16, 0.84])

        # Durations:
        tdur_counts_16[sample][b], tdur_counts_84[sample][b] = np.quantile(tdur_counts_all[sample][:,b], [0.16, 0.84])
        
        # Circular normalized durations:
        tdur_tcirc_1_counts_16[sample][b], tdur_tcirc_1_counts_84[sample][b] = np.quantile(tdur_tcirc_1_counts_all[sample][:,b], [0.16, 0.84])
        tdur_tcirc_2p_counts_16[sample][b], tdur_tcirc_2p_counts_84[sample][b] = np.quantile(tdur_tcirc_2p_counts_all[sample][:,b], [0.16, 0.84])
        
        # Depths:
        D_counts_16[sample][b], D_counts_84[sample][b] = np.quantile(D_counts_all[sample][:,b], [0.16, 0.84])
    
        # Planet radii:
        radii_counts_16[sample][b], radii_counts_84[sample][b] = np.quantile(radii_counts_all[sample][:,b], [0.16, 0.84])
    
        # Stellar radii:
        Rstar_counts_16[sample][b], Rstar_counts_84[sample][b] = np.quantile(Rstar_counts_all[sample][:,b], [0.16, 0.84])
    
        # Depth ratios:
        D_ratio_counts_16[sample][b], D_ratio_counts_84[sample][b] = np.quantile(D_ratio_counts_all[sample][:,b], [0.16, 0.84])
    
        # Log(xi):
        xi_counts_16[sample][b], xi_counts_84[sample][b] = np.quantile(xi_counts_all[sample][:,b], [0.16, 0.84])
    
        # Log(xi) (res):
        xi_res_counts_16[sample][b], xi_res_counts_84[sample][b] = np.quantile(xi_res_counts_all[sample][:,b], [0.16, 0.84])
    
        # Log(xi) (non-res):
        xi_nonres_counts_16[sample][b], xi_nonres_counts_84[sample][b] = np.quantile(xi_nonres_counts_all[sample][:,b], [0.16, 0.84])

        # Radii partitioning:
        radii_partitioning_counts_16[sample][b], radii_partitioning_counts_84[sample][b] = np.quantile(radii_partitioning_counts_all[sample][:,b], [0.16, 0.84])
        
        # Radii monotonicity:
        radii_monotonicity_counts_16[sample][b], radii_monotonicity_counts_84[sample][b] = np.quantile(radii_monotonicity_counts_all[sample][:,b], [0.16, 0.84])
        
        # Gap complexity:
        gap_complexity_counts_16[sample][b], gap_complexity_counts_84[sample][b] = np.quantile(gap_complexity_counts_all[sample][:,b], [0.16, 0.84])

#####





# To make a 'plot' listing the model parameters:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,1,left=fig_lbrt[0],bottom=fig_lbrt[1],right=fig_lbrt[2],top=fig_lbrt[3],wspace=0.1,hspace=0.1)
nrows = 8
for i,param in enumerate(param_vals):
    plt.figtext(x=0.05+0.35*int(i/float(nrows)), y=0.875-0.1*(i%nrows), s=r'%s = %s' % (param_symbols[param], np.round(param_vals[param],3)), fontsize=lfs)
if savefigures == True:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_sim_params.pdf')
    plt.close()

# Multiplicities:
plot_fig_counts_hist_simple(fig_size, [sss_per_sys['Mtot_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['Mtot_obs'] for ssk_per_sys in split_ssk_per_sys], x_min=0, x_llim=0.5, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ms_Kep=['x','x'], lw=lw, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], xlabel_text='Observed planets per system', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    label_this = r'16% and 84%' if i==0 else None
    plt.plot(Mtot_bins_mid, Mtot_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label=label_this)
    plt.plot(Mtot_bins_mid, Mtot_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--')
plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_multiplicities_compare.pdf')
    plt.close()

# Periods:
plot_fig_pdf_simple(fig_size, [sss['P_obs'] for sss in split_sss], [ssk['P_obs'] for ssk in split_ssk], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(P_bins_mid, P_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(P_bins_mid, P_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_compare.pdf')
    plt.close()

# Period ratios (all, with some upper cut-off):
R_max_cut = np.max(Rm_bins)
plot_fig_pdf_simple(fig_size, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut] for sss in split_sss], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut] for ssk in split_ssk], x_min=np.min(Rm_bins), x_max=R_max_cut, n_bins=n_bins, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$\mathcal{P} = P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(Rm_bins_mid, Rm_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(Rm_bins_mid, Rm_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periodratios_compare.pdf')
    plt.close()

# Transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_obs'] for sss in split_sss], [ssk['tdur_obs'] for ssk in split_ssk], x_min=np.min(tdur_bins), x_max=np.max(tdur_bins), n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(tdur_bins_mid, tdur_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(tdur_bins_mid, tdur_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_durations_compare.pdf')
    plt.close()

# Circular normalized transit durations (separate singles and multis):
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_1_obs'] for sss in split_sss], [ssk['tdur_tcirc_1_obs'] for ssk in split_ssk], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], extra_text='Observed singles', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_compare.pdf')
    plt.close()

plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_2p_obs'] for sss in split_sss], [ssk['tdur_tcirc_2p_obs'] for ssk in split_ssk], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], extra_text='Observed multis', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_compare.pdf')
    plt.close()


# Transit depths:
plot_fig_pdf_simple(fig_size, [sss['D_obs'] for sss in split_sss], [ssk['D_obs'] for ssk in split_ssk], x_min=np.min(D_bins), x_max=np.max(D_bins), n_bins=n_bins, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(D_bins_mid, D_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(D_bins_mid, D_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depths_compare.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_simple(fig_size, [sss['radii_obs'] for sss in split_sss], [ssk['radii_obs'] for ssk in split_ssk], x_min=radii_min, x_max=radii_max, n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$R_p (R_\oplus)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(radii_bins_mid, radii_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(radii_bins_mid, radii_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_compare.pdf')
    plt.close()

# Stellar radii:
plot_fig_pdf_simple(fig_size, [sss_per_sys['Rstar_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['Rstar_obs'] for ssk_per_sys in split_ssk_per_sys], x_min=np.min(Rstar_bins), x_max=np.max(Rstar_bins), n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(Rstar_bins_mid, Rstar_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(Rstar_bins_mid, Rstar_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_stellar_radii_compare.pdf')
    plt.close()

# Transit depth ratios:
plot_fig_pdf_simple(fig_size, [sss['D_ratio_obs'] for sss in split_sss], [ssk['D_ratio_obs'] for ssk in split_ssk], x_min=np.min(D_ratio_bins), x_max=np.max(D_ratio_bins), n_bins=n_bins, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(D_ratio_bins_mid, D_ratio_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(D_ratio_bins_mid, D_ratio_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depthratios_compare.pdf')
    plt.close()

# Log(xi):
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_obs']) for sss in split_sss], [np.log10(ssk['xi_obs']) for ssk in split_ssk], x_min=np.min(xi_bins), x_max=np.max(xi_bins), n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(xi_bins_mid, xi_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(xi_bins_mid, xi_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_all_compare.pdf')
    plt.close()

# Log(xi) (not near MMR):
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_nonres_obs']) for sss in split_sss], [np.log10(ssk['xi_nonres_obs']) for ssk in split_ssk], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], extra_text='Not near MMR', xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(xi_nonres_bins_mid, xi_nonres_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(xi_nonres_bins_mid, xi_nonres_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_nonmmr_compare.pdf')
    plt.close()

# Log(xi) (near MMR):
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_res_obs']) for sss in split_sss], [np.log10(ssk['xi_res_obs']) for ssk in split_ssk], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], extra_text='Near MMR', xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(xi_res_bins_mid, xi_res_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(xi_res_bins_mid, xi_res_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_mmr_compare.pdf')
    plt.close()

### GF2020 metrics:
# Planet radii partitioning:
plot_fig_pdf_simple(fig_size, [sss_per_sys['radii_partitioning'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['radii_partitioning'] for ssk_per_sys in split_ssk_per_sys], x_min=1e-5, x_max=1., n_bins=n_bins, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=['Simulated', None], labels_Kep=['Kepler', None], xlabel_text=r'$\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(radii_partitioning_bins_mid, radii_partitioning_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16% and 84%' if i==0 else '')
    plt.plot(radii_partitioning_bins_mid, radii_partitioning_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='')
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_partitioning_compare.pdf')
    plt.close()

# Planet radii monotonicity:
plot_fig_pdf_simple(fig_size, [sss_per_sys['radii_monotonicity'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['radii_monotonicity'] for ssk_per_sys in split_ssk_per_sys], x_min=-0.5, x_max=0.6, n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(radii_monotonicity_bins_mid, radii_monotonicity_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(radii_monotonicity_bins_mid, radii_monotonicity_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_monotonicity_compare.pdf')
    plt.close()

# Gap complexity:
plot_fig_pdf_simple(fig_size, [sss_per_sys['gap_complexity'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['gap_complexity'] for ssk_per_sys in split_ssk_per_sys], x_min=0., x_max=1., n_bins=n_bins, log_x=False, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.plot(gap_complexity_bins_mid, gap_complexity_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='16')
    plt.plot(gap_complexity_bins_mid, gap_complexity_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_gap_complexity_compare.pdf')
    plt.close()

plt.show()
#plt.close()





##### To plot individual panels for each of the 'all', 'bluer', and 'redder' samples:
'''
fig_size = (18,3)
fig_lbrt, wspace = [0.1, 0.3, 0.98, 0.92], 0.3

sample_sss = [sss0, sss1, sss2]
sample_sss_per_sys = [sss_per_sys0, sss_per_sys1, sss_per_sys2]
sample_ssk = [ssk0, ssk1, ssk2]
sample_ssk_per_sys = [ssk_per_sys0, ssk_per_sys1, ssk_per_sys2]
sample_linestyles = ['-', '-', '-']
sample_colors = ['k', 'b', 'r']
    
# Multiplicities:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_counts_hist_simple(ax, [sss_per_sys['Mtot_obs']], [ssk_per_sys['Mtot_obs']], x_min=0, x_llim=0.5, x_ulim=6.5, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ms_Kep=['x'], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text='Observed planets per system', ylabel_text='Number' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(Mtot_bins_mid, Mtot_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label=r'16% and 84%')
    plt.plot(Mtot_bins_mid, Mtot_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--')
    if s==0:
        plt.legend(loc='lower left', bbox_to_anchor=(-0.02,-0.04), ncol=1, frameon=False, fontsize=lfs) #show the legend
    plt.text(x=0.98, y=0.85, s=sample.title(), ha='right', fontsize=lfs, color=cl, transform=ax.transAxes)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_multiplicities_samples_compare.pdf')
    plt.close()

# Periods:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss['P_obs']], [ssk['P_obs']], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs, legend=True if s==0 else False)
    plt.plot(P_bins_mid, P_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(P_bins_mid, P_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_samples_compare.pdf')
    plt.close()

# Period ratios (all, with some upper cut-off):
R_max_cut = np.max(Rm_bins)
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut]], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=np.min(Rm_bins), x_max=R_max_cut, n_bins=n_bins, log_x=True, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$\mathcal{P} = P_{i+1}/P_i$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(Rm_bins_mid, Rm_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(Rm_bins_mid, Rm_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periodratios_samples_compare.pdf')
    plt.close()

# Transit durations:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss['tdur_obs']], [ssk['tdur_obs']], x_min=np.min(tdur_bins), x_max=np.max(tdur_bins), n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$t_{\rm dur}$ (hrs)', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(tdur_bins_mid, tdur_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(tdur_bins_mid, tdur_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_durations_samples_compare.pdf')
    plt.close()

# Circular normalized transit durations (separate singles and multis):
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss['tdur_tcirc_1_obs']], [ssk['tdur_tcirc_1_obs']], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], extra_text='Observed singles', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_samples_compare.pdf')
    plt.close()

fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss['tdur_tcirc_2p_obs']], [ssk['tdur_tcirc_2p_obs']], x_min=0., x_max=1.5, n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], extra_text='Observed multis', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_samples_compare.pdf')
    plt.close()


# Transit depths
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss['D_obs']], [ssk['D_obs']], x_min=np.min(D_bins), x_max=np.max(D_bins), log_x=True, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$\delta$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(D_bins_mid, D_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(D_bins_mid, D_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depths_samples_compare.pdf')
    plt.close()

# Planet radii:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss['radii_obs']], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$R_p (R_\oplus)$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(radii_bins_mid, radii_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(radii_bins_mid, radii_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_samples_compare.pdf')
    plt.close()

# Stellar radii:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss_per_sys['Rstar_obs']], [ssk_per_sys['Rstar_obs']], x_min=np.min(Rstar_bins), x_max=np.max(Rstar_bins), n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$R_\star (R_\odot)$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(Rstar_bins_mid, Rstar_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(Rstar_bins_mid, Rstar_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_stellar_radii_samples_compare.pdf')
    plt.close()

# Transit depth ratios:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss['D_ratio_obs']], [ssk['D_ratio_obs']], x_min=np.min(D_ratio_bins), x_max=np.max(D_ratio_bins), n_bins=n_bins, log_x=True, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(D_ratio_bins_mid, D_ratio_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(D_ratio_bins_mid, D_ratio_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depthratios_samples_compare.pdf')
    plt.close()

# Log(xi):
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [np.log10(sss['xi_obs'])], [np.log10(ssk['xi_obs'])], x_min=np.min(xi_bins), x_max=np.max(xi_bins), n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$\log{\/xi}$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(xi_bins_mid, xi_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(xi_bins_mid, xi_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_all_samples_compare.pdf')
    plt.close()

# Log(xi) (not near MMR):
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [np.log10(sss['xi_nonres_obs'])], [np.log10(ssk['xi_nonres_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], extra_text='Not near MMR', xlabel_text=r'$\log{\/xi}$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(xi_nonres_bins_mid, xi_nonres_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(xi_nonres_bins_mid, xi_nonres_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_nonmmr_samples_compare.pdf')
    plt.close()

# Log(xi) (near MMR):
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [np.log10(sss['xi_res_obs'])], [np.log10(ssk['xi_res_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], extra_text='Near MMR', xlabel_text=r'$\log{\/xi}$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(xi_res_bins_mid, xi_res_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(xi_res_bins_mid, xi_res_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_mmr_samples_compare.pdf')
    plt.close()

### GF2020 metrics:
# Planet radii partitioning:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss_per_sys['radii_partitioning']], [ssk_per_sys['radii_partitioning']], x_min=1e-5, x_max=1., n_bins=n_bins, log_x=True, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$\mathcal{Q}_R$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(radii_partitioning_bins_mid, radii_partitioning_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(radii_partitioning_bins_mid, radii_partitioning_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_partitioning_samples_compare.pdf')
    plt.close()

# Planet radii monotonicity:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss_per_sys['radii_monotonicity']], [ssk_per_sys['radii_monotonicity']], x_min=-0.5, x_max=0.6, n_bins=n_bins, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$\mathcal{M}_R$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(radii_monotonicity_bins_mid, radii_monotonicity_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(radii_monotonicity_bins_mid, radii_monotonicity_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_monotonicity_samples_compare.pdf')
    plt.close()

# Gap complexity:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=wspace, hspace=0)
for s,sample in enumerate(sample_names):
    sss_per_sys, sss = sample_sss_per_sys[s], sample_sss[s]
    ssk_per_sys, ssk = sample_ssk_per_sys[s], sample_ssk[s]
    ls, cl = sample_linestyles[s], sample_colors[s]
    ax = plt.subplot(plot[0,s])
    plot_panel_pdf_simple(ax, [sss_per_sys['gap_complexity']], [ssk_per_sys['gap_complexity']], x_min=0., x_max=1., n_bins=n_bins, log_x=False, c_sim=[cl], c_Kep=[cl], ls_sim=[ls], ls_Kep=[ls], lw=lw, labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'$\mathcal{C}$', ylabel_text='Fraction' if s==0 else '', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(gap_complexity_bins_mid, gap_complexity_counts_16[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='16')
    plt.plot(gap_complexity_bins_mid, gap_complexity_counts_84[sample], drawstyle='steps-mid', color=cl, lw=1, ls='--', label='84')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_gap_complexity_samples_compare.pdf')
    plt.close()

plt.show()
#plt.close()
'''





##### To remake some panels for He, Ford, Ragozzine (2020) (Paper II):
##### Will plot central 68% as shaded regions and Kepler data as solid histograms:

fig_size = (8,6)
fig_lbrt = [0.15, 0.15, 0.95, 0.95]

# Periods:
plot_fig_pdf_simple(fig_size, [ssk['P_obs'] for ssk in split_ssk], [], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=['Kepler',None], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=False, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    label_this = r'Simulated 16-84%' if i==0 else ''
    plt.fill_between(P_bins_mid, P_counts_16[sample], P_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha, label=label_this)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_compare.pdf')
    plt.close()

# Period ratios (all, with some upper cut-off):
R_max_cut = np.max(Rm_bins)
plot_fig_pdf_simple(fig_size, [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut] for ssk in split_ssk], [], x_min=np.min(Rm_bins), x_max=R_max_cut, n_bins=n_bins, log_x=True, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$\mathcal{P} = P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(Rm_bins_mid, Rm_counts_16[sample], Rm_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periodratios_compare.pdf')
    plt.close()

# Transit durations:
plot_fig_pdf_simple(fig_size, [ssk['tdur_obs'] for ssk in split_ssk], [], x_min=np.min(tdur_bins), x_max=np.max(tdur_bins), n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(tdur_bins_mid, tdur_counts_16[sample], tdur_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_durations_compare.pdf')
    plt.close()

# Circular normalized transit durations (separate singles and multis):
plot_fig_pdf_simple(fig_size, [ssk['tdur_tcirc_1_obs'] for ssk in split_ssk], [], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, extra_text='Observed singles', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_16[sample], tdur_tcirc_1_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_compare.pdf')
    plt.close()

plot_fig_pdf_simple(fig_size, [ssk['tdur_tcirc_2p_obs'] for ssk in split_ssk], [], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, extra_text='Observed multis', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_16[sample], tdur_tcirc_2p_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_compare.pdf')
    plt.close()

# Transit depths:
plot_fig_pdf_simple(fig_size, [ssk['D_obs'] for ssk in split_ssk], [], x_min=np.min(D_bins), x_max=np.max(D_bins), n_bins=n_bins, log_x=True, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(D_bins_mid, D_counts_16[sample], D_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depths_compare.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_simple(fig_size, [ssk['radii_obs'] for ssk in split_ssk], [], x_min=radii_min, x_max=radii_max, n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xlabel_text=r'$R_p (R_\oplus)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(radii_bins_mid, radii_counts_16[sample], radii_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_compare.pdf')
    plt.close()

# Stellar radii:
plot_fig_pdf_simple(fig_size, [ssk_per_sys['Rstar_obs'] for ssk_per_sys in split_ssk_per_sys], [], x_min=np.min(Rstar_bins), x_max=np.max(Rstar_bins), n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(Rstar_bins_mid, Rstar_counts_16[sample], Rstar_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_stellar_radii_compare.pdf')
    plt.close()

# Transit depth ratios:
plot_fig_pdf_simple(fig_size, [ssk['D_ratio_obs'] for ssk in split_ssk], [], x_min=np.min(D_ratio_bins), x_max=np.max(D_ratio_bins), n_bins=n_bins, log_x=True, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(D_ratio_bins_mid, D_ratio_counts_16[sample], D_ratio_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depthratios_compare.pdf')
    plt.close()

# Log(xi):
plot_fig_pdf_simple(fig_size, [np.log10(ssk['xi_obs']) for ssk in split_ssk], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(xi_bins_mid, xi_counts_16[sample], xi_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_all_compare.pdf')
    plt.close()

# Log(xi) (not near MMR):
plot_fig_pdf_simple(fig_size, [np.log10(ssk['xi_nonres_obs']) for ssk in split_ssk], [], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, extra_text='Not near MMR', xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(xi_nonres_bins_mid, xi_nonres_counts_16[sample], xi_nonres_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_nonmmr_compare.pdf')
    plt.close()

# Log(xi) (near MMR):
plot_fig_pdf_simple(fig_size, [np.log10(ssk['xi_res_obs']) for ssk in split_ssk], [], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, extra_text='Near MMR', xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(xi_res_bins_mid, xi_res_counts_16[sample], xi_res_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_mmr_compare.pdf')
    plt.close()

### GF2020 metrics:
# Planet radii partitioning:
plot_fig_pdf_simple(fig_size, [ssk_per_sys['radii_partitioning'] for ssk_per_sys in split_ssk_per_sys], [], x_min=1e-5, x_max=1., n_bins=n_bins, log_x=True, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=['Kepler', None], xlabel_text=r'$\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    label_this = r'Simulated 16-84%' if i==0 else ''
    plt.fill_between(radii_partitioning_bins_mid, radii_partitioning_counts_16[sample], radii_partitioning_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha, label=label_this)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_partitioning_compare.pdf')
    plt.close()

# Planet radii monotonicity:
plot_fig_pdf_simple(fig_size, [ssk_per_sys['radii_monotonicity'] for ssk_per_sys in split_ssk_per_sys], [], x_min=-0.5, x_max=0.6, n_bins=n_bins, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xlabel_text=r'$\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(radii_monotonicity_bins_mid, radii_monotonicity_counts_16[sample], radii_monotonicity_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_monotonicity_compare.pdf')
    plt.close()

# Gap complexity:
plot_fig_pdf_simple(fig_size, [ssk_per_sys['gap_complexity'] for ssk_per_sys in split_ssk_per_sys], [], x_min=0., x_max=1., n_bins=n_bins, log_x=False, c_sim=split_colors, ls_sim=split_linestyles, lw=lw, labels_sim=split_names, xlabel_text=r'$\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    plt.fill_between(gap_complexity_bins_mid, gap_complexity_counts_16[sample], gap_complexity_counts_84[sample], step='mid', color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_gap_complexity_compare.pdf')
    plt.close()

plt.show()
#plt.close()





##### Circular normalized durations for singles and multis with PDFs and CDFs:
#'''
fig = plt.figure(figsize=(8,5))
plot = GridSpec(2,1,left=0.15,bottom=0.2,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0]) # CDF
plot_panel_cdf_simple(ax, [sss['tdur_tcirc_1_obs'] for sss in split_sss], [ssk['tdur_tcirc_1_obs'] for ssk in split_ssk], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=['--','--'], lw=lw, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], extra_text='Observed singles', xlabel_text='', afs=afs, tfs=tfs, lfs=lfs, label_dist=True)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.75), ncol=1, frameon=False, fontsize=lfs)
plt.xticks([])
ax = plt.subplot(plot[1,0]) # PDF
plot_panel_pdf_simple(ax, [sss['tdur_tcirc_1_obs'] for sss in split_sss], [ssk['tdur_tcirc_1_obs'] for ssk in split_ssk], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs)
for i,sample in enumerate(split_names):
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='')
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='')
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_compare_with_CDFs.pdf')
    plt.close()



fig = plt.figure(figsize=(8,5))
plot = GridSpec(2,1,left=0.15,bottom=0.2,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0]) # CDF
plot_panel_cdf_simple(ax, [sss['tdur_tcirc_2p_obs'] for sss in split_sss], [ssk['tdur_tcirc_2p_obs'] for ssk in split_ssk], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=['--','--'], lw=lw, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], extra_text='Observed multis', xlabel_text='', afs=afs, tfs=tfs, lfs=lfs, label_dist=True)
plt.xticks([])
ax = plt.subplot(plot[1,0]) # PDF
plot_panel_pdf_simple(ax, [sss['tdur_tcirc_2p_obs'] for sss in split_sss], [ssk['tdur_tcirc_2p_obs'] for ssk in split_ssk], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), y_max=0.2, n_bins=n_bins, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs)
for i,sample in enumerate(split_names):
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='')
    plt.plot(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label='')
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_compare_with_CDFs.pdf')
    plt.close()
plt.show()
#'''

