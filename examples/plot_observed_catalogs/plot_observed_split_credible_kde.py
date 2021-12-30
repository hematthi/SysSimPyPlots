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
from scipy.stats import gaussian_kde as KDE #for gaussian_kde functions
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
##### To first compute the KDE objects for each marginal distribution from Kepler:

sample_names = ['all', 'bluer', 'redder']
ssk_samples = {'all': ssk0, 'bluer': ssk1, 'redder': ssk2}
ssk_per_sys_samples = {'all': ssk_per_sys0, 'bluer': ssk_per_sys1, 'redder': ssk_per_sys2}

P_kde_Kep = {sample: KDE(np.log10(ssk_samples[sample]['P_obs'])) for sample in sample_names}
Rm_kde_Kep = {sample: KDE(np.log10(ssk_samples[sample]['Rm_obs'])) for sample in sample_names}
tdur_kde_Kep = {sample: KDE(ssk_samples[sample]['tdur_obs']) for sample in sample_names}
tdur_tcirc_1_kde_Kep = {sample: KDE(ssk_samples[sample]['tdur_tcirc_1_obs']) for sample in sample_names}
tdur_tcirc_2p_kde_Kep = {sample: KDE(ssk_samples[sample]['tdur_tcirc_2p_obs']) for sample in sample_names}
D_kde_Kep = {sample: KDE(np.log10(ssk_samples[sample]['D_obs'])) for sample in sample_names}
radii_kde_Kep = {sample: KDE(ssk_samples[sample]['radii_obs']) for sample in sample_names}
Rstar_kde_Kep = {sample: KDE(ssk_samples[sample]['Rstar_obs']) for sample in sample_names}
D_ratio_kde_Kep = {sample: KDE(np.log10(ssk_samples[sample]['D_ratio_obs'])) for sample in sample_names}
xi_Kep = {sample: np.log10(ssk_samples[sample]['xi_obs']) for sample in sample_names}
xi_kde_Kep = {sample: KDE(xi_Kep[sample][np.isfinite(xi_Kep[sample])]) for sample in sample_names}
xi_res_Kep = {sample: np.log10(ssk_samples[sample]['xi_res_obs']) for sample in sample_names}
xi_res_kde_Kep = {sample: KDE(xi_res_Kep[sample][np.isfinite(xi_res_Kep[sample])]) for sample in sample_names}
xi_nonres_Kep = {sample: np.log10(ssk_samples[sample]['xi_nonres_obs']) for sample in sample_names}
xi_nonres_kde_Kep = {sample: KDE(xi_nonres_Kep[sample][np.isfinite(xi_nonres_Kep[sample])]) for sample in sample_names}
radii_partitioning_kde_Kep = {sample: KDE(np.log10(ssk_per_sys_samples[sample]['radii_partitioning'])) for sample in sample_names}
radii_monotonicity_kde_Kep = {sample: KDE(ssk_per_sys_samples[sample]['radii_monotonicity']) for sample in sample_names}
gap_complexity_kde_Kep = {sample: KDE(ssk_per_sys_samples[sample]['gap_complexity']) for sample in sample_names}




##### To load and compute the KDE objects for a large number of models:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
#loadfiles_directory = '../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
runs = 100

param_vals_all = []

Mtot_bins = np.arange(10)-0.5
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = {sample: [] for sample in sample_names}
Mtot_normed_counts_all = {sample: [] for sample in sample_names}

P_kde_all = {sample: [] for sample in sample_names} # kde on log values
Rm_kde_all = {sample: [] for sample in sample_names} # kde on log values
tdur_kde_all = {sample: [] for sample in sample_names}
tdur_tcirc_1_kde_all = {sample: [] for sample in sample_names}
tdur_tcirc_2p_kde_all = {sample: [] for sample in sample_names}
D_kde_all = {sample: [] for sample in sample_names} # kde on log values
radii_kde_all = {sample: [] for sample in sample_names}
Rstar_kde_all = {sample: [] for sample in sample_names}
D_ratio_kde_all = {sample: [] for sample in sample_names} # kde on log values
xi_kde_all = {sample: [] for sample in sample_names}
xi_res_kde_all = {sample: [] for sample in sample_names}
xi_nonres_kde_all = {sample: [] for sample in sample_names}
radii_partitioning_kde_all = {sample: [] for sample in sample_names} # kde on log values
radii_monotonicity_kde_all = {sample: [] for sample in sample_names}
gap_complexity_kde_all = {sample: [] for sample in sample_names}

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
        P_kde = KDE(np.log10(samples_sss_i[sample]['P_obs']))
        P_kde_all[sample].append(P_kde)

        # Period ratios:
        Rm_kde = KDE(np.log10(samples_sss_i[sample]['Rm_obs']))
        Rm_kde_all[sample].append(Rm_kde)

        # Durations:
        tdur_kde = KDE(samples_sss_i[sample]['tdur_obs'])
        tdur_kde_all[sample].append(tdur_kde)

        # Circular normalized durations (singles and multis):
        tdur_tcirc_1_kde = KDE(samples_sss_i[sample]['tdur_tcirc_1_obs'])
        tdur_tcirc_2p_kde = KDE(samples_sss_i[sample]['tdur_tcirc_2p_obs'])
        tdur_tcirc_1_kde_all[sample].append(tdur_tcirc_1_kde)
        tdur_tcirc_2p_kde_all[sample].append(tdur_tcirc_2p_kde)

        # Depths:
        D_kde = KDE(np.log10(samples_sss_i[sample]['D_obs']))
        D_kde_all[sample].append(D_kde)

        # Planet radii:
        radii_kde = KDE(samples_sss_i[sample]['radii_obs'])
        radii_kde_all[sample].append(radii_kde)

        # Stellar radii:
        Rstar_kde = KDE(samples_sss_i[sample]['Rstar_obs'])
        Rstar_kde_all[sample].append(Rstar_kde)

        # Depth ratios:
        D_ratio_kde = KDE(np.log10(samples_sss_i[sample]['D_ratio_obs']))
        D_ratio_kde_all[sample].append(D_ratio_kde)

        # Log(xi):
        xi = np.log10(samples_sss_i[sample]['xi_obs'])
        xi_kde = KDE(xi[np.isfinite(xi)])
        xi_kde_all[sample].append(xi_kde)

        # Log(xi) (res):
        xi_res = np.log10(samples_sss_i[sample]['xi_res_obs'])
        xi_res_kde = KDE(xi_res[np.isfinite(xi_res)])
        xi_res_kde_all[sample].append(xi_res_kde)

        # Log(xi) (non-res):
        xi_nonres = np.log10(samples_sss_i[sample]['xi_nonres_obs'])
        xi_nonres_kde = KDE(xi_nonres[np.isfinite(xi_nonres)])
        xi_nonres_kde_all[sample].append(xi_nonres_kde)

        # Radii partitioning:
        radii_partitioning_kde = KDE(np.log10(samples_sss_per_sys_i[sample]['radii_partitioning']))
        radii_partitioning_kde_all[sample].append(radii_partitioning_kde)

        # Radii monotonicity:
        radii_monotonicity_kde = KDE(samples_sss_per_sys_i[sample]['radii_monotonicity'])
        radii_monotonicity_kde_all[sample].append(radii_monotonicity_kde)

        # Gap complexity:
        gap_complexity_kde = KDE(samples_sss_per_sys_i[sample]['gap_complexity'])
        gap_complexity_kde_all[sample].append(gap_complexity_kde)

for sample in sample_names:
    Mtot_counts_all[sample] = np.array(Mtot_counts_all[sample])
Mtot_counts_16, Mtot_counts_84 = {sample: np.zeros(len(Mtot_bins_mid)) for sample in sample_names}, {sample: np.zeros(len(Mtot_bins_mid)) for sample in sample_names}
for b in range(len(Mtot_bins_mid)):
    for sample in sample_names:
        counts_bin_sorted = np.sort(Mtot_counts_all[sample][:,b])
        Mtot_counts_16[sample][b], Mtot_counts_84[sample][b] = counts_bin_sorted[16], counts_bin_sorted[84]





##### To plot the simulated and Kepler catalogs as KDEs:

subdirectory = 'Remake_for_PaperII/kde_lines/kNN_bw/' #'Remake_for_PaperII/kde_shaded/'

fig_size = (8,6) #(8,3) # size of each panel (figure)
fig_lbrt = [0.15, 0.15, 0.95, 0.95] #[0.15, 0.3, 0.95, 0.925]

pts = 201 # number of points to evaluate each kde
kNN_factor = 5 # len(data)/kNN_factor = k Nearest Neighbors
lw_Kep = 3 # linewidth
lw_sim = 0.1
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

# Function to compute the KDE using an adaptive bandwidth (kNN):
def kde_kNN_bw(kde, x_axis, kNN_factor=5):
    # Compute the KDE along an axis ('x_axis') given a KDE object ('kde') with adaptive bandwidth using k nearest neighbors (kNN), where kNN = (number of data points)/kNN_factor
    kde_pts, bw_pts = np.zeros(len(x_axis)), np.zeros(len(x_axis))
    x_data = kde.dataset[0]
    kNN = int(np.round(len(x_data)/kNN_factor)) # number of nearest neighbors
    for p,x in enumerate(x_axis):
        idx_kNN = np.argsort(np.abs(x_data - x))[:kNN] # indices of kNN
        x_kNN = x_data[idx_kNN] # kNN points
        bw = np.max(x_kNN) - np.min(x_kNN) # bandwidth as range of kNN points
        
        kde.set_bandwidth(bw_method=bw)
        kde_pts[p] = kde(x)[0] # evaluate KDE at point
        bw_pts[p] = bw
    return kde_pts, bw_pts



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
plot_fig_counts_hist_simple(fig_size, [sss_per_sys['Mtot_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['Mtot_obs'] for ssk_per_sys in split_ssk_per_sys], x_min=0, x_llim=0.5, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ms_Kep=['x','x'], lw=1, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], xlabel_text='Observed planets per system', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    label_this = r'16% and 84%' if i==0 else None
    plt.plot(Mtot_bins_mid, Mtot_counts_16[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--', label=label_this)
    plt.plot(Mtot_bins_mid, Mtot_counts_84[sample], drawstyle='steps-mid', color=split_colors[i], lw=1, ls='--')
plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_multiplicities_compare.pdf')
    plt.close()

# Periods:
P_axis = np.logspace(np.log10(P_min), np.log10(P_max), pts)
ax = plot_fig_pdf_simple(fig_size, [], [], x_min=P_min, x_max=P_max, y_min=0, y_max=1, log_x=True, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = Rm_kde_Kep[sample](np.log10(Rm_axis))
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(P_kde_Kep[sample], np.log10(P_axis), kNN_factor=kNN_factor)
    plt.plot(P_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i], label='Kepler' if i==0 else '')
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = P_kde_all[sample][n](np.log10(P_axis))
        kde_pts, bw_pts = kde_kNN_bw(P_kde_all[sample][n], np.log10(P_axis), kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(P_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i], label=r'Simulated draws' if i==0 and n==0 else '')
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(P_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha, label=r'Simulated 16-84%' if i==0 else '')
#ax.set_yscale('log')
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_compare.pdf')
    plt.close()

# Period ratios (all, with some upper cut-off):
Rm_axis = np.logspace(np.log10(1.), np.log10(30.), pts)
R_max_cut = np.max(Rm_axis)
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(Rm_axis), x_max=R_max_cut, y_max=5, log_x=True, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$\mathcal{P} = P_{i+1}/P_i$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = Rm_kde_Kep[sample](np.log10(Rm_axis))
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(Rm_kde_Kep[sample], np.log10(Rm_axis), kNN_factor=kNN_factor)
    plt.plot(Rm_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])

    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = Rm_kde_all[sample][n](np.log10(Rm_axis))
        kde_pts, bw_pts = kde_kNN_bw(Rm_kde_all[sample][n], np.log10(Rm_axis), kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(Rm_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(Rm_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periodratios_compare.pdf')
    plt.close()

# Transit durations:
tdur_axis = np.linspace(0., 12., pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(tdur_axis), x_max=np.max(tdur_axis), y_max=0.4, xlabel_text=r'$t_{\rm dur}$ (hrs)', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = tdur_kde_Kep[sample](tdur_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(tdur_kde_Kep[sample], tdur_axis, kNN_factor=kNN_factor)
    plt.plot(tdur_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = tdur_kde_all[sample][n](tdur_axis)
        kde_pts, bw_pts = kde_kNN_bw(tdur_kde_all[sample][n], tdur_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(tdur_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(tdur_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_durations_compare.pdf')
    plt.close()

# Circular normalized transit durations (separate singles and multis):
tdur_tcirc_axis = np.linspace(0., 1.5, pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(tdur_tcirc_axis), x_max=np.max(tdur_tcirc_axis), y_max=5, extra_text='Observed singles', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = tdur_tcirc_1_kde_Kep[sample](tdur_tcirc_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(tdur_tcirc_1_kde_Kep[sample], tdur_tcirc_axis, kNN_factor=kNN_factor)
    plt.plot(tdur_tcirc_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = tdur_tcirc_1_kde_all[sample][n](tdur_tcirc_axis)
        kde_pts, bw_pts = kde_kNN_bw(tdur_tcirc_1_kde_all[sample][n], tdur_tcirc_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(tdur_tcirc_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    #kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    plt.fill_between(tdur_tcirc_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_compare.pdf')
    plt.close()

plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(tdur_tcirc_axis), x_max=np.max(tdur_tcirc_axis), y_max=5, extra_text='Observed multis', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = tdur_tcirc_2p_kde_Kep[sample](tdur_tcirc_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(tdur_tcirc_2p_kde_Kep[sample], tdur_tcirc_axis, kNN_factor=kNN_factor)
    plt.plot(tdur_tcirc_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = tdur_tcirc_2p_kde_all[sample][n](tdur_tcirc_axis)
        kde_pts, bw_pts = kde_kNN_bw(tdur_tcirc_2p_kde_all[sample][n], tdur_tcirc_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(tdur_tcirc_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    #kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    plt.fill_between(tdur_tcirc_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_compare.pdf')
    plt.close()

# Transit depths:
D_axis = np.logspace(-5., -1.5, pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(D_axis), x_max=np.max(D_axis), y_max=1.4, log_x=True, xlabel_text=r'$\delta$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = D_kde_Kep[sample](np.log10(D_axis))
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(D_kde_Kep[sample], np.log10(D_axis), kNN_factor=kNN_factor)
    plt.plot(D_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = D_kde_all[sample][n](np.log10(D_axis))
        kde_pts, bw_pts = kde_kNN_bw(D_kde_all[sample][n], np.log10(D_axis), kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(D_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(D_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depths_compare.pdf')
    plt.close()

# Planet radii:
radii_axis = np.linspace(radii_min, radii_max, pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=radii_min, x_max=radii_max, y_max=0.5, xlabel_text=r'$R_p (R_\oplus)$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = radii_kde_Kep[sample](radii_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(radii_kde_Kep[sample], radii_axis, kNN_factor=kNN_factor)
    plt.plot(radii_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = radii_kde_all[sample][n](radii_axis)
        kde_pts, bw_pts = kde_kNN_bw(radii_kde_all[sample][n], radii_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(radii_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(radii_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_compare.pdf')
    plt.close()

# Stellar radii:
Rstar_axis = np.linspace(0.5, 2.5, pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(Rstar_axis), x_max=np.max(Rstar_axis), y_max=3, xlabel_text=r'$R_\star (R_\odot)$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = Rstar_kde_Kep[sample](Rstar_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(Rstar_kde_Kep[sample], Rstar_axis, kNN_factor=kNN_factor)
    plt.plot(Rstar_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = Rstar_kde_all[sample][n](Rstar_axis)
        kde_pts, bw_pts = kde_kNN_bw(Rstar_kde_all[sample][n], Rstar_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(Rstar_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(Rstar_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_stellar_radii_compare.pdf')
    plt.close()

# Transit depth ratios:
D_ratio_axis = np.logspace(-1.5, 1.5, pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(D_ratio_axis), x_max=np.max(D_ratio_axis), y_max=1.6, log_x=True, xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = D_ratio_kde_Kep[sample](np.log10(D_ratio_axis))
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(D_ratio_kde_Kep[sample], np.log10(D_ratio_axis), kNN_factor=kNN_factor)
    plt.plot(D_ratio_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = D_ratio_kde_all[sample][n](np.log10(D_ratio_axis))
        kde_pts, bw_pts = kde_kNN_bw(D_ratio_kde_all[sample][n], np.log10(D_ratio_axis), kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(D_ratio_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(D_ratio_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_depthratios_compare.pdf')
    plt.close()

# Log(xi):
xi_axis = np.linspace(-0.5, 0.5, pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(xi_axis), x_max=np.max(xi_axis), y_max=10, xlabel_text=r'$\log{\xi}$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = xi_kde_Kep[sample](xi_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(xi_kde_Kep[sample], xi_axis, kNN_factor=kNN_factor)
    plt.plot(xi_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = xi_kde_all[sample][n](xi_axis)
        kde_pts, bw_pts = kde_kNN_bw(xi_kde_all[sample][n], xi_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(xi_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(xi_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_all_compare.pdf')
    plt.close()

# Log(xi) (not near MMR):
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(xi_axis), x_max=np.max(xi_axis), y_max=10, extra_text='Not near MMR', xlabel_text=r'$\log{\xi}$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = xi_nonres_kde_Kep[sample](xi_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(xi_nonres_kde_Kep[sample], xi_axis, kNN_factor=kNN_factor)
    plt.plot(xi_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = xi_nonres_kde_all[sample][n](xi_axis)
        kde_pts, bw_pts = kde_kNN_bw(xi_nonres_kde_all[sample][n], xi_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(xi_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(xi_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_nonmmr_compare.pdf')
    plt.close()

# Log(xi) (near MMR):
plot_fig_pdf_simple(fig_size, [], [], x_min=np.min(xi_axis), x_max=np.max(xi_axis), y_max=10, extra_text='Near MMR', xlabel_text=r'$\log{\xi}$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = xi_res_kde_Kep[sample](xi_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(xi_res_kde_Kep[sample], xi_axis, kNN_factor=kNN_factor)
    plt.plot(xi_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = xi_res_kde_all[sample][n](xi_axis)
        kde_pts, bw_pts = kde_kNN_bw(xi_res_kde_all[sample][n], xi_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(xi_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(xi_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_logxi_mmr_compare.pdf')
    plt.close()

### GF2020 metrics:
# Planet radii partitioning:
radii_partitioning_axis = np.logspace(-5., 0., pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=1e-5, x_max=1., y_max=0.7, xlabel_text=r'$\mathcal{Q}_R$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = radii_partitioning_kde_Kep[sample](np.log10(radii_partitioning_axis))
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(radii_partitioning_kde_Kep[sample], np.log10(radii_partitioning_axis), kNN_factor=kNN_factor)
    plt.plot(radii_partitioning_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = radii_partitioning_kde_all[sample][n](np.log10(radii_partitioning_axis))
        kde_pts, bw_pts = kde_kNN_bw(radii_partitioning_kde_all[sample][n], np.log10(radii_partitioning_axis), kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(radii_partitioning_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(radii_partitioning_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_partitioning_compare.pdf')
    plt.close()

# Planet radii monotonicity:
radii_monotonicity_axis = np.linspace(-0.5, 0.6, pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=-0.5, x_max=0.6, y_max=4, xlabel_text=r'$\mathcal{M}_R$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = radii_monotonicity_kde_Kep[sample](radii_monotonicity_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(radii_monotonicity_kde_Kep[sample], radii_monotonicity_axis, kNN_factor=kNN_factor)
    plt.plot(radii_monotonicity_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = radii_monotonicity_kde_all[sample][n](radii_monotonicity_axis)
        kde_pts, bw_pts = kde_kNN_bw(radii_monotonicity_kde_all[sample][n], radii_monotonicity_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(radii_monotonicity_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(radii_monotonicity_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_radii_monotonicity_compare.pdf')
    plt.close()

# Gap complexity:
gap_complexity_axis = np.linspace(0., 1., pts)
plot_fig_pdf_simple(fig_size, [], [], x_min=0., x_max=1., y_max=10, xlabel_text=r'$\mathcal{C}$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = gap_complexity_kde_Kep[sample](gap_complexity_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(gap_complexity_kde_Kep[sample], gap_complexity_axis, kNN_factor=kNN_factor)
    plt.plot(gap_complexity_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = gap_complexity_kde_all[sample][n](gap_complexity_axis)
        kde_pts, bw_pts = kde_kNN_bw(gap_complexity_kde_all[sample][n], gap_complexity_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(gap_complexity_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(gap_complexity_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
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
plot_panel_cdf_simple(ax, [sss['tdur_tcirc_1_obs'] for sss in split_sss], [ssk['tdur_tcirc_1_obs'] for ssk in split_ssk], x_min=np.min(tdur_tcirc_axis), x_max=np.max(tdur_tcirc_axis), c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=['--','--'], lw=lw_Kep, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], extra_text='Observed singles', xlabel_text='', afs=afs, tfs=tfs, lfs=lfs, label_dist=True)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.75), ncol=1, frameon=False, fontsize=lfs)
plt.xticks([])
ax = plt.subplot(plot[1,0]) # PDF
plot_panel_pdf_simple(ax, [], [], x_min=np.min(tdur_tcirc_axis), x_max=np.max(tdur_tcirc_axis), y_max=6, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = tdur_tcirc_1_kde_Kep[sample](tdur_tcirc_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(tdur_tcirc_1_kde_Kep[sample], tdur_tcirc_axis, kNN_factor=kNN_factor)
    plt.plot(tdur_tcirc_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = tdur_tcirc_1_kde_all[sample][n](tdur_tcirc_axis)
        kde_pts, bw_pts = kde_kNN_bw(tdur_tcirc_1_kde_all[sample][n], tdur_tcirc_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(tdur_tcirc_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(tdur_tcirc_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_compare_with_CDFs.pdf')
    plt.close()



fig = plt.figure(figsize=(8,5))
plot = GridSpec(2,1,left=0.15,bottom=0.2,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0]) # CDF
plot_panel_cdf_simple(ax, [sss['tdur_tcirc_2p_obs'] for sss in split_sss], [ssk['tdur_tcirc_2p_obs'] for ssk in split_ssk], x_min=np.min(tdur_tcirc_axis), x_max=np.max(tdur_tcirc_axis), c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=['--','--'], lw=lw_Kep, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], extra_text='Observed multis', xlabel_text='', afs=afs, tfs=tfs, lfs=lfs, label_dist=True)
plt.xticks([])
ax = plt.subplot(plot[1,0]) # PDF
plot_panel_pdf_simple(ax, [], [], x_min=np.min(tdur_tcirc_axis), x_max=np.max(tdur_tcirc_axis), y_max=6, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Density', afs=afs, tfs=tfs, lfs=lfs)
for i,sample in enumerate(split_names):
    #kde_pts_Kep = tdur_tcirc_2p_kde_Kep[sample](tdur_tcirc_axis)
    kde_pts_Kep, bw_pts_Kep = kde_kNN_bw(tdur_tcirc_2p_kde_Kep[sample], tdur_tcirc_axis, kNN_factor=kNN_factor)
    plt.plot(tdur_tcirc_axis, kde_pts_Kep, ls='--', lw=lw_Kep, color=split_colors[i])
    
    kde_pts_all = np.zeros((runs, pts))
    for n in range(runs):
        #kde_pts = tdur_tcirc_2p_kde_all[sample][n](tdur_tcirc_axis)
        kde_pts, bw_pts = kde_kNN_bw(tdur_tcirc_2p_kde_all[sample][n], tdur_tcirc_axis, kNN_factor=kNN_factor)
        kde_pts_all[n,:] = kde_pts
        plt.plot(tdur_tcirc_axis, kde_pts, lw=lw_sim, alpha=alpha, color=split_colors[i])
    kde_pts_qtls = np.quantile(kde_pts_all, [0.16,0.84], axis=0)
    #plt.fill_between(tdur_tcirc_axis, kde_pts_qtls[0,:], kde_pts_qtls[1,:], color=split_colors[i], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_compare_with_CDFs.pdf')
    plt.close()
plt.show()
#'''

