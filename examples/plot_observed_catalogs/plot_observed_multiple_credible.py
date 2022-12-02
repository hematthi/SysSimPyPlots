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
from scipy.stats import ks_2samp
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
savefigures_directory = '../Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Observed_compare/'
save_name = 'Models_Compare'





compute_ratios = compute_ratios_adjacent #compute_ratios_adjacent
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

# Model 1:
loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number1 = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory1 + 'periods%s.out' % run_number1)

param_vals_all1 = read_sim_params(loadfiles_directory1 + 'periods%s.out' % run_number1)
sss_per_sys1, sss1 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory1, run_number=run_number1, compute_ratios=compute_ratios)

# Model 2:
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
run_number2 = ''

param_vals_all2 = read_sim_params(loadfiles_directory2 + 'periods%s.out' % run_number2)
sss_per_sys2, sss2 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory2, run_number=run_number2, compute_ratios=compute_ratios)



model_sss = [sss1, sss2]
model_sss_per_sys = [sss_per_sys1, sss_per_sys2]
model_names = ['Maximum AMD model', 'Two-Rayleigh model']
model_linestyles = ['-', '--']
model_colors = ['g', 'b']
model_stagger_errorbars = [-0.05, 0.05] # offsets for plotting multiplicity counts in order to stagger errorbars



# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

logxi_Kep_2 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 2, 0])
logxi_Kep_3 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 3, :2].flatten())
logxi_Kep_4 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 4, :3].flatten())
xi_Kep_4p = ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] >= 4]
logxi_Kep_4p = np.log10(xi_Kep_4p[xi_Kep_4p != -1])
xi_Kep_5p = ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] >= 5]
logxi_Kep_5p = np.log10(xi_Kep_5p[xi_Kep_5p != -1])

dists1, dists_w1 = compute_distances_sim_Kepler(sss_per_sys1, sss1, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)

dists2, dists_w2 = compute_distances_sim_Kepler(sss_per_sys2, sss2, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)





#'''
##### To plot the simulated and Kepler catalogs as marginal distributions:

subdirectory = ''

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 3 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'

model_loadfiles_dirs = [loadfiles_directory1, loadfiles_directory2]
models = len(model_loadfiles_dirs)

runs = 100

Mtot_bins = np.arange(10)-0.5
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = []

P_bins = np.logspace(np.log10(P_min), np.log10(P_max), n_bins+1)
P_bins_mid = (P_bins[:-1] + P_bins[1:])/2.
P_counts_all = []

Rm_bins = np.logspace(np.log10(1.), np.log10(30.), n_bins+1)
Rm_bins_mid = (Rm_bins[:-1] + Rm_bins[1:])/2.
Rm_counts_all = []

tdur_bins = np.linspace(0., 15., n_bins+1)
tdur_bins_mid = (tdur_bins[:-1] + tdur_bins[1:])/2.
tdur_counts_all = []

tdur_tcirc_bins = np.linspace(0., 1.5, n_bins+1)
tdur_tcirc_bins_mid = (tdur_tcirc_bins[:-1] + tdur_tcirc_bins[1:])/2.
tdur_tcirc_1_counts_all = []
tdur_tcirc_2p_counts_all = []

D_bins = np.logspace(-5., -1.5, n_bins+1)
D_bins_mid = (D_bins[:-1] + D_bins[1:])/2.
D_counts_all = []

radii_bins = np.linspace(radii_min, radii_max, n_bins+1)
radii_bins_mid = (radii_bins[:-1] + radii_bins[1:])/2.
radii_counts_all = []

Rstar_bins = np.linspace(0.5, 2.5, n_bins+1)
Rstar_bins_mid = (Rstar_bins[:-1] + Rstar_bins[1:])/2.
Rstar_counts_all = []

D_ratio_bins = np.logspace(-1.5, 1.5, n_bins+1)
D_ratio_bins_mid = (D_ratio_bins[:-1] + D_ratio_bins[1:])/2.
D_ratio_counts_all = []

xi_bins = np.linspace(-0.5, 0.5, n_bins+1)
xi_bins_mid = (xi_bins[:-1] + xi_bins[1:])/2.
xi_counts_all = []
xi_res_counts_all = []
xi_nonres_counts_all = []
xi_2_counts_all = []
xi_3_counts_all = []
xi_4_counts_all = []
xi_4p_counts_all = []
xi_5p_counts_all = []

xi_2_pvals_all = []
xi_3_pvals_all = []
xi_4_pvals_all = []
xi_4p_pvals_all = []
xi_5p_pvals_all = []

for loadfiles_dir in model_loadfiles_dirs:
    Mtot_counts = []
    P_counts = []
    Rm_counts = []
    tdur_counts = []
    tdur_tcirc_1_counts = []
    tdur_tcirc_2p_counts = []
    D_counts = []
    radii_counts = []
    Rstar_counts = []
    D_ratio_counts = []
    xi_counts = []
    xi_res_counts = []
    xi_nonres_counts = []
    xi_2_counts = []
    xi_3_counts = []
    xi_4_counts = []
    xi_4p_counts = []
    xi_5p_counts = []

    xi_2_pvals = []
    xi_3_pvals = []
    xi_4_pvals = []
    xi_4p_pvals = []
    xi_5p_pvals = []

    for i in range(1,runs+1): #range(1,runs+1)
        run_number = i
        sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_dir, run_number=run_number, compute_ratios=compute_ratios)
        dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod)

        # Multiplicities:
        counts, bins = np.histogram(sss_per_sys_i['Mtot_obs'], bins=Mtot_bins)
        Mtot_counts.append(counts/float(np.sum(counts)))

        # Periods:
        counts, bins = np.histogram(sss_i['P_obs'], bins=P_bins)
        P_counts.append(counts/float(np.sum(counts)))

        # Period ratios:
        counts, bins = np.histogram(sss_i['Rm_obs'], bins=Rm_bins)
        Rm_counts.append(counts/float(np.sum(counts)))

        # Durations:
        counts, bins = np.histogram(sss_i['tdur_obs'], bins=tdur_bins)
        tdur_counts.append(counts/float(np.sum(counts)))

        # Circular normalized durations (singles and multis):
        counts, bins = np.histogram(sss_i['tdur_tcirc_1_obs'], bins=tdur_tcirc_bins)
        tdur_tcirc_1_counts.append(counts/float(np.sum(counts)))
        counts, bins = np.histogram(sss_i['tdur_tcirc_2p_obs'], bins=tdur_tcirc_bins)
        tdur_tcirc_2p_counts.append(counts/float(np.sum(counts)))

        # Depths:
        counts, bins = np.histogram(sss_i['D_obs'], bins=D_bins)
        D_counts.append(counts/float(np.sum(counts)))

        # Planet radii:
        counts, bins = np.histogram(sss_i['radii_obs'], bins=radii_bins)
        radii_counts.append(counts/float(np.sum(counts)))

        # Stellar radii:
        counts, bins = np.histogram(sss_i['Rstar_obs'], bins=Rstar_bins)
        Rstar_counts.append(counts/float(np.sum(counts)))

        # Depth ratios:
        counts, bins = np.histogram(sss_i['D_ratio_obs'], bins=D_ratio_bins)
        D_ratio_counts.append(counts/float(np.sum(counts)))

        # Log(xi):
        counts, bins = np.histogram(np.log10(sss_i['xi_obs']), bins=xi_bins)
        xi_counts.append(counts/float(np.sum(counts)))

        # Log(xi) (res):
        counts, bins = np.histogram(np.log10(sss_i['xi_res_obs']), bins=xi_bins)
        xi_res_counts.append(counts/float(np.sum(counts)))

        # Log(xi) (non-res):
        counts, bins = np.histogram(np.log10(sss_i['xi_nonres_obs']), bins=xi_bins)
        xi_nonres_counts.append(counts/float(np.sum(counts)))

        # Log(xi) by multiplicity (2,3,4+):
        logxi_2 = np.log10(sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] == 2, 0])
        logxi_3 = np.log10(sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] == 3, :2].flatten())
        logxi_4 = np.log10(sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] == 4, :3].flatten())
        xi_4p = sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] >= 4]
        logxi_4p = np.log10(xi_4p[xi_4p != -1])
        xi_5p = sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] >= 5]
        logxi_5p = np.log10(xi_5p[xi_5p != -1])

        counts, bins = np.histogram(logxi_2, bins=xi_bins)
        xi_2_counts.append(counts/float(np.sum(counts)))
        counts, bins = np.histogram(logxi_3, bins=xi_bins)
        xi_3_counts.append(counts/float(np.sum(counts)))
        counts, bins = np.histogram(logxi_4, bins=xi_bins)
        xi_4_counts.append(counts/float(np.sum(counts)))
        counts, bins = np.histogram(logxi_4p, bins=xi_bins)
        xi_4p_counts.append(counts/float(np.sum(counts)))
        counts, bins = np.histogram(logxi_5p, bins=xi_bins)
        xi_5p_counts.append(counts/float(np.sum(counts)))

        xi_2_pvals.append(ks_2samp(logxi_2, logxi_Kep_2).pvalue)
        xi_3_pvals.append(ks_2samp(logxi_3, logxi_Kep_3).pvalue)
        xi_4_pvals.append(ks_2samp(logxi_4, logxi_Kep_4).pvalue)
        xi_4p_pvals.append(ks_2samp(logxi_4p, logxi_Kep_4p).pvalue)
        xi_5p_pvals.append(ks_2samp(logxi_5p, logxi_Kep_5p).pvalue)

    Mtot_counts_all.append(np.array(Mtot_counts))
    P_counts_all.append(np.array(P_counts))
    Rm_counts_all.append(np.array(Rm_counts))
    tdur_counts_all.append(np.array(tdur_counts))
    tdur_tcirc_1_counts_all.append(np.array(tdur_tcirc_1_counts))
    tdur_tcirc_2p_counts_all.append(np.array(tdur_tcirc_2p_counts))
    D_counts_all.append(np.array(D_counts))
    radii_counts_all.append(np.array(radii_counts))
    Rstar_counts_all.append(np.array(Rstar_counts))
    D_ratio_counts_all.append(np.array(D_ratio_counts))
    xi_counts_all.append(np.array(xi_counts))
    xi_res_counts_all.append(np.array(xi_res_counts))
    xi_nonres_counts_all.append(np.array(xi_nonres_counts))
    xi_2_counts_all.append(np.array(xi_2_counts))
    xi_3_counts_all.append(np.array(xi_3_counts))
    xi_4_counts_all.append(np.array(xi_4_counts))
    xi_4p_counts_all.append(np.array(xi_4p_counts))
    xi_5p_counts_all.append(np.array(xi_5p_counts))

    xi_2_pvals_all.append(np.array(xi_2_pvals))
    xi_3_pvals_all.append(np.array(xi_3_pvals))
    xi_4_pvals_all.append(np.array(xi_4_pvals))
    xi_4p_pvals_all.append(np.array(xi_4p_pvals))
    xi_5p_pvals_all.append(np.array(xi_5p_pvals))

Mtot_counts_all = np.array(Mtot_counts_all)
P_counts_all = np.array(P_counts_all)
Rm_counts_all = np.array(Rm_counts_all)
tdur_counts_all = np.array(tdur_counts_all)
tdur_tcirc_1_counts_all = np.array(tdur_tcirc_1_counts_all)
tdur_tcirc_2p_counts_all = np.array(tdur_tcirc_2p_counts_all)
D_counts_all = np.array(D_counts_all)
radii_counts_all = np.array(radii_counts_all)
Rstar_counts_all = np.array(Rstar_counts_all)
D_ratio_counts_all = np.array(D_ratio_counts_all)
xi_counts_all = np.array(xi_counts_all)
xi_res_counts_all = np.array(xi_res_counts_all)
xi_nonres_counts_all = np.array(xi_nonres_counts_all)
xi_2_counts_all = np.array(xi_2_counts_all)
xi_3_counts_all = np.array(xi_3_counts_all)
xi_4_counts_all = np.array(xi_4_counts_all)
xi_4p_counts_all = np.array(xi_4p_counts_all)
xi_5p_counts_all = np.array(xi_5p_counts_all)

xi_2_pvals_all = np.array(xi_2_pvals_all)
xi_3_pvals_all = np.array(xi_3_pvals_all)
xi_4_pvals_all = np.array(xi_4_pvals_all)
xi_4p_pvals_all = np.array(xi_4p_pvals_all)
xi_5p_pvals_all = np.array(xi_5p_pvals_all)



Mtot_counts_qtls = [np.zeros((len(Mtot_bins_mid),3)) for m in range(models)]

P_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
Rm_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
tdur_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
tdur_tcirc_1_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
tdur_tcirc_2p_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
D_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
radii_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
Rstar_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
D_ratio_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
xi_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
xi_res_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
xi_nonres_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
xi_2_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
xi_3_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
xi_4_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
xi_4p_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
xi_5p_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]

for m in range(models):
    for b in range(len(Mtot_bins_mid)):
        Mtot_counts_qtls[m][b] = np.quantile(Mtot_counts_all[m][:,b], [0.16, 0.5, 0.84])

    for b in range(n_bins):
        # Periods:
        P_counts_qtls[m][b] = np.quantile(P_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Period ratios:
        Rm_counts_qtls[m][b] = np.quantile(Rm_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Durations:
        tdur_counts_qtls[m][b] = np.quantile(tdur_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Circular normalized durations (singles and multis):
        tdur_tcirc_1_counts_qtls[m][b] = np.quantile(tdur_tcirc_1_counts_all[m][:,b], [0.16, 0.5, 0.84])
        tdur_tcirc_2p_counts_qtls[m][b] = np.quantile(tdur_tcirc_2p_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Depths:
        D_counts_qtls[m][b] = np.quantile(D_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Planet radii:
        radii_counts_qtls[m][b] = np.quantile(radii_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Stellar radii:
        Rstar_counts_qtls[m][b] = np.quantile(Rstar_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Depth ratios:
        D_ratio_counts_qtls[m][b] = np.quantile(D_ratio_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Log(xi):
        xi_counts_qtls[m][b] = np.quantile(xi_counts_all[m][:,b], [0.16, 0.5, 0.84])
        xi_res_counts_qtls[m][b] = np.quantile(xi_res_counts_all[m][:,b], [0.16, 0.5, 0.84])
        xi_nonres_counts_qtls[m][b] = np.quantile(xi_nonres_counts_all[m][:,b], [0.16, 0.5, 0.84])
        xi_2_counts_qtls[m][b] = np.quantile(xi_2_counts_all[m][:,b], [0.16, 0.5, 0.84])
        xi_3_counts_qtls[m][b] = np.quantile(xi_3_counts_all[m][:,b], [0.16, 0.5, 0.84])
        xi_4_counts_qtls[m][b] = np.quantile(xi_4_counts_all[m][:,b], [0.16, 0.5, 0.84])
        xi_4p_counts_qtls[m][b] = np.quantile(xi_4p_counts_all[m][:,b], [0.16, 0.5, 0.84])
        xi_5p_counts_qtls[m][b] = np.quantile(xi_5p_counts_all[m][:,b], [0.16, 0.5, 0.84])

#####





# Multiplicities:
plot_fig_counts_hist_simple([], [ssk_per_sys['Mtot_obs']], x_min=0, x_llim=0.5, normalize=True, log_y=True, lw=lw, xlabel_text='Observed planets per system', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.plot(Mtot_bins_mid, Mtot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(Mtot_bins_mid + model_stagger_errorbars[m], Mtot_counts_qtls[m][:,1], yerr=[Mtot_counts_qtls[m][:,1]-Mtot_counts_qtls[m][:,0], Mtot_counts_qtls[m][:,2]-Mtot_counts_qtls[m][:,1]], fmt='none', color=model_colors[m], lw=lw, label='') #capsize=5 #label=r'16% and 84%' if m==0 else ''
plt.legend(loc='lower left', bbox_to_anchor=(0,0), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_multiplicities_compare.pdf')
    plt.close()

# Periods:
plot_fig_pdf_simple([sss['P_obs'] for sss in model_sss], [ssk['P_obs']], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    label_this = r'16%-84%' if m==0 else ''
    plt.fill_between(P_bins_mid, P_counts_qtls[m][:,0], P_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha, label=label_this)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_periods_compare.pdf')
    plt.close()

# Period ratios (all, with some upper cut-off):
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple([sss['Rm_obs'][sss['Rm_obs'] < R_max_cut] for sss in model_sss], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(Rm_bins_mid, Rm_counts_qtls[m][:,0], Rm_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
plt.minorticks_off()
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_periodratios_compare.pdf')
    plt.close()

# Transit durations:
plot_fig_pdf_simple([sss['tdur_obs'] for sss in model_sss], [ssk['tdur_obs']], x_min=0., x_max=15., n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(tdur_bins_mid, tdur_counts_qtls[m][:,0], tdur_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_durations_compare.pdf')
    plt.close()

# Circular normalized transit durations (separate singles and multis):
plot_fig_pdf_simple([sss['tdur_tcirc_1_obs'] for sss in model_sss], [ssk['tdur_tcirc_1_obs']], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, extra_text='Observed singles', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(tdur_tcirc_bins_mid, tdur_tcirc_1_counts_qtls[m][:,0], tdur_tcirc_1_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_tdur_tcirc_singles_compare.pdf')
    plt.close()

plot_fig_pdf_simple([sss['tdur_tcirc_2p_obs'] for sss in model_sss], [ssk['tdur_tcirc_2p_obs']], x_min=np.min(tdur_tcirc_bins), x_max=np.max(tdur_tcirc_bins), n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, extra_text='Observed multis', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(tdur_tcirc_bins_mid, tdur_tcirc_2p_counts_qtls[m][:,0], tdur_tcirc_2p_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_tdur_tcirc_multis_compare.pdf')
    plt.close()

# Transit depths:
plot_fig_pdf_simple([sss['D_obs'] for sss in model_sss], [ssk['D_obs']], x_min=np.min(D_bins), x_max=np.max(D_bins), log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(D_bins_mid, D_counts_qtls[m][:,0], D_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_depths_compare.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_simple([sss['radii_obs'] for sss in model_sss], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$R_p (R_\oplus)$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_bins_mid, radii_counts_qtls[m][:,0], radii_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_radii_compare.pdf')
    plt.close()

# Stellar radii:
plot_fig_pdf_simple([sss['Rstar_obs'] for sss in model_sss], [ssk['Rstar_obs']], x_min=0.5, x_max=2.5, n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(Rstar_bins_mid, Rstar_counts_qtls[m][:,0], Rstar_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_stellar_radii_compare.pdf')
    plt.close()

# Transit depth ratios:
plot_fig_pdf_simple([sss['D_ratio_obs'] for sss in model_sss], [ssk['D_ratio_obs']], x_min=np.min(D_ratio_bins), x_max=np.max(D_ratio_bins), n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(D_ratio_bins_mid, D_ratio_counts_qtls[m][:,0], D_ratio_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_depthratios_compare.pdf')
    plt.close()

# Log(xi):
plot_fig_pdf_simple([np.log10(sss['xi_obs']) for sss in model_sss], [np.log10(ssk['xi_obs'])], x_min=np.min(xi_bins), x_max=np.max(xi_bins), n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(xi_bins_mid, xi_counts_qtls[m][:,0], xi_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_logxi_all_compare.pdf')
    plt.close()

plt.show()
#plt.close()





##### To plot the xi distribution separated by observed multiplicities (m=2,3,4+):

logxi_2_model1 = np.log10(sss_per_sys1['xi_obs'][sss_per_sys1['Mtot_obs'] == 2, 0])
logxi_3_model1 = np.log10(sss_per_sys1['xi_obs'][sss_per_sys1['Mtot_obs'] == 3, :2].flatten())
logxi_4_model1 = np.log10(sss_per_sys1['xi_obs'][sss_per_sys1['Mtot_obs'] == 4, :3].flatten())
xi_4p_model1 = sss_per_sys1['xi_obs'][sss_per_sys1['Mtot_obs'] >= 4]
logxi_4p_model1 = np.log10(xi_4p_model1[xi_4p_model1 != -1])
xi_5p_model1 = sss_per_sys1['xi_obs'][sss_per_sys1['Mtot_obs'] >= 5]
logxi_5p_model1 = np.log10(xi_5p_model1[xi_5p_model1 != -1])

logxi_2_model2 = np.log10(sss_per_sys2['xi_obs'][sss_per_sys2['Mtot_obs'] == 2, 0])
logxi_3_model2 = np.log10(sss_per_sys2['xi_obs'][sss_per_sys2['Mtot_obs'] == 3, :2].flatten())
logxi_4_model2 = np.log10(sss_per_sys2['xi_obs'][sss_per_sys2['Mtot_obs'] == 4, :3].flatten())
xi_4p_model2 = sss_per_sys2['xi_obs'][sss_per_sys2['Mtot_obs'] >= 4]
logxi_4p_model2 = np.log10(xi_4p_model2[xi_4p_model2 != -1])
xi_5p_model2 = sss_per_sys2['xi_obs'][sss_per_sys2['Mtot_obs'] >= 5]
logxi_5p_model2 = np.log10(xi_5p_model2[xi_5p_model2 != -1])

c2, c3, c4p, c5p = 'r', 'b', 'g', 'm'

# m = 2, 3, 4+
ymax = 0.14

fig = plt.figure(figsize=(8,14))
plot = GridSpec(7,1,left=0.2,bottom=0.07,right=0.95,top=0.98,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0])
plot_panel_cdf_simple(ax, [logxi_2_model1, logxi_3_model1, logxi_4p_model1], [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], x_min=np.min(xi_bins), x_max=np.max(xi_bins), c_sim=[c2,c3,c4p], c_Kep=[c2,c3,c4p], ls_sim=['-','-','-'], ls_Kep=[':',':',':'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=['Kepler data',None,None], xlabel_text='', legend=True, afs=afs, tfs=tfs, lfs=lfs, label_dist=False)

ax = plt.subplot(plot[1:3,0])
plot_panel_pdf_simple(ax, [], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.plot(xi_bins_mid, xi_2_counts_qtls[0][:,1], drawstyle='steps-mid', color=c2, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_3_counts_qtls[0][:,1], drawstyle='steps-mid', color=c3, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_4p_counts_qtls[0][:,1], drawstyle='steps-mid', color=c4p, lw=2, ls='-')
plt.text(x=0.98, y=0.9, s=model_names[0], ha='right', fontsize=lfs, transform=ax.transAxes)
plt.fill_between(xi_bins_mid, xi_2_counts_qtls[0][:,0], xi_2_counts_qtls[0][:,2], step='mid', color=c2, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_3_counts_qtls[0][:,0], xi_3_counts_qtls[0][:,2], step='mid', color=c3, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_4p_counts_qtls[0][:,0], xi_4p_counts_qtls[0][:,2], step='mid', color=c4p, alpha=alpha)

ax = plt.subplot(plot[3:5,0])
plot_panel_pdf_simple(ax, [], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.plot(xi_bins_mid, xi_2_counts_qtls[1][:,1], drawstyle='steps-mid', color=c2, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_3_counts_qtls[1][:,1], drawstyle='steps-mid', color=c3, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_4p_counts_qtls[1][:,1], drawstyle='steps-mid', color=c4p, lw=2, ls='-')
plt.text(x=0.98, y=0.9, s=model_names[1], ha='right', fontsize=lfs, transform=ax.transAxes)
plt.fill_between(xi_bins_mid, xi_2_counts_qtls[1][:,0], xi_2_counts_qtls[1][:,2], step='mid', color=c2, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_3_counts_qtls[1][:,0], xi_3_counts_qtls[1][:,2], step='mid', color=c3, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_4p_counts_qtls[1][:,0], xi_4p_counts_qtls[1][:,2], step='mid', color=c4p, alpha=alpha)

ax = plt.subplot(plot[5:,0])
plot_panel_pdf_simple(ax, [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, c_sim=[c2,c3,c4p], ls_sim=['-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None], xlabel_text=r'$\log{\xi}$', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s='Kepler data', ha='right', fontsize=lfs, transform=ax.transAxes)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_logxi_per_mult.pdf')
    plt.close()



# m = 2, 3, 4, 5+
ymax = 0.24

fig = plt.figure(figsize=(8,14))
plot = GridSpec(7,1,left=0.2,bottom=0.07,right=0.95,top=0.98,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0])
plot_panel_cdf_simple(ax, [logxi_2_model1, logxi_3_model1, logxi_4_model1, logxi_5p_model1], [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4, logxi_Kep_5p], x_min=np.min(xi_bins), x_max=np.max(xi_bins), c_sim=[c2,c3,c4p,c5p], c_Kep=[c2,c3,c4p,c5p], ls_sim=['-','-','-','-'], ls_Kep=[':',':',':',':'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4$', r'$m=5+$'], labels_Kep=['Kepler data',None,None,None], xlabel_text='', legend=True, afs=afs, tfs=tfs, lfs=lfs, label_dist=False)

ax = plt.subplot(plot[1:3,0])
plot_panel_pdf_simple(ax, [], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.plot(xi_bins_mid, xi_2_counts_qtls[0][:,1], drawstyle='steps-mid', color=c2, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_3_counts_qtls[0][:,1], drawstyle='steps-mid', color=c3, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_4_counts_qtls[0][:,1], drawstyle='steps-mid', color=c4p, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_5p_counts_qtls[0][:,1], drawstyle='steps-mid', color=c5p, lw=2, ls='-')
plt.text(x=0.98, y=0.9, s=model_names[0], ha='right', fontsize=lfs, transform=ax.transAxes)
plt.fill_between(xi_bins_mid, xi_2_counts_qtls[0][:,0], xi_2_counts_qtls[0][:,2], step='mid', color=c2, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_3_counts_qtls[0][:,0], xi_3_counts_qtls[0][:,2], step='mid', color=c3, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_4_counts_qtls[0][:,0], xi_4_counts_qtls[0][:,2], step='mid', color=c4p, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_5p_counts_qtls[0][:,0], xi_5p_counts_qtls[0][:,2], step='mid', color=c5p, alpha=alpha)

ax = plt.subplot(plot[3:5,0])
plot_panel_pdf_simple(ax, [], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.plot(xi_bins_mid, xi_2_counts_qtls[1][:,1], drawstyle='steps-mid', color=c2, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_3_counts_qtls[1][:,1], drawstyle='steps-mid', color=c3, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_4_counts_qtls[1][:,1], drawstyle='steps-mid', color=c4p, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_5p_counts_qtls[1][:,1], drawstyle='steps-mid', color=c5p, lw=2, ls='-')
plt.text(x=0.98, y=0.9, s=model_names[1], ha='right', fontsize=lfs, transform=ax.transAxes)
plt.fill_between(xi_bins_mid, xi_2_counts_qtls[1][:,0], xi_2_counts_qtls[1][:,2], step='mid', color=c2, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_3_counts_qtls[1][:,0], xi_3_counts_qtls[1][:,2], step='mid', color=c3, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_4_counts_qtls[1][:,0], xi_4_counts_qtls[1][:,2], step='mid', color=c4p, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_5p_counts_qtls[1][:,0], xi_5p_counts_qtls[1][:,2], step='mid', color=c5p, alpha=alpha)

ax = plt.subplot(plot[5:,0])
plot_panel_pdf_simple(ax, [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4, logxi_Kep_5p], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, c_sim=[c2,c3,c4p,c5p], ls_sim=['-','-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4$', r'$m=5+$'], labels_Kep=[None], xlabel_text=r'$\log{\xi}$', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s='Kepler data', ha='right', fontsize=lfs, transform=ax.transAxes)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_logxi_per_mult_2345.pdf')
    plt.close()
