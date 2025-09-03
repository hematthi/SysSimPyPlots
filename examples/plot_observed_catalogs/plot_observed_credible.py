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
from matplotlib.colors import LogNorm #for log color scales
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





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Observed/' + 'Fit_some8p1_KS_params10/'
#savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/H20_model/Observed/'
run_number = ''
model_name = 'Hybrid_model' + run_number #'Non_Clustered_Model', 'Clustered_P_Model', 'Clustered_P_R_Model', 'Maximum_AMD_model', 'Hybrid_NR20_AMD_model1'

compute_ratios = compute_ratios_adjacent
AD_mod = True
weights_all = load_split_stars_weights_only()
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 #'period_ratios_KS',
                 #'durations_KS',
                 #'durations_norm_circ_KS',
                 #'durations_norm_circ_singles_KS',
                 #'durations_norm_circ_multis_KS',
                 #'duration_ratios_KS',
                 #'duration_ratios_nonmmr_KS',
                 #'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radii_KS',
                 'radius_ratios_KS',
                 'radii_partitioning_KS',
                 'radii_monotonicity_KS',
                 #'gap_complexity_KS',
                 ]





##### To load the files with the systems with observed planets:

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
params = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated observed catalog of stars and planets:
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)





#'''
##### To plot the simulated and Kepler catalogs as marginal distributions:

fig_size = (8,3) # size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
n_bins_sys = 50 # fewer bins for system level metrics
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

#loadfiles_directory = '../../ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_dtotmax12_depthmin0.29_models/' #GP_best_models_100/'
runs = 1000

sss_all = []
sss_per_sys_all = []
params_all = []

Mtot_bins = np.arange(10)-0.5
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = []

xi_bins = np.linspace(-0.5, 0.5, n_bins+1)
xi_bins_mid = (xi_bins[:-1] + xi_bins[1:])/2.
xi_2_counts_all = []
xi_3_counts_all = []
xi_4p_counts_all = []

f_pos_monotonicity = [] # fractions of systems with observed positive monotonicity
f_low_gap_complexity = [] # fractions of systems with observed gap complexity < 0.1

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)
    
    sss_all.append(sss_i)
    sss_per_sys_all.append(sss_per_sys_i)
    params_all.append(params_i)

    # Multiplicities:
    counts, bins = np.histogram(sss_per_sys_i['Mtot_obs'], bins=Mtot_bins)
    Mtot_counts_all.append(counts/float(np.sum(counts)))

    # Log(xi) by multiplicity (2,3,4+):
    logxi_2 = np.log10(sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] == 2, 0])
    logxi_3 = np.log10(sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] == 3, :2].flatten())
    xi_4p = sss_per_sys_i['xi_obs'][sss_per_sys_i['Mtot_obs'] >= 4]
    logxi_4p = np.log10(xi_4p[xi_4p != -1])

    counts, bins = np.histogram(logxi_2, bins=xi_bins)
    xi_2_counts_all.append(counts/float(np.sum(counts)))
    counts, bins = np.histogram(logxi_3, bins=xi_bins)
    xi_3_counts_all.append(counts/float(np.sum(counts)))
    counts, bins = np.histogram(logxi_4p, bins=xi_bins)
    xi_4p_counts_all.append(counts/float(np.sum(counts)))
    
    # Extra stats:
    f_pos_M = np.sum(sss_per_sys_i['radii_monotonicity'] > 0) / float(len(sss_per_sys_i['radii_monotonicity']))
    f_low_C = np.sum(sss_per_sys_i['gap_complexity'] < 0.1) / float(len(sss_per_sys_i['gap_complexity']))
    f_pos_monotonicity.append(f_pos_M)
    f_low_gap_complexity.append(f_low_C)

    # To plot the fraction of planets in observed multis on a period-radius diagram:
    #P_bins = np.logspace(np.log10(P_min), np.log10(P_max), 6+1)
    #R_bins = np.logspace(np.log10(radii_min), np.log10(radii_max), 6+1)
    #plot_fig_period_radius_fraction_multis(sss_per_sys_i, sss_i, P_bins, R_bins, save_name=savefigures_directory + model_name + '_%s_PR_grid_fraction_obs_multis.png' % run_number, save_fig=False)
    #plt.show()
Mtot_counts_all = np.array(Mtot_counts_all)
xi_2_counts_all = np.array(xi_2_counts_all)
xi_3_counts_all = np.array(xi_3_counts_all)
xi_4p_counts_all = np.array(xi_4p_counts_all)

f_pos_monotonicity = np.array(f_pos_monotonicity)
f_low_gap_complexity = np.array(f_low_gap_complexity)



Mtot_counts_qtls = np.zeros((len(Mtot_bins_mid),3))
Mtot_cdfs_qtls = np.zeros((len(Mtot_bins_mid),3))
for b in range(len(Mtot_bins_mid)):
    Mtot_counts_qtls[b] = np.quantile(Mtot_counts_all[:,b], [0.16, 0.5, 0.84])
    Mtot_cdfs_qtls[b] = np.quantile(np.cumsum(Mtot_counts_all, axis=1)[:,b], [0.16, 0.5, 0.84])

xi_2_counts_qtls = np.zeros((n_bins,3))
xi_3_counts_qtls = np.zeros((n_bins,3))
xi_4p_counts_qtls = np.zeros((n_bins,3))
for b in range(n_bins):
    xi_2_counts_qtls[b] = np.quantile(xi_2_counts_all[:,b], [0.16, 0.5, 0.84])
    xi_3_counts_qtls[b] = np.quantile(xi_3_counts_all[:,b], [0.16, 0.5, 0.84])
    xi_4p_counts_qtls[b] = np.quantile(xi_4p_counts_all[:,b], [0.16, 0.5, 0.84])
#####





##### To plot the marginal distributions:

overplot_sss = False # TODO: whether to also plot a single catalog

# To make a 'plot' listing the model parameters:
'''
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,1,left=fig_lbrt[0],bottom=fig_lbrt[1],right=fig_lbrt[2],top=fig_lbrt[3],wspace=0.1,hspace=0.1)
nrows = 10
for p,param in enumerate(params):
    param_all = np.array([params_i[param] for params_i in params_all])
    if np.isclose(np.std(param_all), 0.): # for fixed params (same for all catalogs)
        param_str = np.round(params_all[0][param],3)
    else: # for active params (that are varied)
        param_qtls = np.quantile(param_all, [0.16,0.5,0.84])
        param_str = '${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(param_qtls[1], param_qtls[1]-param_qtls[0], param_qtls[2]-param_qtls[1])
    plt.figtext(x=0.05+0.3*int(p/float(nrows)), y=0.875-0.08*(p%nrows), s=r'%s = %s' % (param_symbols[param], param_str), fontsize=12)
if savefigures == True:
    plt.savefig(savefigures_directory + model_name + '_sim_params.pdf')
    plt.close()
'''
# Multiplicities:
plot_fig_counts_hist_simple([ssk_per_sys['Mtot_obs']], [], x_min=0, x_llim=0.5, normalize=True, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, lw=lw, xlabel_text='Observed multiplicity', ylabel_text='Fraction', labels_sim=['Kepler'], afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
plt.fill_between(Mtot_bins_mid, Mtot_counts_qtls[:,0], Mtot_counts_qtls[:,2], step='mid', color='k', alpha=alpha, label=r'Simulated 16-84%')
plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_multiplicities_compare.pdf')
    plt.close()

# Periods:
plot_fig_pdf_credible([sss_i['P_obs'] for sss_i in sss_all], [], [ssk['P_obs']], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, lw=lw, alpha=alpha, xticks_custom=[3,10,30,100,300], xlabel_text=r'Period, $P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=False, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_periods_compare.pdf')
    plt.close()

# Period ratios:
R_max_cut = 30.
plot_fig_pdf_credible([sss_i['Rm_obs'] for sss_i in sss_all], [], [ssk['Rm_obs']], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, alpha=alpha, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'Period ratio, $P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_periodratios_compare.pdf')
    plt.close()

# Transit durations:
plot_fig_pdf_credible([sss_i['tdur_obs'] for sss_i in sss_all], [], [ssk['tdur_obs']], x_min=0., x_max=15., n_bins=n_bins, lw=lw, alpha=alpha, xlabel_text=r'Transit duration, $t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_durations_compare.pdf')
    plt.close()

# Circular normalized transit durations (separate singles and multis):
plot_fig_pdf_credible([sss_i['tdur_tcirc_1_obs'] for sss_i in sss_all], [], [ssk['tdur_tcirc_1_obs']], x_min=0., x_max=1.5, n_bins=n_bins, lw=lw, alpha=alpha, extra_text='Observed singles', xlabel_text=r'Circular-normalized transit duration, $t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_tdur_tcirc_singles_compare.pdf')
    plt.close()

plot_fig_pdf_credible([sss_i['tdur_tcirc_2p_obs'] for sss_i in sss_all], [], [ssk['tdur_tcirc_2p_obs']], x_min=0., x_max=1.5, n_bins=n_bins, lw=lw, alpha=alpha, extra_text='Observed multis', xlabel_text=r'Circular-normalized transit duration, $t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_tdur_tcirc_multis_compare.pdf')
    plt.close()

# Transit depths:
plot_fig_pdf_credible([sss_i['D_obs'] for sss_i in sss_all], [], [ssk['D_obs']], x_min=1e-5, x_max=10**(-1.5), y_min=0., log_x=True, lw=lw, alpha=alpha, xlabel_text=r'Transit depth, $\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_depths_compare.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_credible([sss_i['radii_obs'] for sss_i in sss_all], [], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, y_max=0.06, log_x=False, lw=lw, alpha=alpha, xlabel_text=r'Planet radius, $R_p$ ($R_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_compare.pdf')
    plt.close()

# Transit depth ratios:
plot_fig_pdf_credible([sss_i['D_ratio_obs'] for sss_i in sss_all], [], [ssk['D_ratio_obs']], x_min=10**(-1.5), x_max=10**1.5, y_min=0, n_bins=n_bins, log_x=True, lw=lw, alpha=alpha, xlabel_text=r'Transit depth ratio, $\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_depthratios_compare.pdf')
    plt.close()

# Log(xi):
plot_fig_pdf_credible([np.log10(sss_i['xi_obs']) for sss_i in sss_all], [], [np.log10(ssk['xi_obs'])], x_min=-0.5, x_max=0.5, y_min=0., n_bins=n_bins, lw=lw, alpha=alpha, xlabel_text=r'Period-normalized transit duration ratio, $\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_logxi_compare.pdf')
    plt.close()

# Log(xi) (separate near vs. not near MMRs):
plot_fig_pdf_credible([np.log10(sss_i['xi_res_obs']) for sss_i in sss_all], [np.log10(sss_i['xi_nonres_obs']) for sss_i in sss_all], [np.log10(ssk['xi_res_obs']), np.log10(ssk['xi_nonres_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim1='m', c_sim2='g', c_Kep=['m','g'], ls_Kep=['-','-'], lw=lw, alpha=alpha, label_sim1='', label_sim2='', labels_Kep=['Near MMR', 'Not near MMR'], xlabel_text=r'Period-normalized transit duration ratio, $\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_logxi_mmrs_compare.pdf')
    plt.close()

# Radius partitioning:
plot_fig_pdf_credible([sss_per_sys_i['radii_partitioning'] for sss_per_sys_i in sss_per_sys_all], [], [ssk_per_sys['radii_partitioning']], x_min=1e-5, x_max=1., y_max=0.075, n_bins=n_bins_sys, log_x=True, lw=lw, alpha=alpha, xlabel_text=r'Radius partitioning, $\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_partitioning_compare.pdf')
    plt.close()

# Radius monotonicity:
plot_fig_pdf_credible([sss_per_sys_i['radii_monotonicity'] for sss_per_sys_i in sss_per_sys_all], [], [ssk_per_sys['radii_monotonicity']], x_min=-0.5, x_max=0.6, n_bins=n_bins_sys, log_x=False, lw=lw, alpha=alpha, xlabel_text=r'Radius monotonicity, $\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_monotonicity_compare.pdf')
    plt.close()

# Gap complexity:
plot_fig_pdf_credible([sss_per_sys_i['gap_complexity'] for sss_per_sys_i in sss_per_sys_all], [], [ssk_per_sys['gap_complexity']], x_min=0., x_max=1., n_bins=n_bins_sys, log_x=False, lw=lw, alpha=alpha, xlabel_text=r'Gap complexity, $\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_gap_complexity_compare.pdf')
    plt.close()

# Stellar radii:
plot_fig_pdf_credible([sss_i['Rstar_obs'] for sss_i in sss_all], [], [ssk['Rstar_obs']], x_min=0.5, x_max=2.5, y_min=0., n_bins=n_bins, lw=lw, alpha=alpha, xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_stellar_radii_compare.pdf')
    plt.close()

plt.show()
#plt.close()





##### To plot the marginal distributions (CDFs):

# Multiplicities:
plot_fig_mult_cdf_simple([ssk_per_sys['Mtot_obs']], [], y_min=0.6, lw=lw, labels_sim=['Kepler'], xlabel_text='Observed multiplicity', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
plt.fill_between(Mtot_bins_mid, Mtot_cdfs_qtls[:,0], Mtot_cdfs_qtls[:,2], step='post', color='k', alpha=alpha, label=r'Simulated 16-84%')
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_multiplicities_compare_CDFs.pdf')
    plt.close()

# Periods:
plot_fig_cdf_credible([sss_i['P_obs'] for sss_i in sss_all], [], [ssk['P_obs']], x_min=P_min, x_max=P_max, log_x=True, lw=lw, alpha=alpha, label_sim1='Simulated', xticks_custom=[3,10,30,100,300], xlabel_text=r'Period, $P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_periods_compare_CDFs.pdf')
    plt.close()

# Period ratios:
plot_fig_cdf_credible([sss_i['Rm_obs'] for sss_i in sss_all], [], [ssk['Rm_obs']], x_min=1., x_max=R_max_cut, log_x=True, lw=lw, alpha=alpha, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'Period ratio, $P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_periodratios_compare_CDFs.pdf')
    plt.close()

# Transit durations:
plot_fig_cdf_credible([sss_i['tdur_obs'] for sss_i in sss_all], [], [ssk['tdur_obs']], x_min=0., x_max=15., lw=lw, alpha=alpha, xlabel_text=r'Transit duration, $t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_durations_compare_CDFs.pdf')
    plt.close()

# Circular normalized transit durations (separate singles and multis):
plot_fig_cdf_credible([sss_i['tdur_tcirc_1_obs'] for sss_i in sss_all], [sss_i['tdur_tcirc_2p_obs'] for sss_i in sss_all], [ssk['tdur_tcirc_1_obs'], ssk['tdur_tcirc_2p_obs']], x_min=0., x_max=1.5, c_sim1='b', c_sim2='g', c_Kep=['b','g'], lw=lw, alpha=alpha, ls_Kep=['-','-'], labels_Kep=['Observed singles', 'Observed multis'], xlabel_text=r'Circular-normalized transit duration, $t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_tdur_tcirc_singles_multis_compare_CDFs.pdf')
    plt.close()

# Transit depths:
plot_fig_cdf_credible([sss_i['D_obs'] for sss_i in sss_all], [], [ssk['D_obs']], x_min=1e-5, x_max=10**(-1.5), log_x=True, lw=lw, alpha=alpha, xlabel_text=r'Transit depth, $\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_depths_compare_CDFs.pdf')
    plt.close()

# Planet radii:
plot_fig_cdf_credible([sss_i['radii_obs'] for sss_i in sss_all], [], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, lw=lw, alpha=alpha, xlabel_text=r'Planet radius, $R_p$ ($R_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_compare_CDFs.pdf')
    plt.close()

# Transit depth ratios:
plot_fig_cdf_credible([sss_i['D_ratio_obs'] for sss_i in sss_all], [], [ssk['D_ratio_obs']], x_min=10**(-1.5), x_max=10**1.5, log_x=True, lw=lw, alpha=alpha, xlabel_text=r'Transit depth ratio, $\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_depthratios_compare_CDFs.pdf')
    plt.close()

# Log(xi):
plot_fig_cdf_credible([np.log10(sss_i['xi_obs']) for sss_i in sss_all], [], [np.log10(ssk['xi_obs'])], x_min=-0.5, x_max=0.5, lw=lw, alpha=alpha, xlabel_text=r'Period-normalized transit duration ratio, $\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_logxi_compare_CDFs.pdf')
    plt.close()

# Log(xi) (separate near vs not-near MMRs):
plot_fig_cdf_credible([np.log10(sss_i['xi_res_obs']) for sss_i in sss_all], [np.log10(sss_i['xi_nonres_obs']) for sss_i in sss_all], [np.log10(ssk['xi_res_obs']), np.log10(ssk['xi_nonres_obs'])], x_min=-0.5, x_max=0.5, c_sim1='m', c_sim2='g', c_Kep=['m','g'], lw=lw, alpha=alpha, label_sim1='', label_sim2='', ls_Kep=['-','-'], labels_Kep=['Near MMR', 'Not near MMR'], xlabel_text=r'Period-normalized transit duration ratio, $\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_logxi_mmrs_compare_CDFs.pdf')
    plt.close()

# Radius partitioning:
plot_fig_cdf_credible([sss_per_sys_i['radii_partitioning'] for sss_per_sys_i in sss_per_sys_all], [], [ssk_per_sys['radii_partitioning']], x_min=1e-5, x_max=1., log_x=True, lw=lw, alpha=alpha, xlabel_text=r'Radius partitioning, $\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_partitioning_compare_CDFs.pdf')
    plt.close()

# Radius monotonicity:
plot_fig_cdf_credible([sss_per_sys_i['radii_monotonicity'] for sss_per_sys_i in sss_per_sys_all], [], [ssk_per_sys['radii_monotonicity']], x_min=-0.5, x_max=0.6, log_x=False, lw=lw, alpha=alpha, xlabel_text=r'Radius monotonicity, $\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_monotonicity_compare_CDFs.pdf')
    plt.close()

# Gap complexity:
plot_fig_cdf_credible([sss_per_sys_i['gap_complexity'] for sss_per_sys_i in sss_per_sys_all], [], [ssk_per_sys['gap_complexity']], x_min=0., x_max=1., log_x=False, lw=lw, alpha=alpha, xlabel_text=r'Gap complexity, $\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_gap_complexity_compare_CDFs.pdf')
    plt.close()

# Stellar radii:
plot_fig_cdf_credible([sss_i['Rstar_obs'] for sss_i in sss_all], [], [ssk['Rstar_obs']], x_min=0.5, x_max=2.5, lw=lw, xlabel_text=r'Stellar radius, $R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_stellar_radii_compare_CDFs.pdf')
    plt.close()

plt.show()
#plt.close()





##### To plot the xi distribution separated by observed multiplicities (m=2,3,4+):

logxi_Kep_2 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 2, 0])
logxi_Kep_3 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 3, :2].flatten())
xi_Kep_4p = ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] >= 4]
logxi_Kep_4p = np.log10(xi_Kep_4p[xi_Kep_4p != -1])

logxi_2 = np.log10(sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] == 2, 0])
logxi_3 = np.log10(sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] == 3, :2].flatten())
xi_4p = sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] >= 4]
logxi_4p = np.log10(xi_4p[xi_4p != -1])

c2, c3, c4p = 'r', 'b', 'g'

fig = plt.figure(figsize=(8,12))
plot = GridSpec(5,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0])
plot_panel_cdf_simple(ax, [logxi_2, logxi_3, logxi_4p], [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], x_min=np.min(xi_bins), x_max=np.max(xi_bins), c_sim=[c2,c3,c4p], c_Kep=['r','b','g'], ls_sim=['-','-','-'], ls_Kep=[':',':',':'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=['Kepler data',None,None], xlabel_text='', legend=True, afs=afs, tfs=tfs, lfs=lfs, label_dist=False)

ax = plt.subplot(plot[1:3,0])
#plot_panel_pdf_simple(ax, [logxi_2, logxi_3, logxi_4p], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=0.13, n_bins=n_bins, c_sim=[c2,c3,c4p], ls_sim=['-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None], xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plot_panel_pdf_simple(ax, [], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=0.13, n_bins=n_bins, xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.plot(xi_bins_mid, xi_2_counts_qtls[:,1], drawstyle='steps-mid', color=c2, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_3_counts_qtls[:,1], drawstyle='steps-mid', color=c3, lw=2, ls='-')
plt.plot(xi_bins_mid, xi_4p_counts_qtls[:,1], drawstyle='steps-mid', color=c4p, lw=2, ls='-')
plt.text(x=0.98, y=0.9, s=model_name, ha='right', fontsize=lfs, transform=ax.transAxes) #s='Maximum AMD model'; s='Two-Rayleigh model'
plt.fill_between(xi_bins_mid, xi_2_counts_qtls[:,0], xi_2_counts_qtls[:,2], step='mid', color=c2, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_3_counts_qtls[:,0], xi_3_counts_qtls[:,2], step='mid', color=c3, alpha=alpha)
plt.fill_between(xi_bins_mid, xi_4p_counts_qtls[:,0], xi_4p_counts_qtls[:,2], step='mid', color=c4p, alpha=alpha)

ax = plt.subplot(plot[3:,0])
plot_panel_pdf_simple(ax, [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=0.13, n_bins=n_bins, c_sim=[c2,c3,c4p], ls_sim=['-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None], xlabel_text=r'$\log{\xi}$', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s='Kepler data', ha='right', fontsize=lfs, transform=ax.transAxes)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_logxi_per_mult.pdf')
    plt.close()
plt.show()





##### To plot the proximity to MMR parameter (Pratio/Pratio_MMR - 1):
'''
fig = plt.figure(figsize=(8,10))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)

pr_mmrs = [(j+1)/j for j in range(1,5)]
pr_mmrs_labels = ['2:1', '3:2', '4:3', '5:4']

ax = plt.subplot(plot[:,:])
plot_panel_pdf_simple(ax, [ssk['Rm_obs']/pr - 1. for pr in pr_mmrs], [], x_min=-0.1, x_max=0.1, n_bins=20, c_sim=['r','b','g','m'], ls_sim=['-','-','-','-'], lw=2, labels_sim=pr_mmrs_labels, labels_Kep=[None], xlabel_text=r'$\mathcal{P}/\mathcal{P}_{\rm mmr} - 1$', legend=False, afs=afs, tfs=tfs, lfs=lfs)
'''





##### To plot period ratio distribution again but with MMRs marked:
'''
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple([sss['Rm_obs'][sss['Rm_obs'] < R_max_cut]], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20], labels_sim=['Simulated'], labels_Kep=['Kepler'], xlabel_text=r'Period ratio $P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_size, fig_lbrt=fig_lbrt)
#plt.plot(Rm_bins_mid, Rm_counts_qtls[:,0], drawstyle='steps-mid', color='r', lw=1, ls='--', label='16')
#plt.plot(Rm_bins_mid, Rm_counts_qtls[:,2], drawstyle='steps-mid', color='r', lw=1, ls='--', label='84')
for pr in [1.5, 2.]:
    plt.fill_betweenx([0,1], x1=pr, x2=(1.+res_width)*pr, alpha=0.2, color='r')
'''





##### To make a collection of some marginal distributions for a GIF:
'''
directory = '/Users/hematthi/Documents/GradSchool/Conferences/Exoplanet_Demographics_2020/Figures/GIF_images/Marginals_observed/'

n_bins = 100
lw = 3 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

for i in range(runs):
    fig = plt.figure(figsize=(8,15))
    plot = GridSpec(5,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0.4)

    ax = plt.subplot(plot[0,0]) # multiplicities
    plot_panel_counts_hist_simple(ax, [], [ssk_per_sys['Mtot_obs']], x_min=0, x_llim=0.5, normalize=True, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, lw=lw, xlabel_text='Observed planets per system', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(Mtot_bins_mid, Mtot_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-', label='Simulated')
    plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), ncol=1, frameon=False, fontsize=lfs) #show the legend

    ax = plt.subplot(plot[1,0]) # periods
    plot_panel_pdf_simple(ax, [], [ssk['P_obs']], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(P_bins_mid, P_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-')

    ax = plt.subplot(plot[2,0]) # period ratios
    R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
    plot_panel_pdf_simple(ax, [], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(Rm_bins_mid, Rm_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-')

    ax = plt.subplot(plot[3,0]) # depths
    plot_panel_pdf_simple(ax, [], [ssk['D_obs']], x_min=np.min(D_bins), x_max=np.max(D_bins), log_x=True, lw=lw, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(D_bins_mid, D_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-')

    ax = plt.subplot(plot[4,0]) # depth ratios
    plot_panel_pdf_simple(ax, [], [ssk['D_ratio_obs']], x_min=np.min(D_ratio_bins), x_max=np.max(D_ratio_bins), n_bins=n_bins, log_x=True, lw=lw, xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs)
    plt.plot(D_ratio_bins_mid, D_ratio_counts_all[i,:], drawstyle='steps-mid', color='k', lw=lw, ls='-')

    if savefigures:
        plt.savefig(directory + model_name + '_some_marginals_compare_%s.png' % i)
        plt.close()
    plt.show()
'''
