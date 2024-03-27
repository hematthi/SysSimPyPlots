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





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
#loadfiles_directory =  '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Underlying/'
#savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/Underlying/'
run_number = ''
model_name = 'Hybrid_NR20_AMD_model1' + run_number
model_label, model_color = 'Hybrid model', 'b' #'Maximum AMD model', 'g' #'Two-Rayleigh model', 'b'





# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated physical catalog of stars and planets:
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)





##### To plot the simulated catalog as marginal distributions:

fig_size = (8,3) # size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 2
alpha = 0.2

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_all_KS/Params12/GP_best_models_100/'
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'

runs = 100

sssp_all = []
sssp_per_sys_all = []
params_all = []

Mtot_bins = np.arange(23)-1.5 #include a -1 bin with all zeros for plotting purposes
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = []
Mtot_earth_counts_all = []
mean_pl_all = [] # mean number of planets per system with at least 1 planet
mean_pl_per_star_all = [] # mean number of planets per star

clustertot_bins = np.arange(12)-0.5 #includes 0 bin but will not count them
clustertot_bins_mid = (clustertot_bins[:-1] + clustertot_bins[1:])/2.
clustertot_counts_all = []

pl_per_cluster_bins = np.arange(12)-0.5 #includes 0 bin but will not count them
pl_per_cluster_bins_mid = (pl_per_cluster_bins[:-1] + pl_per_cluster_bins[1:])/2.
pl_per_cluster_counts_all = []

'''
e_bins = np.logspace(np.log10(1e-3), 0., n_bins+1)
e_bins_mid = (e_bins[:-1] + e_bins[1:])/2.
e_counts_all = []
e1_counts_all = [] # singles only
e2p_counts_all = [] # multis only
'''

# To also store median e and im per multiplicity for power-law fitting:
n_array = np.arange(2,11)
log_n = np.log10(n_array)
e_med_1_all = np.zeros(runs)
e_med_n_all = np.zeros((runs, len(n_array)))
im_med_n_all = np.zeros((runs, len(n_array)))

for i in range(runs):
    run_number = i+1
    print(i)
    N_sim_i = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)[0]
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

    # Catalogs and parameters:
    sssp_all.append(sssp_i)
    sssp_per_sys_all.append(sssp_per_sys_i)
    params_all.append(params_i)
    
    # Multiplicities:
    counts, bins = np.histogram(sssp_per_sys_i['Mtot_all'], bins=Mtot_bins)
    #counts[1] = N_sim_i - len(sssp_per_sys_i['Mtot_all'])
    Mtot_counts_all.append(counts/float(np.sum(counts)))

    mean_pl_all.append(np.mean(sssp_per_sys_i['Mtot_all'][sssp_per_sys_i['Mtot_all'] > 0]))
    mean_pl_per_star_all.append(np.mean(sssp_per_sys_i['Mtot_all']))
    print(len(sssp_per_sys_i['Mtot_all']), ' --- ', N_sim_i)

    # Multiplicities for Earth-sized planets:
    Earth_bools_per_sys = (sssp_per_sys_i['radii_all'] > 0.75) & (sssp_per_sys_i['radii_all'] < 1.25)
    Earth_counts_per_sys = np.sum(Earth_bools_per_sys, axis=1)
    counts, bins = np.histogram(Earth_counts_per_sys, bins=Mtot_bins)
    counts[1] = N_sim_i - len(Earth_counts_per_sys)
    Mtot_earth_counts_all.append(counts/float(np.sum(counts)))

    # Numbers of clusters:
    counts, bins = np.histogram(sssp_i['clustertot_all'], bins=clustertot_bins)
    clustertot_counts_all.append(counts/float(np.sum(counts)))

    # Numbers of planets per cluster:
    counts, bins = np.histogram(sssp_i['pl_per_cluster_all'], bins=pl_per_cluster_bins)
    pl_per_cluster_counts_all.append(counts/float(np.sum(counts)))

    # Eccentricities:
    '''
    # Singles only:
    e1 = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == 1, 0]
    counts1, bins = np.histogram(e1, bins=e_bins)
    e1_counts_all.append(counts1/float(np.sum(counts))) # normalize counts to all planets
    # Multis only:
    e2p = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] > 1]
    e2p = e2p[sssp_per_sys_i['P_all'][sssp_per_sys_i['Mtot_all'] > 1] > 0]
    counts2p, bins = np.histogram(e2p, bins=e_bins)
    e2p_counts_all.append(counts2p/float(np.sum(counts))) # normalize counts to all planets
    '''
    
    # Median eccentricity and mutual inclination per multiplicity:
    e_1 = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == 1,0]
    e_med_1_all[i] = np.median(e_1)
    for j,n in enumerate(n_array):
        e_n = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
        e_n = e_n.flatten()
        im_n = sssp_per_sys_i['inclmut_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
        im_n = im_n.flatten() * (180./np.pi)

        e_med_n_all[i,j] = np.median(e_n)
        im_med_n_all[i,j] = np.median(im_n)

Mtot_counts_all = np.array(Mtot_counts_all)
Mtot_earth_counts_all = np.array(Mtot_earth_counts_all)
mean_pl_all = np.array(mean_pl_all)
mean_pl_per_star_all = np.array(mean_pl_per_star_all)
clustertot_counts_all = np.array(clustertot_counts_all)
pl_per_cluster_counts_all = np.array(pl_per_cluster_counts_all)

Mtot_counts_qtls = np.zeros((len(Mtot_bins_mid),3))
clustertot_counts_qtls = np.zeros((len(clustertot_bins_mid),3))
pl_per_cluster_counts_qtls = np.zeros((len(pl_per_cluster_bins_mid),3))
for b in range(len(Mtot_bins_mid)):
    counts_bin_sorted = np.sort(Mtot_counts_all[:,b])
    Mtot_counts_qtls[b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
for b in range(len(clustertot_bins_mid)):
    counts_bin_sorted = np.sort(clustertot_counts_all[:,b])
    clustertot_counts_qtls[b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
for b in range(len(pl_per_cluster_bins_mid)):
    counts_bin_sorted = np.sort(pl_per_cluster_counts_all[:,b])
    pl_per_cluster_counts_qtls[b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
#####





##### To plot the marginal distributions:

overplot_sssp = False # whether to also plot a single catalog for comparison

#'''
# Multiplicities:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
plt.plot(Mtot_bins_mid, Mtot_counts_qtls[:,1], drawstyle='steps-mid', lw=lw, label=model_label, color=model_color)
plt.errorbar(Mtot_bins_mid, Mtot_counts_qtls[:,1], yerr=[Mtot_counts_qtls[:,1]-Mtot_counts_qtls[:,0], Mtot_counts_qtls[:,2]-Mtot_counts_qtls[:,1]], fmt='none', lw=lw, label='', color=model_color) #capsize=5 #label=r'16% and 84%' if m==0 else ''
if overplot_sssp:
    plt.hist(sssp_per_sys['Mtot_all'], bins=Mtot_bins, weights=np.ones(N_sim)/N_sim, histtype='step', lw=lw, label='Simulated catalog', color='k')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([-0.5, 10.5]) #[0, np.max(x)]
plt.ylim([0., 0.4])
plt.xlabel('Intrinsic planet multiplicity', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_multiplicities.pdf')
    plt.close()

# Number of clusters:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
plt.plot(clustertot_bins_mid, clustertot_counts_qtls[:,1], drawstyle='steps-mid', lw=lw, label=model_label, color=model_color)
plt.errorbar(clustertot_bins_mid, clustertot_counts_qtls[:,1], yerr=[clustertot_counts_qtls[:,1]-clustertot_counts_qtls[:,0], clustertot_counts_qtls[:,2]-clustertot_counts_qtls[:,1]], fmt='.', lw=lw, label='', color=model_color)
if overplot_sssp:
    plt.hist(sssp['clustertot_all'], bins=clustertot_bins, weights=np.ones(N_sim)/N_sim, histtype='step', lw=lw, label='Simulated catalog', color='k')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 5.5])
plt.ylim([0., 1.])
plt.xlabel(r'Clusters per system, $N_c$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_clusters.pdf')
    plt.close()

# Number of planets per cluster:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
plt.plot(pl_per_cluster_bins_mid, pl_per_cluster_counts_qtls[:,1], drawstyle='steps-mid', lw=lw, label=model_label, color=model_color)
plt.errorbar(pl_per_cluster_bins_mid, pl_per_cluster_counts_qtls[:,1], yerr=[pl_per_cluster_counts_qtls[:,1]-pl_per_cluster_counts_qtls[:,0], pl_per_cluster_counts_qtls[:,2]-pl_per_cluster_counts_qtls[:,1]], fmt='.', lw=lw, label='', color=model_color)
if overplot_sssp:
    N_clusters = len(sssp['pl_per_cluster_all'])
    plt.hist(sssp['pl_per_cluster_all'], bins=pl_per_cluster_bins, weights=np.ones(N_clusters)/N_clusters, histtype='step', lw=lw, label='Simulated catalog', color='k')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 7.5])
plt.ylim([0., 0.7])
plt.xlabel(r'Planets per cluster, $N_p$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_planets_per_cluster.pdf')
    plt.close()

# Periods:
plot_fig_pdf_credible([sssp_i['P_all'] for sssp_i in sssp_all], [], [sssp['P_all']] if overplot_sssp else [], x_min=P_min, x_max=P_max, y_min=1e-3, n_bins=n_bins, step=None, plot_median=True, log_x=True, log_y=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xticks_custom=[3,10,30,100,300], xlabel_text=r'Period, $P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_periods.pdf')
    plt.close()

# Period ratios:
plot_fig_pdf_credible([sssp_i['Rm_all'] for sssp_i in sssp_all], [], [sssp['Rm_all']] if overplot_sssp else [], x_min=1., x_max=20., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'Period ratio, $P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_periodratios.pdf')
    plt.close()

# Eccentricities:
plot_fig_pdf_credible([sssp_i['e_all'] for sssp_i in sssp_all], [], [sssp['e_all']] if overplot_sssp else [], x_min=1e-3, x_max=1., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xticks_custom=[0.01, 0.1, 1.], xlabel_text=r'Eccentricity, $e$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_eccentricities.pdf')
    plt.close()

# Mutual inclinations:
plot_fig_pdf_credible([sssp_i['inclmut_all']*(180./np.pi) for sssp_i in sssp_all], [], [sssp['inclmut_all']*(180./np.pi)] if overplot_sssp else [], x_min=1e-2, x_max=45., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Mutual inclination, $i_m$ (deg)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_mutualinclinations.pdf')
    plt.close()

# Planet masses:
plot_fig_pdf_credible([sssp_i['mass_all'] for sssp_i in sssp_all], [], [sssp['mass_all']] if overplot_sssp else [], x_min=1e-2, x_max=1e3, n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Planet mass, $M_p$ ($M_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_masses.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_credible([sssp_i['radii_all'] for sssp_i in sssp_all], [], [sssp['radii_all']] if overplot_sssp else [], x_min=radii_min, x_max=radii_max, n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, xticks_custom=[0.5,1,2,4,10], label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Planet radius, $R_p$ ($R_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_radii.pdf')
    plt.close()

# Planet radii ratios:
plot_fig_pdf_credible([sssp_i['radii_ratio_all'] for sssp_i in sssp_all], [], [sssp['radii_ratio_all']] if overplot_sssp else [], x_min=0.1, x_max=10., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Radius ratio, $R_{p,i+1}/R_{p,i}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_radii_ratios.pdf')
    plt.close()

# Separations in mutual Hill radii:
plot_fig_pdf_credible([sssp_i['N_mH_all'] for sssp_i in sssp_all], [], [sssp['N_mH_all']] if overplot_sssp else [], x_min=1., x_max=200., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Minimum separation, $\Delta$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_deltas.pdf')
    plt.close()

# Dynamical masses:
plot_fig_pdf_credible([sssp_per_sys_i['dynamical_mass'] for sssp_per_sys_i in sssp_per_sys_all], [], [sssp_per_sys['dynamical_mass']] if overplot_sssp else [], x_min=2e-7, x_max=3e-3, n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Dynamical mass, $\mu$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_dynamical_masses.pdf')
    plt.close()

# Planet radii partitioning:
plot_fig_pdf_credible([sssp_per_sys_i['radii_partitioning'] for sssp_per_sys_i in sssp_per_sys_all], [], [sssp_per_sys['radii_partitioning']] if overplot_sssp else [], x_min=1e-4, x_max=1., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Radius partitioning, $\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_radii_partitioning.pdf')
    plt.close()

# Planet radii monotonicity:
plot_fig_pdf_credible([sssp_per_sys_i['radii_monotonicity'] for sssp_per_sys_i in sssp_per_sys_all], [], [sssp_per_sys['radii_monotonicity']] if overplot_sssp else [], x_min=-0.7, x_max=0.7, n_bins=n_bins, step=None, plot_median=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Radius monotonicity, $\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_radii_monotonicity.pdf')
    plt.close()

# Gap complexity:
plot_fig_pdf_credible([sssp_per_sys_i['gap_complexity'] for sssp_per_sys_i in sssp_per_sys_all], [], [sssp_per_sys['gap_complexity']] if overplot_sssp else [], x_min=0., x_max=1., n_bins=n_bins, step=None, plot_median=True, c_sim1=model_color, lw=lw, alpha=alpha, label_sim1=model_label, labels_Kep=['Simulated catalog'], xlabel_text=r'Gap complexity, $\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_underlying_gap_complexity.pdf')
    plt.close()

plt.show()
plt.close()
#'''




##### To fit power-laws to the median eccentricities and mutual inclinations as a function of intrinsic multiplicity n:
##### NOTE: We are actually fitting a line to log(e) and log(im) vs log(n)
'''
n_norm = 5 # n for normalization constant
f_linear = lambda p, x: p[0] + p[1]*x - p[1]*np.log10(n_norm) # extra term for normalizing
f_err = lambda p, x, y: y - f_linear(p,x)
e_p0 = [0.03, -2.] # initial guesses for log10(mu_e) and alpha_e
im_p0 = [1., -2.] # initial guesses for log10(mu_im) and alpha_im

mu_e_all, alpha_e_all = np.zeros(runs), np.zeros(runs)
mu_im_all, alpha_im_all = np.zeros(runs), np.zeros(runs)
for i in range(runs):
    log_e_med = np.log10(e_med_n_all[i])
    log_im_med = np.log10(im_med_n_all[i])

    e_fit = scipy.optimize.leastsq(f_err, e_p0, args=(log_n, log_e_med), full_output=1)
    e_logmu_alpha = e_fit[0]
    mu_e, alpha_e = 10.**(e_logmu_alpha[0]), e_logmu_alpha[1]
    mu_e_all[i], alpha_e_all[i] = mu_e, alpha_e

    im_fit = scipy.optimize.leastsq(f_err, im_p0, args=(log_n, log_im_med), full_output=1)
    im_logmu_alpha = im_fit[0]
    mu_im, alpha_im = 10.**(im_logmu_alpha[0]), im_logmu_alpha[1]
    mu_im_all[i], alpha_im_all[i] = mu_im, alpha_im
'''





##### To plot median eccentricities and mutual inclinations vs. intrinsic multiplicity n, along with power-law fits, and for models with other values of f_amd_crit:
'''
def compute_power_law_at_n_quantiles(n_array, mu_all, alpha_all, n_norm=5, qtl=[0.16, 0.5, 0.84]):
    assert len(mu_all) == len(alpha_all)
    power_law_n_all = np.zeros((len(mu_all), len(n_array)))
    for i in range(len(mu_all)):
        power_law_n_all[i,:] = mu_all[i] * (n_array/n_norm)**alpha_all[i]
    return np.quantile(power_law_n_all, qtl, axis=0)

# To load other catalogs with different values of f_amd_crit:
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/f_amd_crit_all/Params12_KS/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
sssp_per_sys_0p5, sssp_0p5 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory + 'f_amd_crit_0p5/')
sssp_per_sys_2, sssp_2 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory + 'f_amd_crit_2/')

e_med_n_fcrits = np.zeros((2, len(n_array)))
im_med_n_fcrits = np.zeros((2, len(n_array)))
for i,sssp_per_sys_i in enumerate([sssp_per_sys_0p5, sssp_per_sys_2]):
    # Median eccentricity and mutual inclination per multiplicity:
    for j,n in enumerate(n_array):
        e_n = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
        e_n = e_n.flatten()
        im_n = sssp_per_sys_i['inclmut_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
        im_n = im_n.flatten() * (180./np.pi)

        e_med_n_fcrits[i,j] = np.median(e_n)
        im_med_n_fcrits[i,j] = np.median(im_n)



fig = plt.figure(figsize=(8,10))
plot = GridSpec(2,1,left=0.15,bottom=0.1,right=0.98,top=0.98,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0]) # ecc vs n
e_med_1_qtls = np.quantile(e_med_1_all, [0.16,0.5,0.84])
e_med_n_qtls = np.quantile(e_med_n_all, [0.16,0.5,0.84], axis=0)
e_plfit_n_qtls = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all, alpha_e_all, n_norm=n_norm)
plt.errorbar(n_array, e_med_n_qtls[1,:], yerr=[e_med_n_qtls[1,:]-e_med_n_qtls[0,:], e_med_n_qtls[2,:]-e_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='Maximum AMD model')
plt.errorbar([1], [e_med_1_qtls[1]], yerr=[[e_med_1_qtls[1]-e_med_1_qtls[0]], [e_med_1_qtls[2]-e_med_1_qtls[1]]], fmt='o', capsize=5, ls='none', color='c', label='')
plt.plot(n_array, e_med_n_fcrits[0,:], 'o', lw=lw, color='k', label=r'$f_{\rm crit} = 0.5$')
plt.plot(n_array, e_med_n_fcrits[1,:], 'o', lw=lw, color='r', label=r'$f_{\rm crit} = 2$')
#plt.plot(n_array, e_plfit_n_qtls[1,:], ls='-', lw=lw, color='b', label='')
plt.fill_between(np.arange(10)+1, e_plfit_n_qtls[0,:], e_plfit_n_qtls[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.031_{-0.003}^{+0.004}$, $\alpha_e = -1.74_{-0.07}^{+0.11}$')
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.yticks([0.01, 0.1, 1.])
plt.xlim([0.5, 10.5])
plt.ylim([0.005, 1.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.ylabel('Median $e$', fontsize=tfs)
handles, labels = ax.get_legend_handles_labels()
handles = [handles[-1]] + handles[:-1]
labels = [labels[-1]] + labels[:-1]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1,0]) # incl vs n
im_med_n_qtls = np.quantile(im_med_n_all, [0.16,0.5,0.84], axis=0)
im_plfit_n_qtls = compute_power_law_at_n_quantiles(n_array, mu_im_all, alpha_im_all, n_norm=n_norm)
plt.errorbar(n_array, im_med_n_qtls[1,:], yerr=[im_med_n_qtls[1,:]-im_med_n_qtls[0,:], im_med_n_qtls[2,:]-im_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='')
plt.plot(n_array, im_med_n_fcrits[0,:], 'o', lw=lw, color='k', label='')
plt.plot(n_array, im_med_n_fcrits[1,:], 'o', lw=lw, color='r', label='')
#plt.plot(n_array, im_plfit_n_qtls[1,:], ls='-', lw=lw, color='b', label='')
plt.fill_between(n_array, im_plfit_n_qtls[0,:], im_plfit_n_qtls[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.10_{-0.11}^{+0.15}$, $\alpha_i = -1.73_{-0.08}^{+0.09}$')
plt.plot(n_array, incl_mult_power_law_Zhu2018(n_array)*np.sqrt(2.*np.log(2.)), 'x', ls='-', lw=2, ms=5, color='orange', label=r'$\sigma_{i,5}[^\circ] = 0.8$, $\alpha_i = -3.5$')
plt.plot(n_array, incl_mult_power_law_Zhu2018(n_array, alpha=-2.8)*np.sqrt(2.*np.log(2.)), 'x', ls='-', lw=2, ms=5, color='m', label=r'$\sigma_{i,5}[^\circ] = 0.8$, $\alpha_i = -2.8$') # Yang, Xie, Zhou (2020) best fit across all spectral bins, with sigma_{i,5} assumed from Zhu et al (2018)
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.arange(10)+1)
plt.yticks([0.1, 1., 10.])
plt.xlim([0.5, 10.5])
plt.ylim([0.1, 30.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xlabel('Intrinsic planet multiplicity $n$', fontsize=tfs)
plt.ylabel('Median $i_m$ ($^\circ$)', fontsize=tfs)
handles, labels = ax.get_legend_handles_labels()
handles = [handles[-1]] + handles[:-1]
labels = [labels[-1]] + labels[:-1]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_ecc_incl_vs_mult_fits.pdf')
    plt.close()
plt.show()
'''





##### To remake/simplify the above figure for proposals:
'''
directory = '/Users/hematthi/Documents/GradSchool/Postdoctoral_Applications/2021/Figures/'

fig_size = (6,3) #size of each panel (figure)
fig_lbrt = [0.2, 0.3, 0.95, 0.925]

n_bins = 100
lw = 3 #linewidth
alpha = 0.2

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

# Eccentricity and mutual inclinations vs. intrinsic multiplicity:

fig = plt.figure(figsize=(12,6))
plot = GridSpec(1,2,left=0.1,bottom=0.15,right=0.98,top=0.95,wspace=0.3,hspace=0)

ax = plt.subplot(plot[0,0]) # ecc vs n
e_med_1_qtls = np.quantile(e_med_1_all, [0.16,0.5,0.84])
e_med_n_qtls = np.quantile(e_med_n_all, [0.16,0.5,0.84], axis=0)
e_plfit_n_qtls = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all, alpha_e_all, n_norm=n_norm)
plt.errorbar(n_array, e_med_n_qtls[1,:], yerr=[e_med_n_qtls[1,:]-e_med_n_qtls[0,:], e_med_n_qtls[2,:]-e_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='Maximum AMD model')
plt.errorbar([1], [e_med_1_qtls[1]], yerr=[[e_med_1_qtls[1]-e_med_1_qtls[0]], [e_med_1_qtls[2]-e_med_1_qtls[1]]], fmt='o', capsize=5, ls='none', color='c', label='')
#plt.plot(n_array, e_med_n_fcrits[0,:], 'o', lw=lw, color='k', label=r'$f_{\rm crit} = 0.5$')
#plt.plot(n_array, e_med_n_fcrits[1,:], 'o', lw=lw, color='r', label=r'$f_{\rm crit} = 2$')
#plt.plot(n_array, e_plfit_n_qtls[1,:], ls='-', lw=lw, color='b', label='')
plt.fill_between(np.arange(10)+1, e_plfit_n_qtls[0,:], e_plfit_n_qtls[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.031_{-0.003}^{+0.004}$,' + '\n' + r'$\alpha_e = -1.74_{-0.07}^{+0.10}$')
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.arange(10)+1)
plt.yticks([0.01, 0.1, 1.])
plt.xlim([0.5, 10.5])
plt.ylim([0.005, 1.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xlabel('Intrinsic planet multiplicity $n$', fontsize=tfs)
plt.ylabel('Median eccentricity $e$', fontsize=tfs)
handles, labels = ax.get_legend_handles_labels()
handles = [handles[-1]] + handles[:-1]
labels = [labels[-1]] + labels[:-1]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[0,1]) # incl vs n
im_med_n_qtls = np.quantile(im_med_n_all, [0.16,0.5,0.84], axis=0)
im_plfit_n_qtls = compute_power_law_at_n_quantiles(n_array, mu_im_all, alpha_im_all, n_norm=n_norm)
plt.errorbar(n_array, im_med_n_qtls[1,:], yerr=[im_med_n_qtls[1,:]-im_med_n_qtls[0,:], im_med_n_qtls[2,:]-im_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='')
#plt.plot(n_array, im_med_n_fcrits[0,:], 'o', lw=lw, color='k', label='')
#plt.plot(n_array, im_med_n_fcrits[1,:], 'o', lw=lw, color='r', label='')
#plt.plot(n_array, im_plfit_n_qtls[1,:], ls='-', lw=lw, color='b', label='')
plt.fill_between(n_array, im_plfit_n_qtls[0,:], im_plfit_n_qtls[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.10_{-0.11}^{+0.15}$,' + '\n' + r'$\alpha_i = -1.73_{-0.08}^{+0.09}$')
plt.plot(n_array, incl_mult_power_law_Zhu2018(n_array)*np.sqrt(2.*np.log(2.)), 'x', ls='-', lw=2, ms=5, color='orange', label=r'$\sigma_{i,5}[^\circ] = 0.8$, $\alpha_i = -3.5$')
#plt.plot(n_array, incl_mult_power_law_Zhu2018(n_array, alpha=-2.8)*np.sqrt(2.*np.log(2.)), 'x', ls='-', lw=2, ms=5, color='m', label=r'$\sigma_{i,5}[^\circ] = 0.8$, $\alpha_i = -2.8$') # Yang, Xie, Zhou (2020) best fit across all spectral bins, with sigma_{i,5} assumed from Zhu et al (2018)
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.arange(9)+2)
plt.yticks([0.1, 1., 10.])
plt.xlim([1.5, 10.5])
plt.ylim([0.1, 30.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xlabel('Intrinsic planet multiplicity $n$', fontsize=tfs)
plt.ylabel('Median mutual inclination $i_m$ ($^\circ$)', fontsize=tfs)
handles, labels = ax.get_legend_handles_labels()
handles = handles[1:] + [handles[0]]
labels = labels[1:] + [labels[0]]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(directory + 'Models_Compare_ecc_incl_vs_mult_fits.pdf')
    plt.close()
plt.show()
'''
