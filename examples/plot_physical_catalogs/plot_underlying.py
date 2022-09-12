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
import scipy.optimize #for fitting functions
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #Extrapolate_P1000d/
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/' #Extrapolate_P1000d/lambdac5/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Underlying/' #'/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Underlying/' #'/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/Best_models/GP_med/Underlying/'
run_number = ''
model_name = 'Clustered_P_R_Model' + run_number
model_label, model_color = 'Two-Rayleigh model', 'b' #'Maximum AMD model', 'g' #'Two-Rayleigh model', 'b'





##### To load the underlying populations:

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated physical catalog of stars and planets:
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)





##### To plot the simulated catalog as marginal distributions:

subdirectory = '' #'Paper_Figures/'; 'Talk_Figures/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 1 #linewidth

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size



#'''
# Multiplicities:
x = np.concatenate((sssp_per_sys['Mtot_all'], np.zeros(N_sim - len(sssp_per_sys['Mtot_all']), dtype='int')))
plot_fig_counts_hist_simple(fig_size, [x], [], x_min=-1, x_llim=-0.5, x_ulim=10.5, normalize=True, lw=lw, xlabel_text='Intrinsic planet multiplicity', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, show_counts_sim=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_multiplicities.pdf', save_fig=savefigures)

# Clusters per system:
plot_fig_counts_hist_simple(fig_size, [sssp['clustertot_all']], [], x_llim=0.5, x_ulim=5.5, normalize=True, lw=lw, xlabel_text=r'Clusters per system $N_c$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, show_counts_sim=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_clusters_per_system.pdf', save_fig=savefigures)

# Planets per cluster:
plot_fig_counts_hist_simple(fig_size, [sssp['pl_per_cluster_all']], [], x_llim=0.5, x_ulim=7.5, normalize=True, lw=lw, xlabel_text=r'Planets per cluster $N_p$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, show_counts_sim=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_planets_per_cluster.pdf', save_fig=savefigures)

# Periods:
plot_fig_pdf_simple(fig_size, [sssp['P_all']], [], x_min=P_min, x_max=P_max, n_bins=n_bins, normalize=True, log_x=True, log_y=True, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_periods.pdf', save_fig=savefigures)

# Period ratios (all):
plot_fig_pdf_simple(fig_size, [sssp['Rm_all']], [], x_min=1., x_max=20., n_bins=n_bins, normalize=True, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20,50,100], xlabel_text=r'$P_{i+1}/P_i$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_periodratios.pdf', save_fig=savefigures)

# Period ratios (< 5):
plot_fig_pdf_simple(fig_size, [sssp['Rm_all'][sssp['Rm_all'] < 5]], [], x_min=1., x_max=5., n_bins=n_bins, normalize=True, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5], xlabel_text=r'$P_{i+1}/P_i$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_periodratios_less5.pdf', save_fig=savefigures)

# Eccentricities:
plot_fig_pdf_simple(fig_size, [sssp['e_all']], [], x_min=1e-3, x_max=1., n_bins=n_bins, normalize=True, log_x=True, lw=lw, xticks_custom=[1e-3,1e-2,1e-1,1.], xlabel_text=r'$e$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_eccentricities.pdf', save_fig=savefigures)

# Mutual inclinations:
plot_fig_pdf_simple(fig_size, [sssp['inclmut_all']*(180./np.pi)], [], x_min=1e-2, x_max=90., n_bins=n_bins, normalize=True, log_x=True, lw=lw, xticks_custom=[1e-2,1e-1,1.,10.,1e2], xlabel_text=r'$i_m$ (deg)', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_mutualinclinations.pdf', save_fig=savefigures)

# Planet masses:
plot_fig_pdf_simple(fig_size, [sssp['mass_all']], [], x_min=0.09, x_max=1e2, n_bins=n_bins, normalize=True, log_x=True, lw=lw, xlabel_text=r'$M_p$ ($M_\oplus$)', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_masses.pdf', save_fig=savefigures)

# Planet radii:
plot_fig_pdf_simple(fig_size, [sssp['radii_all']], [], x_min=0.5, x_max=10., n_bins=n_bins, normalize=True, log_x=True, lw=lw, xticks_custom=[0.5,1,2,4,10], xlabel_text=r'$R_p$ ($R_\oplus$)', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_radii.pdf', save_fig=savefigures)

# Planet radii (above and below the photoevaporation boundary):
plot_fig_pdf_simple(fig_size, [sssp['radii_above_all'], sssp['radii_below_all']], [], x_min=0.5, x_max=10., n_bins=n_bins, normalize=True, log_x=True, c_sim=['b','r'], ls_sim=['-','-'], lw=lw, labels_sim=['Above','Below'], xticks_custom=[0.5,1,2,4,10], xlabel_text=r'$R_p$ ($R_\oplus$)', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, legend=True, save_name=savefigures_directory + subdirectory + model_name + '_underlying_radii_photoevap.pdf', save_fig=savefigures)

# Planet radii ratios:
plot_fig_pdf_simple(fig_size, [sssp['radii_ratio_all']], [], x_min=0.1, x_max=10., n_bins=n_bins, normalize=True, log_x=True, lw=lw, xlabel_text=r'$R_{p,i+1}/R_{p,i}$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_radii_ratios.pdf', save_fig=savefigures)

# Planet radii ratios (above, below, and across the photoevaporation boundary):
plot_fig_pdf_simple(fig_size, [sssp['radii_ratio_above_all'], sssp['radii_ratio_below_all'], sssp['radii_ratio_across_all']], [], x_min=0.1, x_max=10., n_bins=n_bins, normalize=True, log_x=True, c_sim=['b','r','k'], ls_sim=['-','-','-'], lw=lw, labels_sim=['Above','Below','Across'], xlabel_text=r'$R_{p,i+1}/R_{p,i}$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, legend=True, save_name=savefigures_directory + subdirectory + model_name + '_underlying_radii_photoevap.pdf', save_fig=savefigures)

# Separations in mutual Hill radii:
plot_fig_pdf_simple(fig_size, [sssp['N_mH_all']], [], x_min=1., x_max=200., n_bins=n_bins, normalize=True, log_x=True, lw=lw, xlabel_text=r'$\Delta$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_deltas.pdf', save_fig=savefigures)

# Stellar radii:
plot_fig_pdf_simple(fig_size, [sssp['Rstar_all']], [], n_bins=n_bins, normalize=True, lw=lw, xlabel_text=r'$R_\star (R_\odot)$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_stellar_radii.pdf', save_fig=savefigures)

### GF2020 metrics, but for the underlying systems:
# Dynamical masses CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['dynamical_mass']], [], x_min=2e-7, x_max=1e-3, n_bins=n_bins, normalize=True, log_x=True, lw=lw, xlabel_text=r'$\mu$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_dynamical_masses.pdf', save_fig=savefigures)

# Planet radii partitioning CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['radii_partitioning']], [], x_min=1e-5, x_max=1., n_bins=n_bins, normalize=True, log_x=True, lw=lw, xlabel_text=r'$\mathcal{Q}_R$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_radii_partitioning.pdf', save_fig=savefigures)

# Planet radii monotonicity CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['radii_monotonicity']], [], x_min=-0.6, x_max=0.6, n_bins=n_bins, normalize=True, lw=lw, xlabel_text=r'$\mathcal{M}_R$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_radii_monotonicity.pdf', save_fig=savefigures)

# Gap complexity CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['gap_complexity']], [], x_min=0., x_max=1., n_bins=n_bins, normalize=True, log_x=False, lw=lw, xlabel_text=r'$\mathcal{C}$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_gap_complexity.pdf', save_fig=savefigures)

plt.show()
plt.close()
#'''





##### To plot the underlying multi-systems by period to visualize the systems (similar to Fig 1 in Fabrycky et al. 2014, but for ALL the planets):
##### Note: since there are way too many simulated systems to plot them all, we will randomly sample a number of systems to plot
#plot_figs_multis_underlying_gallery(sssp_per_sys, sssp, n_min=3, x_min=2., x_max=500., fig_size=(12,12), N_sys_sample=100, N_sys_per_plot=100, colorby='clusterid', tfs=tfs, save_name_base=savefigures_directory + subdirectory + model_name + '_underlying_systems_gallery', save_fig=False)
#plot_figs_multis_underlying_gallery(sssp_per_sys, sssp, n_min=3, n_det_min=1, x_min=2., x_max=300., fig_size=(5,12), panels_per_fig=1, N_sys_sample=100, N_sys_per_plot=100, plot_line_per=1, colorby='clusterid', mark_det=True, tfs=tfs, save_name_base=savefigures_directory + subdirectory + model_name + '_underlying_systems_gallery', save_fig=False)
plot_figs_multis_underlying_gallery(sssp_per_sys, sssp, n_min=3, n_det_min=1, x_min=2., x_max=300., fig_size=(6,8), panels_per_fig=1, N_sys_sample=140, N_sys_per_plot=140, plot_line_per=200, tfs=tfs, save_name_base=savefigures_directory + subdirectory + model_name + '_underlying_systems_gallery', save_fig=False)

#plot_figs_multis_underlying_gallery(sssp_per_sys, sssp, n_min=3, n_det_min=1, x_min=2., x_max=300., fig_size=(16,12), panels_per_fig=10, N_sys_sample=1000, N_sys_per_plot=100, plot_line_per=1, colorby='clusterid', mark_det=True, tfs=tfs, save_name_base=savefigures_directory + subdirectory + model_name + '_underlying_systems_gallery', save_fig=False)






##### To plot correlations between planet multiplicity, AMD, eccentricity, and mutual inclination:

# Planet multiplicity vs. AMD tot, eccentricity, and mutual inclination:
plot_fig_underlying_mult_vs_amd_ecc_incl(sssp_per_sys, sssp, n_min_max=[0.5, 13.5], amd_min_max=[1e-11, 1e-4], ecc_min_max=[1e-3, 1.], incl_min_max=[2e-2, 180.], afs=afs, tfs=tfs, lfs=lfs)
n_range = np.arange(2,15)
plt.scatter(incl_mult_power_law_Zhu2018(n_range, sigma_5=0.8, alpha=-3.5)*np.sqrt(2.*np.log(2.)), n_range, color='b', marker='x', s=100, label=r'$\sigma_{i,n} = 0.8(n/5)^{-3.5}$')
plt.scatter(incl_mult_power_law_Zhu2018(n_range, sigma_5=1., alpha=-2.)*np.sqrt(2.*np.log(2.)), n_range, color='r', marker='x', s=100, label=r'$\sigma_{i,n} = 1.0(n/5)^{-2}$')
plt.legend(loc='lower left', bbox_to_anchor=(-0.02,-0.02), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_mult_vs_amd_ecc_incl.pdf')
    plt.close()

# AMD tot vs. eccentricity and mutual inclination:
ax1, ax2 = plot_fig_underlying_amd_vs_ecc_incl(sssp_per_sys, sssp, show_singles=True, limit_singles=1000, amd_min_max=[2e-11, 1e-4], ecc_min_max=[1e-3, 1.], incl_min_max=[2e-2, 180.], afs=afs, tfs=tfs, lfs=lfs)
ax2.text(x=0.95, y=0.05, s=model_label, ha='right', fontsize=tfs, color=model_color, transform=ax2.transAxes)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_amd_vs_ecc_incl.pdf')
    plt.close()

# Mutual inclination vs eccentricity:
ax = plot_fig_underlying_ecc_vs_incl(sssp_per_sys, sssp, fig_lbrt=[0.2, 0.1, 0.95, 0.95], ecc_min_max=[1e-3, 1.], incl_min_max=[1e-2, 180.], afs=afs, tfs=tfs, lfs=lfs)
ax.text(x=0.95, y=0.05, s=model_label, ha='right', fontsize=tfs, color=model_color, transform=ax.transAxes)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_ecc_vs_incl.pdf')
    plt.close()

# Planet mass vs. AMD, eccentricity, and mutual inclination:
ax1, ax2, ax3 = plot_fig_underlying_mass_vs_amd_ecc_incl(sssp_per_sys, sssp, show_singles=True, limit_singles=1000, mass_min_max=[5e-2, 5e2], amd_min_max=[1e-13, 5e-5], ecc_min_max=[1e-3, 1.], incl_min_max=[2e-2, 180.], afs=afs, tfs=tfs, lfs=lfs)
ax1.text(x=0.05, y=0.95, s=model_label, ha='left', fontsize=tfs, color=model_color, transform=ax1.transAxes)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_mass_vs_amd_ecc_incl.pdf')
    plt.close()

# Minimum period ratio vs. AMD tot, eccentricity, and mutual inclination:
ax1, ax2, ax3 = plot_fig_underlying_pratio_min_vs_amd_ecc_incl(sssp_per_sys, sssp, pratio_min_max=[1., 5.], amd_min_max=[1e-11, 9e-5], ecc_min_max=[1e-3, 1.], incl_min_max=[2e-2, 180.], afs=afs, tfs=tfs, lfs=lfs)
ax1.text(x=0.05, y=0.95, s=model_label, ha='left', fontsize=tfs, color=model_color, transform=ax1.transAxes)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_pratio_min_vs_amd_ecc_incl.pdf')
    plt.close()

plt.show()
plt.close()



'''
for n in range(2,11):
    # AMD vs. eccentricity and mutual inclination:
    plot_fig_underlying_amd_vs_ecc_incl(sssp_per_sys, sssp, n_min=n, n_max=n, show_singles=True, amd_min_max=[2e-11, 1e-4], ecc_min_max=[1e-3, 1.], incl_min_max=[2e-2, 50.], afs=afs, tfs=tfs, lfs=lfs, save_name=savefigures_directory + subdirectory + model_name + '_underlying_amd_vs_ecc_incl_n%s.png' % n, save_fig=savefigures)
plt.show()

for n in range(2,11):
    # Mutual inclination vs eccentricity:
    plot_fig_underlying_ecc_vs_incl(sssp_per_sys, sssp, n_min=n, n_max=n, ecc_min_max=[1e-3, 1.], incl_min_max=[1e-2, 50.], afs=afs, tfs=tfs, lfs=lfs, save_name=savefigures_directory + subdirectory + model_name + '_underlying_ecc_vs_incl_n%s.png' % n, save_fig=savefigures)
plt.show()

for n in range(2,11):
    # Planet mass vs. AMD, eccentricity, and mutual inclination:
    plot_fig_underlying_mass_vs_amd_ecc_incl(sssp_per_sys, sssp, n_min=n, n_max=n, show_singles=True, mass_min_max=[5e-2, 5e2], amd_min_max=[1e-13, 5e-5], ecc_min_max=[1e-3, 1.], incl_min_max=[2e-2, 50.], afs=afs, tfs=tfs, lfs=lfs, save_name=savefigures_directory + subdirectory + model_name + '_underlying_mass_vs_amd_ecc_incl_n%s.png' % n, save_fig=savefigures)
plt.show()

for n in range(2,11):
    # Minimum period ratio vs. AMD, eccentricity, and mutual inclination:
    plot_fig_underlying_pratio_min_vs_amd_ecc_incl(sssp_per_sys, sssp, n_min=n, n_max=n, pratio_min_max=[1., 5.], amd_min_max=[1e-11, 9e-5], ecc_min_max=[1e-3, 1.], incl_min_max=[2e-2, 50.], afs=afs, tfs=tfs, lfs=lfs, save_name=savefigures_directory + subdirectory + model_name + '_underlying_pratio_min_vs_amd_ecc_incl_n%s.png' % n, save_fig=savefigures)
plt.show()
plt.close()
'''





##### To plot the eccentricity and mutual inclination distributions split by planet multiplicity, and fit a Rayleigh distribution to each:

#plot_fig_underlying_ecc_incl_per_mult(sssp_per_sys, sssp, n_min=1, n_max=10, fit_dists=True, log_x=True, alpha=0.3, ecc_min_max=[1e-3, 1.], incl_min_max=[1e-2, 90.], afs=afs, tfs=tfs, lfs=lfs, save_name=savefigures_directory + subdirectory + model_name + '_underlying_mult_vs_ecc_incl_dists.pdf', save_fig=savefigures)

#plot_fig_underlying_amd_ecc_incl_per_mult(sssp_per_sys, sssp, n_min=1, n_max=10, fit_dists=True, log_x=True, alpha=0.3, fig_size=(16,12), amd_min_max=[5e-11, 1e-4], ecc_min_max=[1e-3, 1.], incl_min_max=[2e-2, 90.], afs=afs, tfs=tfs, lfs=lfs, save_name=savefigures_directory + subdirectory + model_name + '_underlying_mult_vs_amd_ecc_incl_dists_long.pdf', save_fig=savefigures)

plt.show()





##### To plot radii monotonicity for different multiplicities:
'''
Mtot_2p = sssp_per_sys['Mtot_all'][sssp_per_sys['Mtot_all'] >= 2]
assert len(Mtot_2p) == len(sssp_per_sys['radii_monotonicity'])
#n = 2
for n in range(2,10):
    radii_mon_n = sssp_per_sys['radii_monotonicity'][Mtot_2p == n]
    plot_fig_pdf_simple(fig_size, [sssp_per_sys['radii_monotonicity'], radii_mon_n], [], x_min=-0.6, x_max=0.6, n_bins=n_bins, normalize=True, c_sim=['b','r'], ls_sim=['-','-'], lw=lw, labels_sim=['All',r'$n = %s$' % n], xlabel_text=r'$\mathcal{M}_R$', ylabel_text='Fraction', legend=True, afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_underlying_radii_monotonicity_%s.pdf' % n, save_fig=savefigures)
'''





##### To fit a power-law to the median eccentricities and mutual inclinations as a function of intrinsic multiplicity n:

n_array = np.arange(2,11)
e_med_array = np.zeros(len(n_array))
im_med_array = np.zeros(len(n_array))
for i,n in enumerate(n_array):
    e_n = sssp_per_sys['e_all'][sssp_per_sys['Mtot_all'] == n,:n]
    e_n = e_n.flatten()
    im_n = sssp_per_sys['inclmut_all'][sssp_per_sys['Mtot_all'] == n,:n]
    im_n = im_n.flatten() * (180./np.pi)

    e_med_array[i] = np.median(e_n)
    im_med_array[i] = np.median(im_n)

log_n = np.log10(n_array)
log_e_med = np.log10(e_med_array)
log_im_med = np.log10(im_med_array)

f_linear = lambda p, x: p[0] + p[1]*x - p[1]*np.log10(5.) # extra term for normalizing at n=5
f_err = lambda p, x, y: y - f_linear(p,x)
e_p0 = [0.03, -2.]
im_p0 = [1., -2.]

e_fit = scipy.optimize.leastsq(f_err, e_p0, args=(log_n, log_e_med), full_output=1)
e_logmu_alpha = e_fit[0]
mu_e5, alpha_e = 10.**(e_logmu_alpha[0]), e_logmu_alpha[1]

im_fit = scipy.optimize.leastsq(f_err, im_p0, args=(log_n, log_im_med), full_output=1)
im_logmu_alpha = im_fit[0]
mu_im5, alpha_im = 10.**(im_logmu_alpha[0]), im_logmu_alpha[1]





##### To plot period ratios and separations in mutual Hill radii vs. planet masses:

mass_sums_per_sys = sssp_per_sys['mass_all'][:,:-1] + sssp_per_sys['mass_all'][:,1:]

fig = plt.figure(figsize=(8,8)) # period ratios vs. sum of masses
plot = GridSpec(5,5,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
xmin, xmax = 0.1, 3e3
ymin, ymax = 1., 20.

ax = plt.subplot(plot[1:,:4])
corner.hist2d(np.log10(mass_sums_per_sys[sssp_per_sys['Rm_all'] > 0]), np.log10(sssp_per_sys['Rm_all'][sssp_per_sys['Rm_all']> 0]), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
#plt.scatter(np.log10(mass_sums_per_sys[sssp_per_sys['Rm_all'] > 0]), np.log10(sssp_per_sys['Rm_all'][sssp_per_sys['Rm_all']> 0]), marker='.', s=1, color='k')
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([0.1, 1., 10., 100., 1000.])
ytick_vals = np.array([1., 2., 3., 4., 5., 10., 20.])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(xmin), np.log10(xmax)])
plt.ylim([np.log10(ymin), np.log10(ymax)])
plt.xlabel(r'$m_{i+1}+m_i$ ($M_\oplus$)', fontsize=20)
plt.ylabel(r'$P_{i+1}/P_i$', fontsize=20)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist(mass_sums_per_sys[sssp_per_sys['Rm_all'] > 0], bins=np.logspace(np.log10(xmin), np.log10(xmax), n_bins+1), histtype='step', color='k', ls='-')
plt.gca().set_xscale("log")
plt.xlim([xmin, xmax])
plt.xticks([])
plt.yticks([])

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(sssp_per_sys['Rm_all'][sssp_per_sys['Rm_all']> 0], bins=np.logspace(np.log10(ymin), np.log10(ymax), n_bins+1), histtype='step', orientation='horizontal', color='k', ls='-')
plt.gca().set_yscale("log")
plt.ylim([ymin, ymax])
plt.xticks([])
plt.yticks([])

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_pratio_vs_mass.pdf')
    plt.close()
plt.show()



fig = plt.figure(figsize=(8,8)) # separations in mutual Hill radii vs. sum of masses
plot = GridSpec(5,5,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
xmin, xmax = 0.1, 3e3
ymin, ymax = 1., 300.

ax = plt.subplot(plot[1:,:4])
corner.hist2d(np.log10(mass_sums_per_sys[sssp_per_sys['N_mH_all'] > 0]), np.log10(sssp_per_sys['N_mH_all'][sssp_per_sys['N_mH_all']> 0]), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
#plt.scatter(np.log10(mass_sums_per_sys[sssp_per_sys['N_mH_all'] > 0]), np.log10(sssp_per_sys['N_mH_all'][sssp_per_sys['N_mH_all']> 0]), marker='.', s=1, color='k')
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([0.1, 1., 10., 100., 1000.])
ytick_vals = np.array([1., 10., 100., 1000.])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(xmin), np.log10(xmax)])
plt.ylim([np.log10(ymin), np.log10(ymax)])
plt.xlabel(r'$m_{i+1}+m_i$ ($M_\oplus$)', fontsize=20)
plt.ylabel(r'$\Delta$', fontsize=20)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.hist(mass_sums_per_sys[sssp_per_sys['N_mH_all'] > 0], bins=np.logspace(np.log10(xmin), np.log10(xmax), n_bins+1), histtype='step', color='k', ls='-')
plt.gca().set_xscale("log")
plt.xlim([xmin, xmax])
plt.xticks([])
plt.yticks([])

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(sssp_per_sys['N_mH_all'][sssp_per_sys['N_mH_all']> 0], bins=np.logspace(np.log10(ymin), np.log10(ymax), n_bins+1), histtype='step', orientation='horizontal', color='k', ls='-')
plt.gca().set_yscale("log")
plt.ylim([ymin, ymax])
plt.xticks([])
plt.yticks([])

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_mHill_vs_mass.pdf')
    plt.close()
plt.show()





##### To plot mutual inclinations and eccentricities vs periods:

x = sssp_per_sys['P_all'][np.sum(sssp_per_sys['P_all'] > 0., axis=1) > 1]
x = x[x > 0]

# Mutual inclinations:
y = sssp_per_sys['inclmut_all'][np.sum(sssp_per_sys['P_all'] > 0., axis=1) > 1]
y = 180./np.pi * y[y > 0]

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
corner.hist2d(np.log10(x), np.log10(y), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
ax.tick_params(axis='both', labelsize=afs)
#plt.gca().set_xscale("log")
#plt.gca().set_yscale("log")
plt.xlim([0.5, 2.5])
plt.ylim([-3., 2.])
plt.xlabel(r'$\log_{10}(P / {\rm days})$', fontsize=20)
plt.ylabel(r'$\log_{10}(i_m / {\rm deg})$', fontsize=20)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_mutualinclinations_vs_periods.pdf')
    plt.close()
plt.show()

# Eccentricities:
y = sssp_per_sys['e_all'][np.sum(sssp_per_sys['P_all'] > 0., axis=1) > 1]
y = y[y > 0]

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
corner.hist2d(np.log10(x), np.log10(y), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
ax.tick_params(axis='both', labelsize=afs)
#plt.gca().set_xscale("log")
#plt.gca().set_yscale("log")
plt.xlim([0.5, 2.5])
plt.ylim([-3., 0.])
plt.xlabel(r'$\log_{10}(P / {\rm days})$', fontsize=20)
plt.ylabel(r'$\log_{10}(e)$', fontsize=20)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_underlying_eccentricities_vs_periods.pdf')
    plt.close()
plt.show()
