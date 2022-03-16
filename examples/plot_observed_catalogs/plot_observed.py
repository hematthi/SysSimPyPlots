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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *
from src.functions_plot_catalogs import *
from src.functions_plot_params import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #Extrapolate_P1000d/AMD_in_out/' #Extra_outputs/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/SysSimExClusters_plotting/examples/'
#savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Underlying/' #'/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/Best_models/GP_med/Underlying/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Misc_Presentations/PhD_Thesis_Defense/Figures/'
run_number = ''
model_name = 'Maximum_AMD_Model' + run_number

compute_ratios = compute_ratios_adjacent
AD_mod = True
weights_all = load_split_stars_weights_only()
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 'period_ratios_KS',
                 #'durations_KS',
                 #'durations_norm_circ_KS',
                 'durations_norm_circ_singles_KS',
                 'durations_norm_circ_multis_KS',
                 'duration_ratios_nonmmr_KS',
                 'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radius_ratios_KS',
                 'radii_partitioning_KS',
                 'radii_monotonicity_KS',
                 'gap_complexity_KS',
                 ]





##### To load the files with the systems with observed planets:

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated observed catalog of stars and planets:
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)





#'''
##### To plot the simulated and Kepler catalogs as marginal distributions:

subdirectory = '' #'Paper_Figures/'; 'Talk_Figures/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 3 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size



#'''
# To make a 'plot' listing the model parameters:
fig = plt.figure(figsize=fig_size)
plot = GridSpec(1,1,left=fig_lbrt[0],bottom=fig_lbrt[1],right=fig_lbrt[2],top=fig_lbrt[3],wspace=0.1,hspace=0.1)
nrows = 8
for i,param in enumerate(param_vals_all):
    plt.figtext(x=0.03+0.37*int(i/float(nrows)), y=0.875-0.1*(i%nrows), s=r'%s = %s' % (param_symbols[param], np.round(param_vals_all[param],3)), fontsize=lfs)
if savefigures == True:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_sim_params.pdf')
    plt.close()

# Multiplicities:
plot_fig_counts_hist_simple(fig_size, [sss_per_sys['Mtot_obs']], [ssk_per_sys['Mtot_obs']], x_min=0, x_max=7, y_max=2e3, x_llim=0.5, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, lw=lw, xlabel_text='Observed planets per system', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_sim=False, show_counts_Kep=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_multiplicities_compare.pdf', save_fig=savefigures)

# Periods:
plot_fig_pdf_simple(fig_size, [sss['P_obs']], [ssk['P_obs']], x_min=3., x_max=300., y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periods_compare.pdf', save_fig=savefigures)

# Period ratios (all, with some upper cut-off):
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple(fig_size, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut]], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, legend=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periodratios_compare.pdf', save_fig=savefigures)

# Period ratios (< 5):
plot_fig_pdf_simple(fig_size, [sss['Rm_obs'][sss['Rm_obs'] < 5.]], [ssk['Rm_obs'][ssk['Rm_obs'] < 5.]], x_min=1., x_max=5., n_bins=n_bins, lw=lw, xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periodratios_less5_compare.pdf', save_fig=savefigures)

# Transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_obs']], [ssk['tdur_obs']], x_min=0., x_max=15., n_bins=n_bins, lw=lw, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_durations_compare.pdf', save_fig=savefigures)

# Circular normalized transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_obs']], [ssk['tdur_tcirc_obs']], x_min=0., x_max=1.5, n_bins=n_bins, lw=lw, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_compare.pdf', save_fig=savefigures)

# Transit depths:
plot_fig_pdf_simple(fig_size, [sss['D_obs']], [ssk['D_obs']], log_x=True, lw=lw, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depths_compare.pdf', save_fig=savefigures)

# Transit depths (above and below the photoevaporation boundary):
plot_fig_pdf_simple(fig_size, [sss['D_above_obs'], sss['D_below_obs']], [ssk['D_above_obs'], ssk['D_below_obs']], n_bins=n_bins, log_x=True, c_sim=['b','r'], c_Kep=['b','r'], ls_sim=['-','-'], ls_Kep=['-','-'], lw=lw, labels_sim=['Above', 'Below'], labels_Kep=[None, None], xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depths_photoevap_compare.pdf', save_fig=savefigures)

# Planet radii:
plot_fig_pdf_simple(fig_size, [sss['radii_obs']], [ssk['radii_obs']], n_bins=n_bins, lw=lw, xlabel_text=r'$R_p (R_\oplus)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_compare.pdf', save_fig=savefigures)

# Transit depth ratios:
plot_fig_pdf_simple(fig_size, [sss['D_ratio_obs']], [ssk['D_ratio_obs']], n_bins=n_bins, log_x=True, lw=lw, xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, legend=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depthratios_compare.pdf', save_fig=savefigures)

# Transit depth ratios (above, below, and across the photoevaporation boundary):
plot_fig_pdf_simple(fig_size, [sss['D_ratio_above_obs'], sss['D_ratio_below_obs'], sss['D_ratio_across_obs']], [ssk['D_ratio_above_obs'], ssk['D_ratio_below_obs'], ssk['D_ratio_across_obs']], n_bins=n_bins, log_x=True, c_sim=['b','r','k'], c_Kep=['b','r','k'], ls_sim=['-','-','-'], ls_Kep=['-','-','-'], lw=lw, labels_sim=['Above', 'Below', 'Across'], labels_Kep=[None, None, None], xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depthratios_photoevap_compare.pdf', save_fig=savefigures)

# Log(xi):
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_obs'])], [np.log10(ssk['xi_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, lw=lw, xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_all_compare.pdf', save_fig=savefigures)

# Log(xi) by res/non-res:
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_res_obs']), np.log10(sss['xi_nonres_obs'])], [np.log10(ssk['xi_res_obs']), np.log10(ssk['xi_nonres_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=['m','g'], c_Kep=['m','g'], ls_sim=['-','-'], ls_Kep=['-','-'], lw=lw, labels_sim=['Near MMR', 'Not near MMR'], labels_Kep=[None, None], xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_compare.pdf', save_fig=savefigures)

# Log(xi) within res:
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_res32_obs']), np.log10(sss['xi_res21_obs'])], [np.log10(ssk['xi_res32_obs']), np.log10(ssk['xi_res21_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=['r','b'], c_Kep=['r','b'], ls_sim=['-','-'], ls_Kep=['-','-'], lw=lw, labels_sim=['Near 3:2 MMR', 'Near 2:1 MMR'], labels_Kep=[None, None], xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_res_compare.pdf', save_fig=savefigures)

# Stellar radii:
plot_fig_pdf_simple(fig_size, [sss['Rstar_obs']], [ssk['Rstar_obs']], n_bins=n_bins, lw=lw, xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_stellar_radii_compare.pdf', save_fig=savefigures)

# Stellar masses:
plot_fig_pdf_simple(fig_size, [sss['Mstar_obs']], [ssk['Mstar_obs']], x_max=2., n_bins=n_bins, lw=lw, xlabel_text=r'$M_\star (M_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_stellar_masses_compare.pdf', save_fig=savefigures)

# Stellar teff:
plot_fig_pdf_simple(fig_size, [sss['teff_obs']], [ssk['teff_obs']], n_bins=n_bins, lw=lw, xlabel_text=r'$t_{\rm eff} (K)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_stellar_teff_compare.pdf', save_fig=savefigures)

plt.show()
#plt.close()
#'''


# Multiplicity CDFs:
plot_fig_mult_cdf_simple(fig_size, [sss_per_sys['Mtot_obs']], [ssk_per_sys['Mtot_obs']], y_min=0.6, y_max=1., lw=lw, xlabel_text='Observed planets per system', ylabel_text='CDF', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.figtext(0.925, 0.45, r'$D_f = %s$' % np.round(dists['delta_f'], 4), ha='right', fontsize=lfs)
plt.figtext(0.925, 0.35, r'$\rho_{\rm CRPD} = %s$' % np.round(dists['mult_CRPD_r'], 3), ha='right', fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_multiplicities_CDFs.pdf')
    plt.close()

# Periods CDFs:
plot_fig_cdf_simple(fig_size, [sss['P_obs']], [ssk['P_obs']], x_min=P_min, x_max=P_max, log_x=True, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periods_CDFs.pdf', save_fig=savefigures)

# Period ratios CDFs:
plot_fig_cdf_simple(fig_size, [sss['Rm_obs']], [ssk['Rm_obs']], log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10,20,40], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periodratios_CDFs.pdf', save_fig=savefigures)

# Transit durations CDFs:
plot_fig_cdf_simple(fig_size, [sss['tdur_obs']], [ssk['tdur_obs']], x_min=0., x_max=15., lw=lw, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_durations_CDFs.pdf', save_fig=savefigures)

# Circular normalized transit durations CDFs:
plot_fig_cdf_simple(fig_size, [sss['tdur_tcirc_obs']], [ssk['tdur_tcirc_obs']], x_min=0., x_max=1.5, lw=lw, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_CDFs.pdf', save_fig=savefigures)

# Transit depths CDFs:
plot_fig_cdf_simple(fig_size, [sss['D_obs']], [ssk['D_obs']], log_x=True, lw=lw, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depths_CDFs.pdf', save_fig=savefigures)

# Transit depths CDFs (above and below the photoevaporation boundary):
plot_fig_cdf_simple(fig_size, [sss['D_above_obs'], sss['D_below_obs']], [ssk['D_above_obs'], ssk['D_below_obs']], log_x=True, c_sim=['b','r'], c_Kep=['b','r'], ls_sim=['-','-'], ls_Kep=['--','--'], lw=lw, labels_sim=['Above', 'Below'], labels_Kep=[None, None], xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, legend=True, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depths_photoevap_CDFs.pdf', save_fig=savefigures)

# Planet radii CDFs:
plot_fig_cdf_simple(fig_size, [sss['radii_obs']], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, lw=lw, xlabel_text=r'$R_p (R_\oplus)$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_CDFs.pdf', save_fig=savefigures)

# Transit depth ratios CDFs:
plot_fig_cdf_simple(fig_size, [sss['D_ratio_obs']], [ssk['D_ratio_obs']], x_min=10.**-1.5, x_max=10.**1.5, log_x=True, lw=lw, xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depthratios_CDFs.pdf', save_fig=savefigures)

# Transit depth ratios CDFs (above, below, and across the photoevaporation boundary):
plot_fig_cdf_simple(fig_size, [sss['D_ratio_above_obs'], sss['D_ratio_below_obs'], sss['D_ratio_across_obs']], [ssk['D_ratio_above_obs'], ssk['D_ratio_below_obs'], ssk['D_ratio_across_obs']], log_x=True, c_sim=['b','r','k'], c_Kep=['b','r','k'], ls_sim=['-','-','-'], ls_Kep=['--','--','--'], lw=lw, labels_sim=['Above', 'Below', 'Across'], labels_Kep=[None, None, None], xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, legend=True, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depthratios_photoevap_CDFs.pdf', save_fig=savefigures)

# Log(xi) CDFs:
plot_fig_cdf_simple(fig_size, [np.log10(sss['xi_obs'])], [np.log10(ssk['xi_obs'])], x_min=-0.5, x_max=0.5, lw=lw, xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_all_CDFs.pdf', save_fig=savefigures)

# Log(xi) CDFs by res/non-res:
plot_fig_cdf_simple(fig_size, [np.log10(sss['xi_res_obs']), np.log10(sss['xi_nonres_obs'])], [np.log10(ssk['xi_res_obs']), np.log10(ssk['xi_nonres_obs'])], x_min=-0.5, x_max=0.5, c_sim=['m','g'], c_Kep=['m','g'], ls_sim=['-','-'], ls_Kep=['--','--'], lw=lw, labels_sim=['Near MMR', 'Not near MMR'], labels_Kep=[None, None], xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_CDFs.pdf', save_fig=savefigures)

# Log(xi) CDFs within res:
plot_fig_cdf_simple(fig_size, [np.log10(sss['xi_res32_obs']), np.log10(sss['xi_res21_obs'])], [np.log10(ssk['xi_res32_obs']), np.log10(ssk['xi_res21_obs'])], x_min=-0.5, x_max=0.5, c_sim=['r','b'], c_Kep=['r','b'], ls_sim=['-','-'], ls_Kep=['--','--'], lw=lw, labels_sim=['Near 3:2 MMR', 'Near 2:1 MMR'], labels_Kep=[None, None], xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_res_CDFs.pdf', save_fig=savefigures)

plt.show()
#plt.close()
#'''





##### To plot the observed multi-systems by period to visualize the systems (similar to Fig 1 in Fabrycky et al. 2014):

#load_cat_obs_and_plot_figs_multis_gallery(loadfiles_directory, run_number=run_number, n_pl=3, plot_Kep=True, show_title=False, fig_size=(8,8), N_sys_per_plot=140, plot_line_per=1, tfs=20, save_name_base=savefigures_directory + subdirectory + model_name + '_multis', save_fig=savefigures)
#load_cat_obs_and_plot_figs_multis_gallery(loadfiles_directory, run_number=run_number, n_pl=3, plot_Kep=False, show_title=False, fig_size=(6,8), N_sys_per_plot=140, plot_line_per=1, tfs=20, save_name_base=savefigures_directory + subdirectory + model_name + '_multis', save_fig=savefigures)
plt.show()
#plt.close()





##### To plot the inner vs. outer period ratios of triplets (in 3+ systems) (similar to Fig 6 in Zhu et al. 2019 and Fig 7 in Weiss et al. 2018a):
'''
p_per_sys_subset = sss_per_sys['P_obs'][np.random.choice(np.arange(len(sss_per_sys['P_obs'])), int(round(len(sss_per_sys['P_obs'])/(N_sim/N_Kep))), replace=False)]
compute_pratio_in_out_and_plot_fig([p_per_sys_subset, ssk_per_sys['P_obs']], colors=['b','k'], labels=['Simulated', 'Kepler'], xymax=25., xyticks_custom=[1,2,3,4,5,10,20], afs=afs, tfs=tfs, lfs=lfs, save_name='Meeting_plots/July_9_2019/' + model_name + '_observed_pratio_in_out.pdf', save_fig=False)

compute_pratio_in_out_and_plot_fig_pdf([sss_per_sys['P_obs'], ssk_per_sys['P_obs']], n_bins=100, x_min=0.1, x_max=10., colors=['b','k'], ls=['-','-'], labels=['Simulated', 'Kepler'], afs=afs, tfs=tfs, lfs=lfs, save_name='Meeting_plots/July_9_2019/' + model_name + '_observed_pratio_out_in_ratio.pdf', save_fig=False)
plt.show()
plt.close()
'''




##### To plot a histogram of the zeta statistic (similar to Fig 5 in Fabrycky et al. 2014):

pratios_small_sim = sss['Rm_obs'][sss['Rm_obs'] < 2.5]
pratios_small_Kep = ssk['Rm_obs'][ssk['Rm_obs'] < 2.5]
zeta1_small_sim = zeta1(pratios_small_sim)
zeta1_small_Kep = zeta1(pratios_small_Kep)
plot_fig_pdf_simple(fig_size, [zeta1_small_sim], [zeta1_small_Kep], x_min=-1., x_max=1., n_bins=30, normalize=True, lw=lw, labels_Kep=['Period ratios < 2.5'], xlabel_text=r'$\zeta_1$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_zeta1_small_compare.pdf', save_fig=savefigures)
plt.show()
#plt.close()





##### To plot the eccentricities of the planets in observed singles and multis (requires loading the physical catalog and matching target_id's):
'''
cat_phys = load_cat_phys(loadfiles_directory + 'physical_catalog%s.csv' % run_number)
cat_obs = load_cat_obs(loadfiles_directory + 'observed_catalog%s.csv' % run_number)

e_singles = []
e_multis = []
for i,target_id in enumerate(cat_obs['target_id']):
    i_target = np.where(cat_phys['target_id'] == target_id)[0]
    system_phys = cat_phys[i_target]
    e_planet = system_phys['ecc'][np.argmin(np.abs(system_phys['period'] - cat_obs['period'][i]))]
    system_M = np.sum(cat_obs['target_id'] == target_id)
    if system_M > 1:
        e_multis.append(e_planet)
    elif system_M == 1:
        e_singles.append(e_planet)
    else:
        print('Error: unexpected value for system_M?')
e_singles = np.array(e_singles)
e_multis = np.array(e_multis)

plot_fig_pdf_simple(fig_size, [e_singles, e_multis, cat_phys['ecc']], [], n_bins=n_bins, c_sim=['b','r','k'], ls_sim=['-','--',':'], lw=lw, labels_sim=['Observed singles', 'Observed multis', 'Intrinsic distribution'], xlabel_text=r'$e$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_ecc_singles_multis.pdf', save_fig=savefigures)

plot_fig_cdf_simple(fig_size, [e_singles, e_multis, cat_phys['ecc']], [], c_sim=['b','r','k'], ls_sim=['-','--',':'], lw=lw, labels_sim=['Observed singles', 'Observed multis', 'Intrinsic distribution'], xlabel_text=r'$e$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_ecc_singles_multis_CDFs.pdf', save_fig=savefigures)

plt.show()
#plt.close()
'''





##### To plot CDFs for some of the system-level metrics defined in GF2020:

# Planet to star radii ratio CDFs:
plot_fig_cdf_simple(fig_size, [sss_per_sys['radii_star_ratio']*(Rearth/Rsun)], [ssk_per_sys['radii_star_ratio']*(Rearth/Rsun)], log_x=True, lw=lw, xlabel_text=r'$\sum{R_p/R_\star}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_star_ratio_CDFs.pdf', save_fig=savefigures)

# Planet radii partitioning CDFs:
plot_fig_cdf_simple(fig_size, [sss_per_sys['radii_partitioning']], [ssk_per_sys['radii_partitioning']], lw=lw, xlabel_text=r'$\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_partitioning_CDFs.pdf', save_fig=savefigures)
#plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), ncol=1, frameon=False, fontsize=lfs)

# Planet radii monotonicity CDFs:
plot_fig_cdf_simple(fig_size, [sss_per_sys['radii_monotonicity']], [ssk_per_sys['radii_monotonicity']], lw=lw, xlabel_text=r'$\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_monotonicity_CDFs.pdf', save_fig=savefigures)

# Gap complexity CDFs:
plot_fig_cdf_simple(fig_size, [sss_per_sys['gap_complexity']], [ssk_per_sys['gap_complexity']], lw=lw, xlabel_text=r'$\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_gap_complexity_CDFs.pdf', save_fig=savefigures)

plt.show()





##### To plot the period, depth, duration, and normalized durations separating the singles and the multis:

#'''
# Transit durations:
tdur_1 = sss_per_sys['tdur_obs'][sss_per_sys['Mtot_obs'] == 1, 0] # observed singles
tdur_Kep_1 = ssk_per_sys['tdur_obs'][ssk_per_sys['Mtot_obs'] == 1, 0]
tdur_2p = sss_per_sys['tdur_obs'][sss_per_sys['Mtot_obs'] > 1] # observed multis
tdur_2p = tdur_2p[tdur_2p > 0]
tdur_Kep_2p = ssk_per_sys['tdur_obs'][ssk_per_sys['Mtot_obs'] > 1]
tdur_Kep_2p = tdur_Kep_2p[tdur_Kep_2p > 0]
plot_fig_pdf_simple(fig_size, [tdur_1], [tdur_Kep_1], x_min=0., x_max=15., n_bins=n_bins, lw=lw, labels_sim=['Singles'], labels_Kep=[''], xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_durations_singles_compare.pdf', save_fig=savefigures)
plot_fig_pdf_simple(fig_size, [tdur_2p], [tdur_Kep_2p], x_min=0., x_max=15., n_bins=n_bins, lw=lw, labels_sim=['Multis'], labels_Kep=[''], xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_durations_multis_compare.pdf', save_fig=savefigures)

# Circular normalized transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_1_obs']], [ssk['tdur_tcirc_1_obs']], x_min=0., x_max=1.5, n_bins=n_bins, lw=lw, labels_sim=['Singles'], labels_Kep=[''], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_compare.pdf', save_fig=savefigures)
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_2p_obs']], [ssk['tdur_tcirc_2p_obs']], x_min=0., x_max=1.5, n_bins=n_bins, lw=lw, labels_sim=['Multis'], labels_Kep=[''], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_compare.pdf', save_fig=savefigures)

# Periods:
P_1 = sss_per_sys['P_obs'][sss_per_sys['Mtot_obs'] == 1, 0] # observed singles
P_Kep_1 = ssk_per_sys['P_obs'][ssk_per_sys['Mtot_obs'] == 1, 0]
P_2p = sss_per_sys['P_obs'][sss_per_sys['Mtot_obs'] > 1] # observed multis
P_2p = P_2p[P_2p > 0]
P_Kep_2p = ssk_per_sys['P_obs'][ssk_per_sys['Mtot_obs'] > 1]
P_Kep_2p = P_Kep_2p[P_Kep_2p > 0]
plot_fig_pdf_simple(fig_size, [P_1], [P_Kep_1], x_min=3., x_max=300., n_bins=n_bins, log_x=True, log_y=False, lw=lw, xticks_custom=[3,10,30,100,300], labels_sim=['Singles'], labels_Kep=[''], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periods_singles_compare.pdf', save_fig=savefigures)
plot_fig_pdf_simple(fig_size, [P_2p], [P_Kep_2p], x_min=3., x_max=300., n_bins=n_bins, log_x=True, log_y=False, lw=lw, xticks_custom=[3,10,30,100,300], labels_sim=['Multis'], labels_Kep=[''], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periods_multis_compare.pdf', save_fig=savefigures)

# Transit depths:
D_1 = sss_per_sys['D_obs'][sss_per_sys['Mtot_obs'] == 1, 0] # observed singles
D_Kep_1 = ssk_per_sys['D_obs'][ssk_per_sys['Mtot_obs'] == 1, 0]
D_2p = sss_per_sys['D_obs'][sss_per_sys['Mtot_obs'] > 1] # observed multis
D_2p = D_2p[D_2p > 0]
D_Kep_2p = ssk_per_sys['D_obs'][ssk_per_sys['Mtot_obs'] > 1]
D_Kep_2p = D_Kep_2p[D_Kep_2p > 0]
plot_fig_pdf_simple(fig_size, [D_1], [D_Kep_1], n_bins=n_bins, log_x=True, lw=lw, labels_sim=['Singles'], labels_Kep=[''], xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depths_singles_compare.pdf', save_fig=savefigures)
plot_fig_pdf_simple(fig_size, [D_2p], [D_Kep_2p], n_bins=n_bins, log_x=True, lw=lw, labels_sim=['Multis'], labels_Kep=[''], xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depths_multis_compare.pdf', save_fig=savefigures)

plt.show()
#'''





##### To plot the xi distribution separated by observed multiplicities (m=2,3,4+):

logxi_Kep_2 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 2, 0])
logxi_Kep_3 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 3, :2].flatten())
xi_Kep_4p = ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] >= 4]
logxi_Kep_4p = np.log10(xi_Kep_4p[xi_Kep_4p != -1])

logxi_2 = np.log10(sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] == 2, 0])
logxi_3 = np.log10(sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] == 3, :2].flatten())
xi_4p = sss_per_sys['xi_obs'][sss_per_sys['Mtot_obs'] >= 4]
logxi_4p = np.log10(xi_4p[xi_4p != -1])

plot_fig_pdf_simple(fig_size, [logxi_2, logxi_3, logxi_4p], [np.log10(ssk['xi_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=['r','b','g'], c_Kep=['k'], ls_sim=['-','-','-'], ls_Kep=['-'], lw=lw, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None], xlabel_text=r'$\log{\xi}$', legend=True, afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_per_mult_compare.pdf', save_fig=savefigures)
plot_fig_cdf_simple(fig_size, [logxi_2, logxi_3, logxi_4p], [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], x_min=-0.5, x_max=0.5, c_sim=['r','b','g'], c_Kep=['r','b','g'], ls_sim=['-','-','-'], ls_Kep=[':',':',':'], lw=1, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None,None,None], xlabel_text=r'$\log{\xi}$', legend=True, afs=afs, tfs=tfs, lfs=lfs, label_dist=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_per_mult_CDFs.pdf', save_fig=savefigures)

# Just Kepler data:
plot_fig_pdf_simple(fig_size, [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], [], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=['r','b','g'], ls_sim=['-','-','-'], lw=lw, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None], xlabel_text=r'$\log{\xi}$', legend=True, afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_per_mult_Kepler.pdf', save_fig=savefigures)
plot_fig_cdf_simple(fig_size, [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], [], x_min=-0.5, x_max=0.5, c_sim=['r','b','g'], ls_sim=[':',':',':'], lw=1, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], xlabel_text=r'$\log{\xi}$', legend=True, afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_per_mult_CDFs_Kepler.pdf', save_fig=savefigures)





##### For proposals:

plot_fig_pdf_simple((8,4), [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut]], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=10., n_bins=n_bins, log_x=True, lw=lw, xticks_custom=[1,2,3,4,5,10], xlabel_text=r'Period ratio $P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periodratios_compare.pdf', save_fig=savefigures)
plt.minorticks_off()
plt.fill_betweenx(np.array([0,1]), 1.5, 1.5*1.05, alpha=0.2, color='r')
plt.fill_betweenx(np.array([0,1]), 2., 2.*1.05, alpha=0.2, color='r')
plt.text(1.5, 0.045, s='3:2', ha='right', fontsize=lfs)
plt.text(2., 0.045, s='2:1', ha='right', fontsize=lfs)
plt.show()





##### To plot circular normalized transit durations for each observed multiplicity:

tdur_tcirc_Kep_per_m = [] # 1,2,3,4+
for m in [1,2,3,4]:
    if m==4:
        tdur_tcirc_m = sss_per_sys['tdur_tcirc_obs'][sss_per_sys['Mtot_obs'] == m]
        tdur_tcirc_Kep_m = ssk_per_sys['tdur_tcirc_obs'][ssk_per_sys['Mtot_obs'] == m]
        tdur_tcirc_m = tdur_tcirc_m.flatten()
        tdur_tcirc_m = tdur_tcirc_m[~np.isnan(tdur_tcirc_m)]
        tdur_tcirc_Kep_m = tdur_tcirc_Kep_m.flatten()
        tdur_tcirc_Kep_m = tdur_tcirc_Kep_m[~np.isnan(tdur_tcirc_Kep_m)]
        label = r'$m = %s+$' % m
    else:
        tdur_tcirc_m = sss_per_sys['tdur_tcirc_obs'][sss_per_sys['Mtot_obs'] == m, :m]
        tdur_tcirc_Kep_m = ssk_per_sys['tdur_tcirc_obs'][ssk_per_sys['Mtot_obs'] == m, :m]
        label = r'$m = %s$' % m
    
    plot_fig_pdf_simple(fig_size, [tdur_tcirc_m.flatten()], [tdur_tcirc_Kep_m.flatten()], x_min=0., x_max=1.5, n_bins=n_bins, lw=lw, labels_sim=[label], labels_Kep=[''], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_m%s_compare.pdf' % m, save_fig=savefigures)

    tdur_tcirc_Kep_per_m.append(tdur_tcirc_Kep_m.flatten())

plot_fig_cdf_simple(fig_size, [], tdur_tcirc_Kep_per_m, x_min=0., x_max=1.5, lw=lw, c_Kep=['k','b','g','r'], ls_Kep=['--','--','--','--'], labels_Kep=[r'$m=1$',r'$m=2$',r'$m=3$',r'$m=4+$'], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_per_mult_CDFs.pdf', save_fig=savefigures)






##### To plot the fraction of planets in observed multis on a period-radius diagram:

P_bins = np.logspace(np.log10(P_min), np.log10(P_max), 6+1)
R_bins = np.logspace(np.log10(radii_min), np.log10(radii_max), 6+1)
#P_bins = np.array([4., 8., 16., 32., 64., 128., 256.])
#R_bins = np.array([0.5, 1., 1.5, 2., 3., 4., 6.])

plot_fig_period_radius_fraction_multis(sss_per_sys, sss, P_bins, R_bins)
plot_fig_period_radius_fraction_multis(ssk_per_sys, ssk, P_bins, R_bins)
plt.show()

plot_fig_period_radius_fraction_multis_higher(sss_per_sys, sss, P_bins, R_bins)
plot_fig_period_radius_fraction_multis_higher(ssk_per_sys, ssk, P_bins, R_bins)
plt.show()





##### To plot observed multiplicity vs. period:

fig = plt.figure(figsize=(10,8)) # separations in mutual Hill radii vs. sum of masses
plot = GridSpec(2,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
bins = np.logspace(np.log10(P_min), np.log10(P_max), 21)

ax = plt.subplot(plot[0,:]) # simulated catalog
plt.text(x=0.02, y=0.05, s='Maximum AMD model', ha='left', fontsize=lfs, transform=ax.transAxes)
for m in [1,2,3,4]:
    if m == 4: # group all 4+
        P_m = sss_per_sys['P_obs'][sss_per_sys['Mtot_obs'] >= m,:]
        P_m = P_m[P_m > 0]
        m_label = r'$m = 4+$'
    else: # m < 4
        P_m = sss_per_sys['P_obs'][sss_per_sys['Mtot_obs'] == m,:m]
        P_m = P_m.flatten()
        m_label = r'$m = %s$' % m
    print(len(P_m), ' -- ', np.sum(P_m > 100.))
    plt.hist(P_m, bins=bins, weights=np.ones(len(P_m))/5, histtype='step', lw=2, label=m_label) #weights=np.ones(len(P_m))/len(P_m)
    #plt.scatter(P_m, m*np.ones(len(P_m)))
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xlim([P_min, P_max])
plt.xlabel(r'$P$ (days)', fontsize=20)
plt.ylabel(r'Counts', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1,:]) # Kepler
plt.text(x=0.02, y=0.05, s='Kepler data', ha='left', fontsize=lfs, transform=ax.transAxes)
for m in [1,2,3,4]:
    if m == 4: # group all 4+
        P_m = ssk_per_sys['P_obs'][ssk_per_sys['Mtot_obs'] >= m,:]
        P_m = P_m[P_m > 0]
        m_label = r'$m = 4+$'
    else: # m < 4
        P_m = ssk_per_sys['P_obs'][ssk_per_sys['Mtot_obs'] == m,:m]
        P_m = P_m.flatten()
        m_label = r'$m = %s$' % m
    print(len(P_m), ' -- ', np.sum(P_m > 100.))
    plt.hist(P_m, bins=bins, histtype='step', lw=2, label=m_label) #weights=np.ones(len(P_m))/len(P_m)
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xlim([P_min, P_max])
plt.xlabel(r'$P$ (days)', fontsize=20)
plt.ylabel(r'Counts', fontsize=20)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_periods_per_mult_counts.pdf')
    plt.close()
plt.show()





##### To plot radius of outer vs inner planet pairs for proposals:
'''
directory = '/Users/hematthi/Documents/GradSchool/Postdoctoral_Applications/2021/Figures/'

Rp_per_sys_subset = sss_per_sys['radii_obs'][np.random.choice(np.arange(len(sss_per_sys['radii_obs'])), int(round(len(sss_per_sys['radii_obs'])/(N_sim/N_Kep))), replace=False)]
Rp_in_out = []
for Rp_sys in Rp_per_sys_subset:
    n_pl = len(Rp_sys)
    if n_pl > 1:
        for i in range(n_pl-1):
            Rp_in_out.append([Rp_sys[i], Rp_sys[i+1]])
Rp_in_out = np.array(Rp_in_out)

Rp_in_out_Kep = []
for Rp_sys in ssk_per_sys['radii_obs']:
    n_pl = len(Rp_sys)
    if n_pl > 1:
        for i in range(n_pl-1):
            Rp_in_out_Kep.append([Rp_sys[i], Rp_sys[i+1]])
Rp_in_out_Kep = np.array(Rp_in_out_Kep)

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5,5,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
plt.scatter(np.log10(Rp_in_out[:,0]), np.log10(Rp_in_out[:,1]), s=5, marker='o', c='b', label='Simulated systems')
plt.scatter(np.log10(Rp_in_out_Kep[:,0]), np.log10(Rp_in_out_Kep[:,1]), s=5, marker='o', c='k', label='Kepler systems')
plt.plot(np.linspace(-1.,1.,100), np.linspace(-1.,1.,100), ls='--', c='r', label=r'$y=x$ (equal radii)')
ax.tick_params(axis='both', labelsize=16)
xtick_vals = np.array([0.5,1,1.5,2,3,4,10])
ytick_vals = np.array([0.5,1,1.5,2,3,4,10])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(0.5), np.log10(10.)])
plt.ylim([np.log10(0.5), np.log10(10.)])
plt.xlabel(r'Inner planet radius, $R_{p,i}$ ($R_\oplus$)', fontsize=20)
plt.ylabel(r'Outer planet radius, $R_{p,i+1}$ ($R_\oplus$)', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=16)
if savefigures:
    plt.savefig(directory + 'radius_out_vs_in_pairs.pdf')
    plt.close()
plt.show()
'''
