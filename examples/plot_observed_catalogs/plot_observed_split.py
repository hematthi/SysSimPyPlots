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
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/'
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

#bins, i_blue_per_bin, i_red_per_bin = split_colors_per_cdpp_bin(stars_cleaned, nbins=10)

ssk_per_sys0, ssk0 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
ssk_per_sys1, ssk1 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_max=bp_rp_corr_med) #_max=_med
ssk_per_sys2, ssk2 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=bp_rp_corr_med) #_min=_med

label1, label2 = 'Bluer', 'Redder'



sss_per_sys0, sss0 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios) # combined sample
sss_per_sys1, sss1 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_max=bp_rp_corr_med, compute_ratios=compute_ratios)
sss_per_sys2, sss2 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, bp_rp_min=bp_rp_corr_med, compute_ratios=compute_ratios)

split_sss = [sss1, sss2]
split_sss_per_sys = [sss_per_sys1, sss_per_sys2]
split_ssk = [ssk1, ssk2]
split_ssk_per_sys = [ssk_per_sys1, ssk_per_sys2]
split_names = [label1, label2]
split_linestyles = ['-', '-']
split_linestyles_Kep = ['--', '--']
split_colors = ['b', 'r']



dists0, dists_w0 = compute_distances_sim_Kepler(sss_per_sys0, sss0, ssk_per_sys0, ssk0, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
dists1, dists_w1 = compute_distances_sim_Kepler(sss_per_sys1, sss1, ssk_per_sys1, ssk1, weights_all['bluer'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
dists2, dists_w2 = compute_distances_sim_Kepler(sss_per_sys2, sss2, ssk_per_sys2, ssk2, weights_all['redder'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)





##### To plot the simulated and Kepler catalogs as marginal distributions:

subdirectory = ''

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 2 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





# Multiplicities:
#'''
plot_fig_counts_hist_simple(fig_size, [sss_per_sys['Mtot_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['Mtot_obs'] for ssk_per_sys in split_ssk_per_sys], x_min=0, y_min=1e-1, y_max=1e4, x_llim=0.5, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ms_Kep=['x','x'], lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text='Observed planets per system', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_sim=True, show_counts_Kep=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_multiplicities_compare.pdf', save_fig=savefigures)

# Periods:
plot_fig_pdf_simple(fig_size, [sss['P_obs'] for sss in split_sss], [ssk['P_obs'] for ssk in split_ssk], x_min=P_min, x_max=P_max, y_min=0.5, y_max=1e2, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_x=True, log_y=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periods_compare.pdf', save_fig=savefigures)

# Periods of inner-most planet:
plot_fig_pdf_simple(fig_size, [sss_per_sys['P_obs'][:,0] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['P_obs'][:,0] for ssk_per_sys in split_ssk_per_sys], x_min=P_min, x_max=P_max, y_min=0.5, y_max=1e2, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_x=True, log_y=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P_1$ (days)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periods_inner_compare.pdf', save_fig=savefigures)

# Period ratios (all, with some upper cut-off):
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple(fig_size, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut] for sss in split_sss], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut] for ssk in split_ssk], x_min=1., x_max=R_max_cut, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$\mathcal{P} = P_{i+1}/P_i$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periodratios_compare.pdf', save_fig=savefigures)

# Transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_obs'] for sss in split_sss], [ssk['tdur_obs'] for ssk in split_ssk], x_min=0., x_max=15., n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$t_{\rm dur}$ (hrs)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_durations_compare.pdf', save_fig=savefigures)

# Circular normalized transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_obs'] for sss in split_sss], [ssk['tdur_tcirc_obs'] for ssk in split_ssk], x_min=0., x_max=1.5, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_compare.pdf', save_fig=savefigures)

# Circular normalized transit durations (separate singles and multis):
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_1_obs'] for sss in split_sss], [ssk['tdur_tcirc_1_obs'] for ssk in split_ssk], x_min=0., x_max=1.5, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], extra_text='Observed singles', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_compare.pdf', save_fig=savefigures)
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_2p_obs'] for sss in split_sss], [ssk['tdur_tcirc_2p_obs'] for ssk in split_ssk], x_min=0., x_max=1.5, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], extra_text='Observed multis', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_compare.pdf', save_fig=savefigures)

# Log(xi):
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_obs']) for sss in split_sss], [np.log10(ssk['xi_obs']) for ssk in split_ssk], x_min=-0.5, x_max=0.5, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\log{\xi}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_all_compare.pdf', save_fig=savefigures)

# Transit depths:
plot_fig_pdf_simple(fig_size, [sss['D_obs'] for sss in split_sss], [ssk['D_obs'] for ssk in split_ssk], x_min=1e-5, x_max=10.**-1.5, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\delta$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depths_compare.pdf', save_fig=savefigures)

# Transit depth ratios:
plot_fig_pdf_simple(fig_size, [sss['D_ratio_obs'] for sss in split_sss], [ssk['D_ratio_obs'] for ssk in split_ssk], x_min=10.**-1.5, x_max=10.**1.5, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depthratios_compare.pdf', save_fig=savefigures)

# Planet radii:
plot_fig_pdf_simple(fig_size, [sss['radii_obs'] for sss in split_sss], [ssk['radii_obs'] for ssk in split_ssk], x_min=radii_min, x_max=radii_max, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$R_p (R_\oplus)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_compare.pdf', save_fig=savefigures)

# Stellar radii:
plot_fig_pdf_simple(fig_size, [sss_per_sys['Rstar_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['Rstar_obs'] for ssk_per_sys in split_ssk_per_sys], n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$R_\star (R_\odot)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_stellar_radii_compare.pdf', save_fig=savefigures)

# Stellar masses:
plot_fig_pdf_simple(fig_size, [sss_per_sys['Mstar_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['Mstar_obs'] for ssk_per_sys in split_ssk_per_sys], x_max=2., n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$M_\star (M_\odot)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_stellar_masses_compare.pdf', save_fig=savefigures)

# Stellar teff:
plot_fig_pdf_simple(fig_size, [sss_per_sys['teff_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['teff_obs'] for ssk_per_sys in split_ssk_per_sys], n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$t_{\rm eff} (K)$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_stellar_teff_compare.pdf', save_fig=savefigures)

# Stellar colors:
plot_fig_pdf_simple(fig_size, [sss_per_sys['bp_rp_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['bp_rp_obs'] for ssk_per_sys in split_ssk_per_sys], n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$b_p - r_p$ (mag)', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_stellar_bprp_compare.pdf', save_fig=savefigures)

### GF2020 metrics:
# Planet radii partitioning:
plot_fig_pdf_simple(fig_size, [sss_per_sys['radii_partitioning'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['radii_partitioning'] for ssk_per_sys in split_ssk_per_sys], x_min=1e-5, x_max=1., n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\mathcal{Q}_R$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_partitioning_compare.pdf', save_fig=savefigures)

# Planet radii monotonicity:
plot_fig_pdf_simple(fig_size, [sss_per_sys['radii_monotonicity'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['radii_monotonicity'] for ssk_per_sys in split_ssk_per_sys], x_min=-0.5, x_max=0.6, n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\mathcal{M}_R$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_monotonicity_compare.pdf', save_fig=savefigures)

# Gap complexity:
plot_fig_pdf_simple(fig_size, [sss_per_sys['gap_complexity'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['gap_complexity'] for ssk_per_sys in split_ssk_per_sys], x_min=0., x_max=1., n_bins=n_bins, normalize=False, N_sim_Kep_factor=float(N_sim)/N_Kep, log_x=False, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\mathcal{C}$', ylabel_text='Number', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_gap_complexity_compare.pdf', save_fig=savefigures)

plt.show()
#'''





'''
x_sim_all = [[sss_per_sys['Mtot_obs'] for sss_per_sys in split_sss_per_sys],
             [sss['P_obs'] for sss in split_sss],
             [sss['Rm_obs'] for sss in split_sss],
             [sss['tdur_tcirc_obs'] for sss in split_sss],
             [np.log10(sss['xi_obs']) for sss in split_sss],
             [sss['D_obs'] for sss in split_sss],
             [sss['D_ratio_obs'] for sss in split_sss]]
x_Kep_all = [[ssk_per_sys['Mtot_obs'] for ssk_per_sys in split_ssk_per_sys],
             [ssk['P_obs'] for ssk in split_ssk],
             [ssk['Rm_obs'] for ssk in split_ssk],
             [ssk['tdur_tcirc_obs'] for ssk in split_ssk],
             [np.log10(ssk['xi_obs']) for ssk in split_ssk],
             [ssk['D_obs'] for ssk in split_ssk],
             [ssk['D_ratio_obs'] for ssk in split_ssk]]
x_mins = [0, P_min, 1., 0., -0.5, 1e-5, 10.**-1.5]
x_maxs = [None, P_max, 30., 1.5, 0.5, 10.**-1.5, 10.**1.5]
y_mins = [1e-2, 1e-1, None, None, None, None, None]
y_maxs = [1e4, 1e3, None, None, None, None, None]
log_xs = [False, True, True, False, False, True, True]
log_ys = [True, True, False, False, False, False, False]
xticks_customs = [None, [3,10,30,100,300], [1,2,3,4,5,10,20], None, None, None, None]
xlabel_texts = ['Observed planets per system', r'$P$ (days)', r'$\mathcal{P} = P_{i+1}/P_i$', r'$t_{\rm dur}/t_{\rm circ}$', r'$\log{\/xi}$', r'$\delta$', r'$\delta_{i+1}/\delta_i$']

plot_fig_pdf_composite(x_sim_all, x_Kep_all, param_vals=param_vals, x_mins=x_mins, x_maxs=x_maxs, y_mins=y_mins, y_maxs=y_maxs, n_bins=n_bins, N_sim_Kep_factor=float(N_sim)/N_Kep, log_xs=log_xs, log_ys=log_ys, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles, lw=lw, alpha=alpha, labels_sim=split_names, labels_Kep=[None, None], xticks_customs=xticks_customs, xlabel_texts=xlabel_texts, ylabel_texts=['Number'], afs=afs, tfs=tfs, lfs=lfs-4, legend_panels=2, fig_lbrt=[0.075, 0.1, 0.975, 0.975], save_name=savefigures_directory + subdirectory + model_name + '_summary_compare.pdf', save_fig=savefigures)
'''





#'''
# Multiplicity CDFs:
plot_fig_mult_cdf_simple(fig_size, [sss_per_sys['Mtot_obs'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['Mtot_obs'] for ssk_per_sys in split_ssk_per_sys], y_min=0.6, y_max=1., lw=lw, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, labels_sim=['Simulated',None], labels_Kep=['Kepler', None], xlabel_text='Observed planets per system', ylabel_text='CDF', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
plt.figtext(0.925, 0.65, r'$D_f = %s$' % np.round(dists1['delta_f'], 4), ha='right', color=split_colors[0], fontsize=lfs)
plt.figtext(0.925, 0.55, r'$\rho_{\rm CRPD} = %s$' % np.round(dists1['mult_CRPD_r'], 3), ha='right', color=split_colors[0], fontsize=lfs)
plt.figtext(0.925, 0.45, r'$D_f = %s$' % np.round(dists2['delta_f'], 4), ha='right', color=split_colors[1], fontsize=lfs)
plt.figtext(0.925, 0.35, r'$\rho_{\rm CRPD} = %s$' % np.round(dists2['mult_CRPD_r'], 3), ha='right', color=split_colors[1], fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + model_name + '_multiplicities_CDFs.pdf')

# Periods CDFs:
plot_fig_cdf_simple(fig_size, [sss['P_obs'] for sss in split_sss], [ssk['P_obs'] for ssk in split_ssk], x_min=P_min, x_max=P_max, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periods_CDFs.pdf', save_fig=savefigures)

# Periods of inner-most planet CDFs:
plot_fig_cdf_simple(fig_size, [sss_per_sys['P_obs'][:,0] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['P_obs'][:,0] for ssk_per_sys in split_ssk_per_sys], x_min=P_min, x_max=P_max, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xticks_custom=[3,10,30,100,300], xlabel_text=r'$P_1$ (days)', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periods_inner_CDFs.pdf', save_fig=savefigures)

# Period ratios CDFs:
plot_fig_cdf_simple(fig_size, [sss['Rm_obs'] for sss in split_sss], [ssk['Rm_obs'] for ssk in split_ssk], x_min=1., log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xticks_custom=[1,2,3,4,5,10,20,40], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_periodratios_CDFs.pdf', save_fig=savefigures)

# Transit durations CDFs:
plot_fig_cdf_simple(fig_size, [sss['tdur_obs'] for sss in split_sss], [ssk['tdur_obs'] for ssk in split_ssk], x_min=0., x_max=15., c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_durations_CDFs.pdf', save_fig=savefigures)

# Circular normalized transit durations:
plot_fig_cdf_simple(fig_size, [sss['tdur_tcirc_obs'] for sss in split_sss], [ssk['tdur_tcirc_obs'] for ssk in split_ssk], x_min=0., x_max=1.5, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_CDFs.pdf', save_fig=savefigures)

# Circular normalized transit durations (separate singles and multis):
plot_fig_cdf_simple(fig_size, [sss['tdur_tcirc_1_obs'] for sss in split_sss], [ssk['tdur_tcirc_1_obs'] for ssk in split_ssk], x_min=0., x_max=1.5, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], extra_text='Observed singles', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_singles_CDFs.pdf', save_fig=savefigures)
plot_fig_cdf_simple(fig_size, [sss['tdur_tcirc_2p_obs'] for sss in split_sss], [ssk['tdur_tcirc_2p_obs'] for ssk in split_ssk], x_min=0., x_max=1.5, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], extra_text='Observed multis', xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_tdur_tcirc_multis_CDFs.pdf', save_fig=savefigures)

# Transit depths CDFs:
plot_fig_cdf_simple(fig_size, [sss['D_obs'] for sss in split_sss], [ssk['D_obs'] for ssk in split_ssk], log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depths_CDFs.pdf', save_fig=savefigures)

# Planet radii CDFs:
plot_fig_cdf_simple(fig_size, [sss['radii_obs'] for sss in split_sss], [ssk['radii_obs'] for ssk in split_ssk], x_min=radii_min, x_max=radii_max, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$R_p (R_\oplus)$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_CDFs.pdf', save_fig=savefigures)

# Transit depth ratios CDFs:
plot_fig_cdf_simple(fig_size, [sss['D_ratio_obs'] for sss in split_sss], [ssk['D_ratio_obs'] for ssk in split_ssk], x_min=10.**-1.5, x_max=10.**1.5, log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_depthratios_CDFs.pdf', save_fig=savefigures)

# Log(xi) CDFs:
plot_fig_cdf_simple(fig_size, [np.log10(sss['xi_obs']) for sss in split_sss], [np.log10(ssk['xi_obs']) for ssk in split_ssk], x_min=-0.5, x_max=0.5, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_logxi_all_CDFs.pdf', save_fig=savefigures)

### GF2020 metrics:
# Planet radii partitioning CDFs:
plot_fig_cdf_simple(fig_size, [sss_per_sys['radii_partitioning'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['radii_partitioning'] for ssk_per_sys in split_ssk_per_sys], x_min=1e-5, x_max=1., log_x=True, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_partitioning_CDFs.pdf', save_fig=savefigures)

# Planet radii monotonicity CDFs:
plot_fig_cdf_simple(fig_size, [sss_per_sys['radii_monotonicity'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['radii_monotonicity'] for ssk_per_sys in split_ssk_per_sys], x_min=-0.5, x_max=0.6, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_radii_monotonicity_CDFs.pdf', save_fig=savefigures)

# Gap complexity CDFs:
plot_fig_cdf_simple(fig_size, [sss_per_sys['gap_complexity'] for sss_per_sys in split_sss_per_sys], [ssk_per_sys['gap_complexity'] for ssk_per_sys in split_ssk_per_sys], x_min=0., x_max=1., log_x=False, c_sim=split_colors, c_Kep=split_colors, ls_sim=split_linestyles, ls_Kep=split_linestyles_Kep, lw=lw, labels_sim=split_names, labels_Kep=[None, None], xlabel_text=r'$\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, label_dist=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + model_name + '_gap_complexity_CDFs.pdf', save_fig=savefigures)


plt.show()
#'''





##### To plot the stellar properties of the planet-hosting stars, with our divisions:

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
#plt.scatter(stars_cleaned['radius'], stars_cleaned['teff'], s=1, marker='.', label='All stars')
for i,name in enumerate(split_names):
    plt.scatter(split_sss_per_sys[i]['Rstar_obs'], split_sss_per_sys[i]['teff_obs'], s=1, c=split_colors[i], label=name)
plt.axvline(x=Rstar_med)
plt.axhline(y=teff_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$R_\star (R_\odot)$', fontsize=20)
plt.ylabel(r'$T_{\rm eff} (K)$', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=True, fontsize=16)

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
#plt.scatter(stars_cleaned['mass'], stars_cleaned['teff'], s=1, marker='.', label='All stars')
for i,name in enumerate(split_names):
    plt.scatter(split_sss_per_sys[i]['Mstar_obs'], split_sss_per_sys[i]['teff_obs'], s=1, c=split_colors[i], label=name)
plt.axvline(x=Mstar_med)
plt.axhline(y=teff_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$M_\star (M_\odot)$', fontsize=20)
plt.ylabel(r'$T_{\rm eff} (K)$', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=True, fontsize=16)

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.6)
ax = plt.subplot(plot[0,0])
#plt.scatter(stars_cleaned['radius'], stars_cleaned['mass'], s=1, marker='.', label='All stars')
for i,name in enumerate(split_names):
    plt.scatter(split_sss_per_sys[i]['Rstar_obs'], split_sss_per_sys[i]['Mstar_obs'], s=1, c=split_colors[i], label=name)
plt.axvline(x=Rstar_med)
plt.axhline(y=Mstar_med)
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$R_\star (R_\odot)$', fontsize=20)
plt.ylabel(r'$M_\star (M_\odot)$', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=True, fontsize=16)

plt.show()


