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

savefigures_directory = '/Users/hematthi/Documents/GradSchool/Misc_Presentations/PhD_Thesis_Defense/Figures/'
save_name = 'Models_Compare_Kepler'

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

# Model 1:
loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/incl0_ecc0p02/' #incl0_ecc0p02/ #ecc0_incl1/
run_number1 = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory1 + 'periods%s.out' % run_number1)

param_vals_all1 = read_sim_params(loadfiles_directory1 + 'periods%s.out' % run_number1)
sss_per_sys1, sss1 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory1, run_number=run_number1, compute_ratios=compute_ratios)

# Model 2:
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/incl2_ecc0p02/' #incl2_ecc0p02/ #ecc0p1_incl1/
run_number2 = ''

param_vals_all2 = read_sim_params(loadfiles_directory2 + 'periods%s.out' % run_number2)
sss_per_sys2, sss2 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory2, run_number=run_number2, compute_ratios=compute_ratios)

# Model 3:
#loadfiles_directory3 = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/SysSimExClusters/examples/New_algorithm_AMD/Same_params/Stable/New_sampler_P_Pc/No_final_mHill_check/' #'ACI/Simulated_Data/Julia_v0.7/Kepler_catalog_optimization/q1q17_dr25_gaia_fgk_stars79935/Non_Clustered/f_high_incl_low_incl_mmr/Fit_rate_mult_P_Pratios_D_Dratios_dur_durratios_mmr/Some8_params_CRPDr_KS/Fixed_Rbreak3_Ncrit8/lc_1_8_alphaP_-2_2_alphaR1_-4_2_alphaR2_-6_0_ecc_0_0p1_incl_inclmmr_0_90/targs79935_maxincl0_maxiters5000/sigma_i_greater_sigma_i_mmr/GP_med/'
#run_number3 = ''

#param_vals_all3 = read_sim_params(loadfiles_directory3 + 'periods%s.out' % run_number3)
#sss_per_sys3, sss3 = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory3, run_number=run_number3, compute_ratios=compute_ratios)



model_sss = [sss1, sss2]
model_sss_per_sys = [sss_per_sys1, sss_per_sys2]
model_names = [r'$\sigma_i = 0^\circ$', r'$\sigma_i = 2^\circ$'] #[r'$\sigma_e = 0$', r'$\sigma_e = 0.1$'] #[r'$\sigma_i = 0^\circ$', r'$\sigma_i = 2^\circ$'] #[r'$\omega = {\rm atan}(x,y)$', r'$\omega \sim {\rm Unif}(-\pi,\pi)$'] # Make sure this matches the models loaded!
model_linestyles = ['-', '-']
model_colors = ['b', 'r'] #['b', 'r']

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max)

logxi_Kep_2 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 2, 0])
logxi_Kep_3 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 3, :2].flatten())
logxi_Kep_4 = np.log10(ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] == 4, :3].flatten())
xi_Kep_4p = ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] >= 4]
logxi_Kep_4p = np.log10(xi_Kep_4p[xi_Kep_4p != -1])
xi_Kep_5p = ssk_per_sys['xi_obs'][ssk_per_sys['Mtot_obs'] >= 5]
logxi_Kep_5p = np.log10(xi_Kep_5p[xi_Kep_5p != -1])





##### To plot the simulated and Kepler catalogs as marginal distributions:

subdirectory = '' #'Paper_Figures/'; 'Talk_Figures/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 1 #linewidth
#alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 12 #legend labels font size

#'''
# Multiplicities:
plot_fig_counts_hist_simple(fig_size, [sss_per_sys['Mtot_obs'] for sss_per_sys in model_sss_per_sys], [ssk_per_sys['Mtot_obs']], x_min=0, y_min=1e-2, y_max=1e4, x_llim=0.5, N_sim_Kep_factor=float(N_sim)/N_Kep, log_y=True, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text='Observed planets per system', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_sim=True, show_counts_Kep=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_multiplicities_compare.pdf', save_fig=savefigures)

# Periods:
plot_fig_pdf_simple(fig_size, [sss['P_obs'] for sss in model_sss], [ssk['P_obs']], x_min=3., x_max=300., y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, c_sim=model_colors, log_y=True, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_periods_compare.pdf', save_fig=savefigures)

# Period ratios (all, with some upper cut-off):
R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
plot_fig_pdf_simple(fig_size, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut] for sss in model_sss], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_periodratios_compare.pdf', save_fig=savefigures)

# Period ratios (< 5):
plot_fig_pdf_simple(fig_size, [sss['Rm_obs'][sss['Rm_obs'] < 5.] for sss in model_sss], [ssk['Rm_obs'][ssk['Rm_obs'] < 5.]], x_min=1., x_max=5., n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_periodratios_less5_compare.pdf', save_fig=savefigures)

# Transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_obs'] for sss in model_sss], [ssk['tdur_obs']], x_min=0., x_max=15., n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_durations_compare.pdf', save_fig=savefigures)

# Circular normalized transit durations:
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_obs'] for sss in model_sss], [ssk['tdur_tcirc_obs']], x_min=0., x_max=1.5, n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_tdur_tcirc_compare.pdf', save_fig=savefigures)

# Transit depths:
plot_fig_pdf_simple(fig_size, [sss['D_obs'] for sss in model_sss], [ssk['D_obs']], x_min=10.**(-5.), x_max=10.**(-1.5), n_bins=n_bins, log_x=True, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_depths_compare.pdf', save_fig=savefigures)

# Planet radii:
plot_fig_pdf_simple(fig_size, [sss['radii_obs'] for sss in model_sss], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$R_p (R_\oplus)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_radii_compare.pdf', save_fig=savefigures)

# Stellar radii:
plot_fig_pdf_simple(fig_size, [sss['Rstar_obs'] for sss in model_sss], [ssk['Rstar_obs']], x_min=0.5, x_max=2.5, n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_stellar_radii_compare.pdf', save_fig=savefigures)

# Transit depth ratios:
plot_fig_pdf_simple(fig_size, [sss['D_ratio_obs'] for sss in model_sss], [ssk['D_ratio_obs']], x_min=10.**(-1.5), x_max=10.**(1.5), n_bins=n_bins, log_x=True, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$\delta_{i+1}/\delta_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_depthratios_compare.pdf', save_fig=savefigures)

# Log(xi):
plot_fig_pdf_simple(fig_size, [np.log10(sss['xi_obs']) for sss in model_sss], [np.log10(ssk['xi_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_logxi_all_compare.pdf', save_fig=savefigures)

plt.show()
plt.close()
#'''





##### To plot the inner vs. outer period ratios of triplets (in 3+ systems) (similar to Fig 6 in Zhu et al. 2019 and Fig 7 in Weiss et al. 2018a):
'''
compute_pratio_in_out_and_plot_fig_pdf([sss_per_sys['P_obs'] for sss_per_sys in model_sss_per_sys], last_is_Kep=True, fig_size=(8,6), n_bins=50, x_min=0.1, x_max=10., colors=['b','r'], ls=['-',':'], lw=2, labels=['Clustered P+R', 'Non-clustered'], afs=afs, tfs=tfs, lfs=lfs, save_name=savefigures_directory + subdirectory + save_name + '_observed_pratio_out_in_ratio.pdf', save_fig=savefigures)
plt.show()
plt.close()
'''





##### To plot the circular normalized transit durations again (observed singles vs. multis):
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_1_obs'] for sss in model_sss], [ssk['tdur_tcirc_1_obs']], x_min=0., x_max=1.5, n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, extra_text='Singles', fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_tdur_tcirc_singles_compare.pdf', save_fig=savefigures)
plot_fig_pdf_simple(fig_size, [sss['tdur_tcirc_2p_obs'] for sss in model_sss], [ssk['tdur_tcirc_2p_obs']], x_min=0., x_max=1.5, n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', afs=afs, tfs=tfs, lfs=lfs, extra_text='Multis', fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_tdur_tcirc_multis_compare.pdf', save_fig=savefigures)




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

c2, c3, c4p = 'r', 'b', 'g'
ymax = 0.14

xi_bins = np.linspace(-0.5, 0.5, n_bins+1)

fig = plt.figure(figsize=(8,14))
plot = GridSpec(7,1,left=0.2,bottom=0.07,right=0.95,top=0.98,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0])
plot_panel_cdf_simple(ax, [logxi_2_model1, logxi_3_model1, logxi_4p_model1], [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], x_min=np.min(xi_bins), x_max=np.max(xi_bins), c_sim=[c2,c3,c4p], c_Kep=[c2,c3,c4p], ls_sim=['-','-','-'], ls_Kep=[':',':',':'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=['Kepler data',None,None], xlabel_text='', legend=True, afs=afs, tfs=tfs, lfs=lfs, label_dist=False)

ax = plt.subplot(plot[1:3,0])
plot_panel_pdf_simple(ax, [logxi_2_model1, logxi_3_model1, logxi_4p_model1], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, c_sim=[c2,c3,c4p], ls_sim=['-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s=model_names[0], ha='right', fontsize=lfs, transform=ax.transAxes)

ax = plt.subplot(plot[3:5,0])
plot_panel_pdf_simple(ax, [logxi_2_model2, logxi_3_model2, logxi_4p_model2], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, c_sim=[c2,c3,c4p], ls_sim=['-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], xlabel_text='', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s=model_names[1], ha='right', fontsize=lfs, transform=ax.transAxes)

ax = plt.subplot(plot[5:,0])
plot_panel_pdf_simple(ax, [logxi_Kep_2, logxi_Kep_3, logxi_Kep_4p], [], x_min=np.min(xi_bins), x_max=np.max(xi_bins), y_max=ymax, n_bins=n_bins, c_sim=[c2,c3,c4p], ls_sim=['-','-','-'], lw=2, labels_sim=[r'$m=2$', r'$m=3$', r'$m=4+$'], labels_Kep=[None], xlabel_text=r'$\log{\xi}$', legend=False, afs=afs, tfs=tfs, lfs=lfs)
plt.text(x=0.98, y=0.9, s='Kepler data', ha='right', fontsize=lfs, transform=ax.transAxes)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_logxi_per_mult.pdf')
    plt.close()




##### To remake the log(xi) plot for defense talk:

plot_fig_pdf_simple((8,4), [np.log10(sss['xi_obs']) for sss in model_sss], [], x_min=-0.5, x_max=0.5, n_bins=n_bins, c_sim=model_colors, ls_sim=model_linestyles, lw=3, labels_sim=model_names, xlabel_text=r'$\log{\xi}$', afs=afs, tfs=tfs, lfs=20, legend=True, fig_lbrt=[0.15, 0.2, 0.95, 0.925], save_name=savefigures_directory + subdirectory + save_name + '_logxi_incl.pdf', save_fig=savefigures)
plt.show()
