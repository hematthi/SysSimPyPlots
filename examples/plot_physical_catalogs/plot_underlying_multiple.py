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

savefigures_directory = '' #'/Users/hematthi/Documents/GradSchool/Research/My_Papers/He_Ford_Ragozzine_Clusters_AMD/Figures/Compare_models/'
save_name = 'Models_Compare'





##### To load the underlying populations:

# Model 1:
loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number1 = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory1 + 'periods%s.out' % run_number1)

param_vals_all1 = read_sim_params(loadfiles_directory1 + 'periods%s.out' % run_number1)
sssp_per_sys1, sssp1 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory1, run_number=run_number1, load_full_tables=True)

# Model 2:
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Extrapolate_P1000d/f_amd_crit_2/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
run_number2 = ''

param_vals_all2 = read_sim_params(loadfiles_directory2 + 'periods%s.out' % run_number2)
sssp_per_sys2, sssp2 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory2, run_number=run_number2, load_full_tables=True)

# Model 3:
loadfiles_directory3 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Extrapolate_P1000d/AMD_in_out/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/Extrapolate_P1000d/lambdac5/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/f_amd_crit_all/Params12_KS/durations_norm_circ_singles_multis_GF2020_KS/GP_med/f_amd_crit_2/'
run_number3 = ''

param_vals_all3 = read_sim_params(loadfiles_directory3 + 'periods%s.out' % run_number3)
sssp_per_sys3, sssp3 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory3, run_number=run_number3, load_full_tables=True)



model_sssp = [sssp1, sssp2, sssp3]
model_sssp_per_sys = [sssp_per_sys1, sssp_per_sys2, sssp_per_sys3]
model_names = ['Max AMD model', 'Max AMD model extrapolated', 'Max AMD model in-out'] #[r'$f_{\rm crit} = 0.5$', r'$f_{\rm crit} = 1$', r'$f_{\rm crit} = 2$'] #['Maximum AMD model', 'Two-Rayleigh model'] # Make sure this matches the models loaded!
model_linestyles = ['-', '-', '-'] #['-', '--', '--']
model_colors = ['purple','g','gray'] #['b', 'g', 'r']





##### To plot the simulated catalog as marginal distributions:

subdirectory = ''

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 2 #linewidth

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

#'''
# Multiplicities:
#plot_fig_counts_hist_simple(fig_size, [np.concatenate((sssp_per_sys['Mtot_all'], np.zeros(N_sim-len(sssp_per_sys['Mtot_all']), dtype=int))) for sssp_per_sys in model_sssp_per_sys], [], x_min=-1, x_llim=-0.5, x_ulim=10.5, normalize=True, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text='Planets per system', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_sim=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_multiplicities.pdf', save_fig=savefigures) # if sssp_per_sys['Mtot_all'] does not contain zeros
plot_fig_counts_hist_simple(fig_size, [sssp_per_sys['Mtot_all'] for sssp_per_sys in model_sssp_per_sys], [], x_min=-1, x_llim=-0.5, x_ulim=10.5, normalize=True, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text='Planets per system', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_sim=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_multiplicities.pdf', save_fig=savefigures) # if sssp_per_sys['Mtot_all'] contains zeros

# Clusters per system:
plot_fig_counts_hist_simple(fig_size, [sssp['clustertot_all'] for sssp in model_sssp], [], x_llim=0.5, x_ulim=5.5, normalize=True, lw=lw, c_sim=model_colors, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'Clusters per system $N_c$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, show_counts_sim=False, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_clusters_per_system.pdf', save_fig=savefigures)
'''
for n in [1,2,3]:
    for i,sssp in enumerate(model_sssp):
        x = np.sum(sssp['clustertot_all'] == n)/float(len(sssp['clustertot_all']))
        plt.text(n, x*(0.1)**(i+1), '{:0.2f}'.format(x) if x>0.01 else '<0.01', ha='center', color=model_colors[i], fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_clusters_per_system.pdf')
    plt.close()
'''

# Planets per cluster:
plot_fig_counts_hist_simple(fig_size, [sssp['pl_per_cluster_all'] for sssp in model_sssp], [], x_llim=0.5, x_ulim=7.5, normalize=True, lw=lw, c_sim=model_colors, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'Planets per cluster $N_p$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_planets_per_cluster.pdf', save_fig=savefigures)

# Periods:
#plot_fig_pdf_simple(fig_size, [sssp['P_all'] for sssp in model_sssp], [], x_min=P_min, x_max=P_max, n_bins=n_bins, log_x=True, log_y=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_periods.pdf', save_fig=savefigures)
plot_fig_pdf_simple(fig_size, [sssp['P_all'] for sssp in model_sssp], [], x_min=P_min, x_max=1000., n_bins=n_bins, normalize=False, log_x=True, log_y=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_periods.pdf', save_fig=savefigures) #####

# Period ratios (all):
plot_fig_pdf_simple(fig_size, [sssp['Rm_all'] for sssp in model_sssp], [], x_min=1., x_max=20., n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_periodratios.pdf', save_fig=savefigures)
#plt.minorticks_off()

# Period ratios (< 5):
plot_fig_pdf_simple(fig_size, [sssp['Rm_all'][sssp['Rm_all'] < 5] for sssp in model_sssp], [], x_min=1., x_max=5., n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1,2,3,4,5], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_periodratios_less5.pdf', save_fig=savefigures)

# Eccentricities:
x_min, x_max = 1e-3, 1.
#plot_fig_pdf_simple(fig_size, [sssp['e_all'] for sssp in model_sssp], [], x_min=x_min, x_max=x_max, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1e-3,1e-2,1e-1,1.], xlabel_text=r'$e$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plot_fig_pdf_simple(fig_size, [sssp2['e_all']], [], x_min=x_min, x_max=x_max, n_bins=n_bins, log_x=True, c_sim=[model_colors[1]], lw=lw, ls_sim=[model_linestyles[1]], labels_sim=[''], xticks_custom=[1e-3,1e-2,1e-1,1.], xlabel_text=r'$e$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins+1)
e1 = sssp_per_sys1['e_all'][sssp_per_sys1['Mtot_all'] == 1, 0]
e2p = sssp_per_sys1['e_all'][sssp_per_sys1['Mtot_all'] > 1]
e2p = e2p[sssp_per_sys1['P_all'][sssp_per_sys1['Mtot_all'] > 1] > 0]
plt.hist(e1, bins=bins, histtype='step', weights=np.ones(len(e1))/len(sssp1['e_all']), color=model_colors[0], ls='--', lw=lw, label='Singles')
plt.hist(e2p, bins=bins, histtype='step', weights=np.ones(len(e2p))/len(sssp1['e_all']), color=model_colors[0], ls=model_linestyles[0], lw=lw, label='Multis')
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_eccentricities.pdf')
    plt.close()

# Mutual inclinations:
plot_fig_pdf_simple(fig_size, [sssp['inclmut_all']*(180./np.pi) for sssp in model_sssp], [], x_min=1e-2, x_max=90., n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1e-2,1e-1,1.,10.,1e2], xlabel_text=r'$i_m$ (deg)', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_mutualinclinations.pdf', save_fig=savefigures)

# Planet masses:
plot_fig_pdf_simple(fig_size, [sssp['mass_all'] for sssp in model_sssp], [], x_min=0.09, x_max=1e2, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$M_p$ ($M_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_masses.pdf', save_fig=savefigures)

# Planet radii:
plot_fig_pdf_simple(fig_size, [sssp['radii_all'] for sssp in model_sssp], [], x_min=radii_min, x_max=radii_max, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[0.5,1,2,4,10], xlabel_text=r'$R_p$ ($R_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_radii.pdf', save_fig=savefigures)

# Planet radii ratios:
plot_fig_pdf_simple(fig_size, [sssp['radii_ratio_all'] for sssp in model_sssp], [], x_min=0.1, x_max=10., n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$R_{p,i+1}/R_{p,i}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_radii_ratios.pdf', save_fig=savefigures)

# Separations in mutual Hill radii:
plot_fig_pdf_simple(fig_size, [sssp['N_mH_all'] for sssp in model_sssp], [], x_min=1., x_max=200., n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\Delta$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_deltas.pdf', save_fig=savefigures)

# Stellar radii:
plot_fig_pdf_simple(fig_size, [sssp['Rstar_all'] for sssp in model_sssp], [], x_min=0.5, x_max=2.5, n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_stellar_radii.pdf', save_fig=savefigures)

### GF2020 metrics, but for the underlying systems:
# Dynamical masses CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['dynamical_mass'] for sssp_per_sys in model_sssp_per_sys], [], x_min=2e-7, x_max=1e-3, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\mu$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_dynamical_masses.pdf', save_fig=savefigures)

# Planet radii partitioning CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['radii_partitioning'] for sssp_per_sys in model_sssp_per_sys], [], x_min=1e-5, x_max=1., n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_radii_partitioning.pdf', save_fig=savefigures)

# Planet radii monotonicity CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['radii_monotonicity'] for sssp_per_sys in model_sssp_per_sys], [], x_min=-0.6, x_max=0.6, n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_radii_monotonicity.pdf', save_fig=savefigures)

# Gap complexity CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['gap_complexity'] for sssp_per_sys in model_sssp_per_sys], [], x_min=0., x_max=1., n_bins=n_bins, log_x=False, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt, save_name=savefigures_directory + subdirectory + save_name + '_underlying_gap_complexity.pdf', save_fig=savefigures)

plt.show()
plt.close()
#'''





##### To plot the intrinsic multiplicity, clusters per system, and planets per cluster distributions for multiple models in the same figure:
'''
fig = plt.figure(figsize=(8,8))
plot = GridSpec(3,1,left=0.15,bottom=0.1,right=0.95,top=0.98,wspace=0,hspace=0.4)

ax = plt.subplot(plot[0,0]) # intrinsic multiplicities
plot_panel_counts_hist_simple(ax, [np.concatenate((sssp_per_sys['Mtot_all'], np.zeros(N_sim-len(sssp_per_sys['Mtot_all']), dtype=int))) for sssp_per_sys in model_sssp_per_sys], [], x_min=0, x_ulim=10.5, normalize=True, c_sim=model_colors, ls_sim=model_linestyles, lw=lw, labels_sim=model_names, xlabel_text='Intrinsic planet multiplicity', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_sim=False)

ax = plt.subplot(plot[1,0]) # clusters per system
Nc_max = np.max([np.max(sssp['clustertot_all']) for sssp in model_sssp])
plot_panel_pdf_simple(ax, [sssp['clustertot_all'] for sssp in model_sssp], [], x_min=0.5, x_max=Nc_max+0.5, n_bins=Nc_max, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'Clusters per system $N_c$', afs=afs, tfs=tfs, lfs=lfs)
plt.xlim([0.5,6.5])
#for n in [1,2,3]:
#    for i,sssp in enumerate(model_sssp):
#        x = np.sum(sssp['clustertot_all'] == n)/float(len(sssp['clustertot_all']))
#        plt.text(n, x*(0.1)**(i+1), '{:0.2f}'.format(x) if x>0.01 else '<0.01', ha='center', color=model_colors[i], fontsize=lfs)

ax = plt.subplot(plot[2,0]) # planets per cluster
Np_max = np.max([np.max(sssp['pl_per_cluster_all']) for sssp in model_sssp])
plot_panel_pdf_simple(ax, [sssp['pl_per_cluster_all'] for sssp in model_sssp], [], x_min=0.5, x_max=Np_max+0.5, n_bins=Np_max, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'Planets per cluster $N_p$', afs=afs, tfs=tfs, lfs=lfs)
plt.xlim([0.5,8.5])

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_planet_cluster_multiplicities.pdf')
    plt.close()
'''





'''
# Model 1:
loadfiles_directory1 = 'ACI/Simulated_Data/Julia_v0.7/Kepler_catalog_optimization/q1q17_dr25_gaia_fgk_stars80006/Clustered_P_R/f_high_incl_low_incl_mmr/Fit_rate_mult_P_Pratios_D_Dratios_dur_durratios_mmr/Some11_params_KSweightedrms/lc_lp_0p5_5_alphaP_-2_1_alphaR1_R2_-6_0_ecc_0_0p1_incl_inclmmr_0_90_sigmaR_0_0p5_sigmaP_0_0p3/Fixed_Rbreak3_Ncrit8/targs400030_maxincl0_maxiters5000/sigma_i_greater_sigma_i_mmr/best_N/' #'ExoplanetsSysSim_Clusters/clusters_v0.7/'

n1_1plus = np.zeros(10)
ptot1 = np.zeros(10)
for i in range(10):
    run_number1 = str(i)

    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory1 + 'periods%s.out' % run_number1)

    param_vals_all1 = read_sim_params(loadfiles_directory1 + 'periods%s.out' % run_number1)
    sssp_per_sys1, sssp1 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory1, run_number=run_number1)

    n1_1plus[i] = len(sssp_per_sys1['Mtot_all'])
    ptot1[i] = np.sum(sssp_per_sys1['Mtot_all'])
n1_none = N_sim - n1_1plus
f1_none = n1_none/N_sim
fp1 = ptot1/float(N_sim)
print 'Clustered_P_R'
print 'Number of stars with no planets: ', n1_none
print 'Fraction of stars with no planets: ', f1_none
print 'Total number of planets: ', ptot1
print 'Ratio of planets to stars: ', fp1

# Model 2:
loadfiles_directory2 = 'ACI/Simulated_Data/Julia_v0.7/Kepler_catalog_optimization/q1q17_dr25_gaia_fgk_stars80006/Clustered_P/f_high_incl_low_incl_mmr/Fit_rate_mult_P_Pratios_D_Dratios_dur_durratios_mmr/Some10_params_KSweightedrms/Fixed_Rbreak3_Ncrit8/lc_lp_0p5_5_alphaP_-2_1_alphaR1_R2_-6_0_ecc_0_0p1_incl_inclmmr_0_90_sigmaP_0_0p3/targs400030_maxincl0_maxiters5000/sigma_i_greater_sigma_i_mmr/best_N/'

n2_1plus = np.zeros(10)
ptot2 = np.zeros(10)
for i in range(10):
    run_number2 = str(i)

    param_vals_all2 = read_sim_params(loadfiles_directory2 + 'periods%s.out' % run_number2)
    sssp_per_sys2, sssp2 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory2, run_number=run_number2)

    n2_1plus[i] = len(sssp_per_sys2['Mtot_all'])
    ptot2[i] = np.sum(sssp_per_sys2['Mtot_all'])
n2_none = N_sim - n2_1plus
f2_none = n2_none/N_sim
fp2 = ptot2/float(N_sim)
print 'Clustered_P'
print 'Number of stars with no planets: ', n2_none
print 'Fraction of stars with no planets: ', f2_none
print 'Total number of planets: ', ptot2
print 'Ratio of planets to stars: ', fp2

# Model 3:
loadfiles_directory3 = 'ACI/Simulated_Data/Julia_v0.7/Kepler_catalog_optimization/q1q17_dr25_gaia_fgk_stars80006/Non_Clustered/f_high_incl_low_incl_mmr/Fit_rate_mult_P_Pratios_D_Dratios_dur_durratios_mmr/Some8_params_KSweightedrms/Fixed_Rbreak3_Ncrit8/lc_1_10_alphaP_-2_1_alphaR1_R2_-6_0_ecc_0_0p1_incl_inclmmr_0_90/targs400030_maxincl0_maxiters5000/sigma_i_greater_sigma_i_mmr/best_N/'

n3_1plus = np.zeros(10)
ptot3 = np.zeros(10)
for i in range(10):
    run_number3 = str(i)

    param_vals_all3 = read_sim_params(loadfiles_directory3 + 'periods%s.out' % run_number3)
    sssp_per_sys3, sssp3 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory3, run_number=run_number3)

    n3_1plus[i] = len(sssp_per_sys3['Mtot_all'])
    ptot3[i] = np.sum(sssp_per_sys3['Mtot_all'])
n3_none = N_sim - n3_1plus
f3_none = n3_none/N_sim
fp3 = ptot3/float(N_sim)
print 'Non_clustered:'
print 'Number of stars with no planets: ', n3_none
print 'Fraction of stars with no planets: ', f3_none
print 'Total number of planets: ', ptot3
print 'Ratio of planets to stars: ', fp3
'''





'''
##### To plot the inner vs. outer period ratios of triplets (in 3+ systems) (similar to Fig 6 in Zhu et al. 2019 and Fig 7 in Weiss et al. 2018a, BUT for the intrinsic systems):

# Just for Clustered_P_R model:
#p_per_sys_subset = sssp_per_sys1['P_all'][np.random.choice(np.arange(len(sssp_per_sys1['P_all'])), int(round(len(sssp_per_sys1['P_all'])/(N_sim/N_Kep))), replace=False)]
p_per_sys_subset = sssp_per_sys1['P_all'][np.random.choice(np.arange(len(sssp_per_sys1['P_all'])), 1000, replace=False)]
compute_pratio_in_out_and_plot_fig([p_per_sys_subset], colors=[model_colors[0]], labels=[model_names[0]], xymax=25., xyticks_custom=[1,2,3,4,5,10,20], afs=afs, tfs=tfs, lfs=lfs, save_name='Meeting_plots/July_9_2019/Clustered_P_R_intrinsic_pratio_in_out.pdf', save_fig=False)

# For all three models:
p_per_sys_subset1 = sssp_per_sys1['P_all'][np.random.choice(np.arange(len(sssp_per_sys1['P_all'])), 1000, replace=False)]
p_per_sys_subset2 = sssp_per_sys2['P_all'][np.random.choice(np.arange(len(sssp_per_sys2['P_all'])), 1000, replace=False)]
p_per_sys_subset3 = sssp_per_sys3['P_all'][np.random.choice(np.arange(len(sssp_per_sys3['P_all'])), 1000, replace=False)]
compute_pratio_in_out_and_plot_fig([p_per_sys_subset1, p_per_sys_subset2, p_per_sys_subset3], colors=model_colors, labels=model_names, xymax=25., xyticks_custom=[1,2,3,4,5,10,20], afs=afs, tfs=tfs, lfs=lfs, save_name='Meeting_plots/July_9_2019/Models_Compare_intrinsic_pratio_in_out.pdf', save_fig=False)

compute_pratio_in_out_and_plot_fig_pdf([sssp_per_sys1['P_all'], sssp_per_sys2['P_all'], sssp_per_sys3['P_all']], n_bins=100, x_min=0.1, x_max=10., colors=model_colors, labels=model_names, afs=afs, tfs=tfs, lfs=lfs, save_name='Meeting_plots/July_9_2019/Models_Compare_intrinsic_pratio_out_in_ratio.pdf', save_fig=False)
plt.show()
plt.close()
'''





##### To compute the fraction of all planets near an MMR:

#f_mmr = calc_f_near_pratios(sssp_per_sys1)





##### To plot galleries of a sample of intrinsic multi-planet systems:

#plot_figs_multis_underlying_gallery(sssp_per_sys1, sssp1, n_pl=8, fig_size=(16,8), panels_per_fig=4, N_sys_sample=400, N_sys_per_plot=50, plot_line_per=1, colorby='clusterid', tfs=20, save_name_base=savefigures_directory + subdirectory + save_name + '_underlying_multis', save_fig=False)

#plot_figs_multis_underlying_gallery(sssp_per_sys1, sssp1, n_pl=3, fig_size=(8,16), panels_per_fig=1, N_sys_sample=100, N_sys_per_plot=100, plot_line_per=10, colorby='clusterid', tfs=20, save_name_base=savefigures_directory + subdirectory + save_name + '_underlying_multis', save_fig=False)





##### To plot eccentricity vs mutual inclinations, with attached histograms:

persys_1d_1, perpl_1d_1 = convert_underlying_properties_per_planet_1d(sssp_per_sys1, sssp1)
persys_1d_2, perpl_1d_2 = convert_underlying_properties_per_planet_1d(sssp_per_sys2, sssp2)

ecc_min_max, incl_min_max = [3e-4, 1.], [1e-2, 180.]
bins_log_ecc = np.linspace(np.log10(ecc_min_max[0]), np.log10(ecc_min_max[1]), 101)
bins_log_incl = np.linspace(np.log10(incl_min_max[0]), np.log10(incl_min_max[1]), 101)

fig = plt.figure(figsize=(16,8))
plot = GridSpec(5, 9, left=0.1, bottom=0.1, right=0.975, top=0.975, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4]) # scatter i_m vs ecc (model 1)
corner.hist2d(np.log10(perpl_1d_1['e_all']), np.log10(perpl_1d_1['im_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([-3., -2., -1., 0.])
ytick_vals = np.array([-2., -1., 0., 1., 2.])
plt.xticks(xtick_vals, 10.**xtick_vals)
plt.yticks(ytick_vals, 10.**ytick_vals)
plt.xlim(np.log10(np.array(ecc_min_max)))
plt.ylim(np.log10(np.array(incl_min_max)))
plt.xlabel(r'$e$', fontsize=tfs)
plt.ylabel(r'$i_m$ (deg)', fontsize=tfs)
ax.text(x=0.95, y=0.05, s=model_names[0], ha='right', fontsize=tfs, color=model_colors[0], transform=ax.transAxes)

ax = plt.subplot(plot[0,:4]) # top histogram of ecc (model 1)
e1 = sssp_per_sys1['e_all'][sssp_per_sys1['Mtot_all'] == 1, 0]
plt.hist(np.log10(e1), bins=bins_log_ecc, histtype='step', color=model_colors[0], ls=':', lw=lw, label=r'$n = 1$')
plt.hist(np.log10(perpl_1d_1['e_all']), bins=bins_log_ecc, histtype='step', color=model_colors[0], ls=model_linestyles[0], lw=lw, label=r'$n \geq 2$')
plt.xticks(xtick_vals, [])
plt.yticks([])
plt.xlim(np.log10(np.array(ecc_min_max)))
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4:8]) # scatter i_m vs ecc (model 2)
corner.hist2d(np.log10(perpl_1d_2['e_all']), np.log10(perpl_1d_2['im_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([-3., -2., -1., 0.])
plt.xticks(xtick_vals, 10.**xtick_vals)
plt.yticks([])
plt.xlim(np.log10(np.array(ecc_min_max)))
plt.ylim(np.log10(np.array(incl_min_max)))
plt.xlabel(r'$e$', fontsize=tfs)
ax.text(x=0.95, y=0.05, s=model_names[1], ha='right', fontsize=tfs, color=model_colors[1], transform=ax.transAxes)

ax = plt.subplot(plot[0,4:8]) # top histogram of ecc (model 2)
plt.hist(np.log10(perpl_1d_2['e_all']), bins=bins_log_ecc, histtype='step', color=model_colors[1], ls=model_linestyles[1], lw=lw)
plt.xticks(xtick_vals, [])
plt.yticks([])
plt.xlim(np.log10(np.array(ecc_min_max)))

ax = plt.subplot(plot[1:,8]) # side histogram of i_m (model 1+2)
plt.hist(np.log10(perpl_1d_1['im_all']), bins=bins_log_incl, histtype='step', orientation='horizontal', color=model_colors[0], ls=model_linestyles[0], lw=lw)
plt.hist(np.log10(perpl_1d_2['im_all']), bins=bins_log_incl, histtype='step', orientation='horizontal', color=model_colors[1], ls=model_linestyles[1], lw=lw)
plt.xticks([])
plt.yticks(ytick_vals, [])
plt.ylim(np.log10(np.array(incl_min_max)))

plt.show()





##### For Sarah's TDV paper:

x_min, x_max = 1e-3, 1.
plot_fig_pdf_simple((8,4), [sssp['e_all'] for sssp in model_sssp], [], x_min=x_min, x_max=x_max, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=1, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1e-3,1e-2,1e-1,1.], xlabel_text=r'$e$', afs=16, tfs=16, lfs=16, fig_lbrt=[0.15,0.15,0.95,0.95])
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'f_amd_crit_underlying_eccentricities.pdf')
    plt.close()

plot_fig_pdf_simple((8,4), [sssp['inclmut_all']*(180./np.pi) for sssp in model_sssp], [], x_min=1e-2, x_max=90., n_bins=n_bins, log_x=True, c_sim=model_colors, lw=1, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1e-2,1e-1,1.,10.,1e2], xlabel_text=r'$i$ [deg]', afs=16, tfs=16, lfs=16, fig_lbrt=[0.15,0.15,0.95,0.95])
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + 'f_amd_crit_underlying_mutualinclinations.pdf')
    plt.close()

plt.show()
