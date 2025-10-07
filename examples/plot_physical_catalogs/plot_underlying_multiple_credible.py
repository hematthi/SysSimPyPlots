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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Underlying/'
save_name = 'Models_compare' #'Hybrid_vs_H20_models'





##### To load the underlying populations:

# Model 1:
loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
run_number1 = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory1 + 'periods%s.out' % run_number1)

param_vals_all1 = read_sim_params(loadfiles_directory1 + 'periods%s.out' % run_number1)
sssp_per_sys1, sssp1 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory1, run_number=run_number1)

# Model 2:
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_equal/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
run_number2 = ''

param_vals_all2 = read_sim_params(loadfiles_directory2 + 'periods%s.out' % run_number2)
sssp_per_sys2, sssp2 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory2, run_number=run_number2)



model_sssp = [sssp1, sssp2]
model_sssp_per_sys = [sssp_per_sys1, sssp_per_sys2]

model_names = ['HM-C', 'HM-U', 'H20 model'] #['Hybrid model', 'H20 model']
model_linestyles = ['--', '--', '--']
model_colors = ['g', 'b', 'k']
model_stagger_errorbars = [0., -0.1, 0.1] # offsets for plotting multiplicity counts in order to stagger errorbars





##### To plot the simulated catalog as marginal distributions:

subdirectory = 'Compare_to_hybrid_nonclustered_and_H20/' #'Compare_to_H20_model/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 2
alpha = 0.2
alpha_all = [alpha]

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/'
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8p1_KS/Params9_fix_highM/GP_best_models_100/'
loadfiles_directory3 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
#loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_all_KS/Params8/GP_best_models_100/'
#loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_equal/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'

model_loadfiles_dirs = [loadfiles_directory1, loadfiles_directory2, loadfiles_directory3]
models = len(model_loadfiles_dirs)

runs = 100

sssp_all = []
sssp_per_sys_all = []
params_all = []

Mtot_bins = np.arange(23)-1.5 #include a -1 bin with all zeros for plotting purposes
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = []
Mtot_earth_counts_all = []

clustertot_bins = np.arange(12)-0.5 #includes 0 bin but will not count them
clustertot_bins_mid = (clustertot_bins[:-1] + clustertot_bins[1:])/2.
clustertot_counts_all = []

pl_per_cluster_bins = np.arange(12)-0.5 #includes 0 bin but will not count them
pl_per_cluster_bins_mid = (pl_per_cluster_bins[:-1] + pl_per_cluster_bins[1:])/2.
pl_per_cluster_counts_all = []

# To also store median e and im per multiplicity for power-law fitting:
n_array = np.arange(2,11)
log_n = np.log10(n_array)
e_med_1_all = []
e_med_n_all = []
im_med_n_all = []

for loadfiles_dir in model_loadfiles_dirs:
    sssp_dir = []
    sssp_per_sys_dir = []
    params_dir = []
    
    Mtot_counts = []
    Mtot_earth_counts = []
    clustertot_counts = []
    pl_per_cluster_counts = []
    
    e_med_1 = np.zeros(runs)
    e_med_n = np.zeros((runs, len(n_array)))
    im_med_n = np.zeros((runs, len(n_array)))
    
    for i in range(runs): #range(1,runs+1)
        run_number = i+1
        print(i)
        N_sim_i = read_targets_period_radius_bounds(loadfiles_dir + 'periods%s.out' % run_number)[0]
        params_i = read_sim_params(loadfiles_dir + 'periods%s.out' % run_number)
        sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_dir, run_number=run_number)
        
        # Catalogs and parameters:
        sssp_dir.append(sssp_i)
        sssp_per_sys_dir.append(sssp_per_sys_i)
        params_dir.append(params_i)
        
        # Multiplicities:
        counts, bins = np.histogram(sssp_per_sys_i['Mtot_all'], bins=Mtot_bins)
        counts[1] = N_sim_i - len(sssp_per_sys_i['Mtot_all'])
        Mtot_counts.append(counts/float(np.sum(counts)))
        
        # Multiplicities for Earth-sized planets:
        Earth_bools_per_sys = (sssp_per_sys_i['radii_all'] > 0.75) & (sssp_per_sys_i['radii_all'] < 1.25)
        Earth_counts_per_sys = np.sum(Earth_bools_per_sys, axis=1)
        counts, bins = np.histogram(Earth_counts_per_sys, bins=Mtot_bins)
        counts[1] = N_sim_i - len(Earth_counts_per_sys)
        Mtot_earth_counts.append(counts/float(np.sum(counts)))
        
        # Numbers of clusters:
        counts, bins = np.histogram(sssp_i['clustertot_all'], bins=clustertot_bins)
        clustertot_counts.append(counts/float(np.sum(counts)))
        
        # Numbers of planets per cluster:
        counts, bins = np.histogram(sssp_i['pl_per_cluster_all'], bins=pl_per_cluster_bins)
        pl_per_cluster_counts.append(counts/float(np.sum(counts)))
        
        # Median eccentricity and mutual inclination per multiplicity:
        e_1 = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == 1,0]
        e_med_1[i] = np.median(e_1)
        for j,n in enumerate(n_array):
            e_n = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
            e_n = e_n.flatten()
            im_n = sssp_per_sys_i['inclmut_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
            im_n = im_n.flatten() * (180./np.pi)
            
            e_med_n[i,j] = np.median(e_n)
            im_med_n[i,j] = np.median(im_n)
    
    sssp_all.append(sssp_dir)
    sssp_per_sys_all.append(sssp_per_sys_dir)
    params_all.append(params_dir)
    
    Mtot_counts_all.append(np.array(Mtot_counts))
    Mtot_earth_counts_all.append(np.array(Mtot_earth_counts))
    clustertot_counts_all.append(np.array(clustertot_counts))
    pl_per_cluster_counts_all.append(np.array(pl_per_cluster_counts))
    
    e_med_1_all.append(e_med_1)
    e_med_n_all.append(e_med_n)
    im_med_n_all.append(im_med_n)

Mtot_counts_all = np.array(Mtot_counts_all)
Mtot_earth_counts_all = np.array(Mtot_earth_counts_all)
clustertot_counts_all = np.array(clustertot_counts_all)
pl_per_cluster_counts_all = np.array(pl_per_cluster_counts_all)

Mtot_counts_qtls = [np.zeros((len(Mtot_bins_mid),3)) for m in range(models)]
clustertot_counts_qtls = [np.zeros((len(clustertot_bins_mid),3)) for m in range(models)]
pl_per_cluster_counts_qtls = [np.zeros((len(pl_per_cluster_bins_mid),3)) for m in range(models)]
for m in range(models):
    for b in range(len(Mtot_bins_mid)):
        counts_bin_sorted = np.sort(Mtot_counts_all[m][:,b])
        Mtot_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
    for b in range(len(clustertot_bins_mid)):
        counts_bin_sorted = np.sort(clustertot_counts_all[m][:,b])
        clustertot_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
    for b in range(len(pl_per_cluster_bins_mid)):
        counts_bin_sorted = np.sort(pl_per_cluster_counts_all[m][:,b])
        pl_per_cluster_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
#####





#'''
# Multiplicities:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(Mtot_bins_mid, Mtot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(Mtot_bins_mid + model_stagger_errorbars[m], Mtot_counts_qtls[m][:,1], yerr=[Mtot_counts_qtls[m][:,1]-Mtot_counts_qtls[m][:,0], Mtot_counts_qtls[m][:,2]-Mtot_counts_qtls[m][:,1]], fmt='none', color=model_colors[m], lw=lw, label='') #capsize=5 #label=r'16% and 84%' if m==0 else ''
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([-0.5, 10.5]) #[0, np.max(x)]
plt.ylim([0., 0.4])
plt.xlabel('Intrinsic planet multiplicity', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_multiplicities.pdf')
    plt.close()

# Number of clusters:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(clustertot_bins_mid, clustertot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(clustertot_bins_mid + model_stagger_errorbars[m], clustertot_counts_qtls[m][:,1], yerr=[clustertot_counts_qtls[m][:,1]-clustertot_counts_qtls[m][:,0], clustertot_counts_qtls[m][:,2]-clustertot_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 5.5])
plt.ylim([0., 1.])
plt.xlabel(r'Clusters per system, $N_c$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_clusters.pdf')
    plt.close()

# Number of planets per cluster:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(pl_per_cluster_bins_mid, pl_per_cluster_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(pl_per_cluster_bins_mid + model_stagger_errorbars[m], pl_per_cluster_counts_qtls[m][:,1], yerr=[pl_per_cluster_counts_qtls[m][:,1]-pl_per_cluster_counts_qtls[m][:,0], pl_per_cluster_counts_qtls[m][:,2]-pl_per_cluster_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 7.5])
plt.ylim([0., 0.7])
plt.xlabel(r'Planets per cluster, $N_p$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_planets_per_cluster.pdf')
    plt.close()

# Periods:
plot_fig_pdf_credible([[sssp_i['P_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=P_min, x_max=P_max, y_min=1e-3, n_bins=n_bins, step=None, plot_median=True, log_x=True, log_y=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'Period, $P$ [days]', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_periods.pdf')
    plt.close()

# Period ratios:
plot_fig_pdf_credible([[sssp_i['Rm_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=1., x_max=20., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'Period ratio, $P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_periodratios.pdf')
    plt.close()

# Eccentricities:
plot_fig_pdf_credible([[sssp_i['e_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=1e-3, x_max=1., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xticks_custom=[0.01, 0.1, 1.], xlabel_text=r'Eccentricity, $e$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_eccentricities.pdf')
    plt.close()

# Mutual inclinations:
plot_fig_pdf_credible([[sssp_i['inclmut_all']*(180./np.pi) for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=1e-2, x_max=45., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Mutual inclination, $i_m$ [deg]', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_mutualinclinations.pdf')
    plt.close()

# Planet masses:
plot_fig_pdf_credible([[sssp_i['mass_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=0.09, x_max=1e3, n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Planet mass, $M_p$ [$M_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_masses.pdf')
    plt.close()

# Planet radii (logged and unlogged versions):
plot_fig_pdf_credible([[sssp_i['radii_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=radii_min, x_max=radii_max, n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xticks_custom=[0.5,1,2,4,10], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii.pdf')
    plt.close()
plot_fig_pdf_credible([[sssp_i['radii_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=radii_min, x_max=4., n_bins=n_bins, step=None, plot_median=True, log_x=False, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xticks_custom=[1,2,3,4], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_unlogged.pdf')
    plt.close()

# Initial planet radii:
plot_fig_pdf_credible([[sssp_i['init_radii_all'] for sssp_i in sssp_list] for sssp_list in sssp_all[:2]], [], x_min=radii_min, x_max=4., n_bins=n_bins, step=None, plot_median=True, log_x=False, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xticks_custom=[1,2,3,4], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_initial_radii_unlogged.pdf')
    plt.close()

# Planet radii ratios:
plot_fig_pdf_credible([[sssp_i['radii_ratio_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=0.1, x_max=10., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Radius ratio, $R_{p,i+1}/R_{p,i}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_ratios.pdf')
    plt.close()

# Planet mass ratios:
plot_fig_pdf_credible([[sssp_i['mass_ratio_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=0.01, x_max=100., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Mass ratio, $M_{p,i+1}/M_{p,i}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_mass_ratios.pdf')
    plt.close()

# Separations in mutual Hill radii:
plot_fig_pdf_credible([[sssp_i['N_mH_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=1., x_max=200., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Minimum separation, $\Delta$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_deltas.pdf')
    plt.close()

# Dynamical masses:
plot_fig_pdf_credible([[sssp_per_sys_i['dynamical_mass'] for sssp_per_sys_i in sssp_per_sys_list] for sssp_per_sys_list in sssp_per_sys_all], [], x_min=2e-7, x_max=3e-3, n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Dynamical mass, $\mu$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_dynamical_masses.pdf')
    plt.close()

# Planet mass partitioning:
plot_fig_pdf_credible([[sssp_per_sys_i['mass_partitioning'] for sssp_per_sys_i in sssp_per_sys_list] for sssp_per_sys_list in sssp_per_sys_all], [], x_min=1e-4, x_max=1., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Mass partitioning, $\mathcal{Q}_M$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_mass_partitioning.pdf')
    plt.close()

# Planet radii partitioning:
plot_fig_pdf_credible([[sssp_per_sys_i['radii_partitioning'] for sssp_per_sys_i in sssp_per_sys_list] for sssp_per_sys_list in sssp_per_sys_all], [], x_min=1e-4, x_max=1., n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Radius partitioning, $\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_partitioning.pdf')
    plt.close()

# Planet radii monotonicity:
plot_fig_pdf_credible([[sssp_per_sys_i['radii_monotonicity'] for sssp_per_sys_i in sssp_per_sys_list] for sssp_per_sys_list in sssp_per_sys_all], [], x_min=-0.7, x_max=0.7, n_bins=n_bins, step=None, plot_median=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Radius monotonicity, $\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_monotonicity.pdf')
    plt.close()

# Gap complexity:
plot_fig_pdf_credible([[sssp_per_sys_i['gap_complexity'] for sssp_per_sys_i in sssp_per_sys_list] for sssp_per_sys_list in sssp_per_sys_all], [], x_min=0., x_max=1., n_bins=n_bins, step=None, plot_median=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xlabel_text=r'Gap complexity, $\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_gap_complexity.pdf')
    plt.close()

plt.show()
plt.close()
#'''





##### To plot the planet radii again, with the initial and final distributions (also log and unlogged versions):

init_radii_all = [[sssp_i['init_radii_all'] for sssp_i in sssp_list] for sssp_list in sssp_all[:2]] # list of lists (per model) of arrays (per catalog) of initial radii

# Logged version:
plot_fig_pdf_credible([[sssp_i['radii_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=radii_min, x_max=radii_max, n_bins=n_bins, step=None, plot_median=True, log_x=True, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xticks_custom=[0.5,1,2,4,10], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
bins = np.logspace(np.log10(radii_min), np.log10(radii_max), n_bins+1)
bins_mid = np.sqrt(bins[:-1] * bins[1:])
for i,x_sim in enumerate(init_radii_all):
    counts_all = []
    for xs in x_sim:
        counts, bins = np.histogram(xs, bins=bins)
        counts_all.append(counts/float(np.sum(counts)))
    counts_all = np.array(counts_all)
    counts_qtls = np.quantile(counts_all, [0.16,0.5,0.84], axis=0)
    plt.plot(bins_mid, counts_qtls[1], ls=':', color=model_colors[i], alpha=0.25)
    plt.fill_between(bins_mid, counts_qtls[0], counts_qtls[2], step=None, color=model_colors[i], alpha=0.05)
plt.ylim([0.,0.05])
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_with_initial.pdf')
    plt.close()

# Unlogged version:
plot_fig_pdf_credible([[sssp_i['radii_all'] for sssp_i in sssp_list] for sssp_list in sssp_all], [], x_min=radii_min, x_max=4., n_bins=n_bins, step=None, plot_median=True, log_x=False, c_sim_all=model_colors, lw=lw, alpha_all=alpha_all, labels_sim_all=model_names, xticks_custom=[1,2,3,4], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=fig_size, fig_lbrt=fig_lbrt)
bins = np.linspace(radii_min, 4., n_bins+1)
bins_mid = (bins[:-1] + bins[1:])/2.
for i,x_sim in enumerate(init_radii_all):
    counts_all = []
    for xs in x_sim:
        counts, bins = np.histogram(xs, bins=bins)
        counts_all.append(counts/float(np.sum(counts)))
    counts_all = np.array(counts_all)
    counts_qtls = np.quantile(counts_all, [0.16,0.5,0.84], axis=0)
    plt.plot(bins_mid, counts_qtls[1], ls=':', color=model_colors[i], alpha=0.25)
    plt.fill_between(bins_mid, counts_qtls[0], counts_qtls[2], step=None, color=model_colors[i], alpha=0.05)
plt.ylim([0.,0.05])
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_with_initial_unlogged.pdf')
    plt.close()

plt.show()





##### To compute some occurrence rates statistics:
'''
# Fraction of stars with no planets/at least 1 planet:
fswnp, fswnp_16, fswnp_84 = np.median(Mtot_counts_all[:,0]), Mtot_counts_16[0], Mtot_counts_84[0]
fswp, fswp_16, fswp_84 = 1.-fswnp, 1.-fswnp_84, 1.-fswnp_16

# Mean number of planets per star:
pl_tot_all = np.sum((Mtot_counts_all*N_sim_i)*np.arange(len(Mtot_bins_mid)), axis=1)
pl_per_star_all = np.sort(pl_tot_all/np.float(N_sim_i))
mps, mps_16, mps_84 = np.median(pl_per_star_all), pl_per_star_all[16], pl_per_star_all[84]

# Mean number of planets per planetary system (i.e. stars with at least 1 planet):
n_1plus_all = N_sim_i - (Mtot_counts_all*N_sim_i)[:,0]
pl_per_1plus_all = np.sort(pl_tot_all/n_1plus_all)
mpps, mpps_16, mpps_84 = np.median(pl_per_1plus_all), pl_per_1plus_all[16], pl_per_1plus_all[84]

# Fraction of stars with Earth-sized planets:
fswne, fswne_16, fswne_84 = np.median(Mtot_earth_counts_all[:,0]), np.sort(Mtot_earth_counts_all[:,0])[16], np.sort(Mtot_earth_counts_all[:,0])[84]
fswe, fswe_16, fswe_84 = 1.-fswne, 1.-fswne_84, 1.-fswne_16

# Mean number of Earth-sized planets per star:
ep_tot_all = np.sum((Mtot_earth_counts_all*N_sim_i)*np.arange(len(Mtot_bins_mid)), axis=1)
ep_per_star_all = np.sort(ep_tot_all/np.float(N_sim_i))
meps, meps_16, meps_84 = np.median(ep_per_star_all), ep_per_star_all[16], ep_per_star_all[84]

# Mean number of Earth-sized planets per planetary system (i.e. stars with at least 1 planet):
ep_per_1plus_all = np.sort(ep_tot_all/n_1plus_all)
mepps, mepps_16, mepps_84 = np.median(ep_per_1plus_all), ep_per_1plus_all[16], ep_per_1plus_all[84]

# Fraction of planetary systems with at least one Earth-sized planet (i.e. probability that a planet-hosting star contains an Earth-sized planet):
fswpwe_all = np.sort((1.-Mtot_earth_counts_all[:,0])/(1.-Mtot_counts_all[:,0]))
fswpwe, fswpwe_16, fswpwe_84 = np.median(fswpwe_all), fswpwe_all[16], fswpwe_all[84]
'''





##### To plot eccentricity vs mutual inclinations, with attached histograms:
'''
persys_1d_1, perpl_1d_1 = convert_underlying_properties_per_planet_1d(sssp_per_sys1, sssp1)
persys_1d_2, perpl_1d_2 = convert_underlying_properties_per_planet_1d(sssp_per_sys2, sssp2)

ecc_min_max, incl_min_max = [e_bins[0], e_bins[-1]], [im_bins[0], im_bins[-1]]

fig = plt.figure(figsize=(16,9))
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
plt.ylabel(r'$i_m$ ($^\circ$)', fontsize=tfs)
ax.text(x=0.95, y=0.05, s=model_names[0], ha='right', fontsize=tfs, color=model_colors[0], transform=ax.transAxes)

ax = plt.subplot(plot[0,:4]) # top histogram of ecc (model 1)
e1 = sssp_per_sys1['e_all'][sssp_per_sys1['Mtot_all'] == 1, 0]
#plt.hist(sssp1['e_all'], bins=e_bins, weights=np.ones(len(sssp1['e_all']))/len(sssp1['e_all']), histtype='step', color=model_colors[0], ls=model_linestyles[0], lw=lw)
#plt.fill_between(e_bins_mid, e_counts_qtls[0][:,0], e_counts_qtls[0][:,2], color=model_colors[0], alpha=alpha)
#'#''
plt.hist(e1, bins=e_bins, histtype='step', weights=np.ones(len(e1))/len(sssp1['e_all']), color=model_colors[0], ls=':', lw=lw, label=r'$n = 1$')
plt.hist(perpl_1d_1['e_all'], bins=e_bins, histtype='step', weights=np.ones(len(e2p))/len(sssp1['e_all']), color=model_colors[0], ls=model_linestyles[0], lw=lw, label=r'$n \geq 2$')
plt.fill_between(e_bins_mid, e1_counts_qtls[0][:,0], e1_counts_qtls[0][:,2], color=model_colors[0], alpha=alpha)
plt.fill_between(e_bins_mid, e2p_counts_qtls[0][:,0], e2p_counts_qtls[0][:,2], color=model_colors[0], alpha=alpha)
#'#''
plt.gca().set_xscale("log")
plt.xticks(10.**xtick_vals, [])
plt.yticks([])
plt.xlim(np.array(ecc_min_max))
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
plt.hist(sssp2['e_all'], bins=e_bins, weights=np.ones(len(sssp2['e_all']))/len(sssp2['e_all']), histtype='step', color=model_colors[1], ls=model_linestyles[1], lw=lw)
plt.fill_between(e_bins_mid, e_counts_qtls[1][:,0], e_counts_qtls[1][:,2], color=model_colors[1], alpha=alpha)
plt.gca().set_xscale("log")
plt.xticks(10.**xtick_vals, [])
plt.yticks([])
plt.xlim(np.array(ecc_min_max))

ax = plt.subplot(plot[1:,8]) # side histogram of i_m (model 1+2)
plt.hist(perpl_1d_1['im_all'], bins=im_bins, weights=np.ones(len(perpl_1d_1['im_all']))/len(perpl_1d_1['im_all']), histtype='step', orientation='horizontal', color=model_colors[0], ls=model_linestyles[0], lw=lw)
plt.hist(perpl_1d_2['im_all'], bins=im_bins, weights=np.ones(len(perpl_1d_2['im_all']))/len(perpl_1d_2['im_all']), histtype='step', orientation='horizontal', color=model_colors[1], ls=model_linestyles[1], lw=lw)
for m in range(models):
    plt.fill_betweenx(im_bins_mid, im_counts_qtls[m][:,0], im_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
plt.gca().set_yscale("log")
plt.xticks([])
plt.yticks(10.**ytick_vals, [])
plt.ylim(np.array(incl_min_max))

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_ecc_vs_incl.pdf')
    plt.close()
plt.show()
'''





##### To fit power-laws to the median eccentricities and mutual inclinations as a function of intrinsic multiplicity n:
##### NOTE: We are actually fitting a line to log(e) and log(im) vs log(n)
'''
n_norm = 5 # n for normalization constant
f_linear = lambda p, x: p[0] + p[1]*x - p[1]*np.log10(n_norm) # extra term for normalizing
f_err = lambda p, x, y: y - f_linear(p,x)
e_p0 = [0.03, -2.] # initial guesses for log10(mu_e) and alpha_e
im_p0 = [1., -2.] # initial guesses for log10(mu_im) and alpha_im

mu_e_all, alpha_e_all = [np.zeros(runs), np.zeros(runs)], [np.zeros(runs), np.zeros(runs)]
mu_im_all, alpha_im_all = [np.zeros(runs), np.zeros(runs)], [np.zeros(runs), np.zeros(runs)]
for m in range(models):
    for i in range(runs):
        log_e_med = np.log10(e_med_n_all[m][i])
        log_im_med = np.log10(im_med_n_all[m][i])

        e_fit = scipy.optimize.leastsq(f_err, e_p0, args=(log_n, log_e_med), full_output=1)
        e_logmu_alpha = e_fit[0]
        mu_e, alpha_e = 10.**(e_logmu_alpha[0]), e_logmu_alpha[1]
        mu_e_all[m][i], alpha_e_all[m][i] = mu_e, alpha_e

        im_fit = scipy.optimize.leastsq(f_err, im_p0, args=(log_n, log_im_med), full_output=1)
        im_logmu_alpha = im_fit[0]
        mu_im, alpha_im = 10.**(im_logmu_alpha[0]), im_logmu_alpha[1]
        mu_im_all[m][i], alpha_im_all[m][i] = mu_im, alpha_im
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
loadfiles_directory = '../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/f_amd_crit_all/Params12_KS/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
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
e_med_1_qtls = np.quantile(e_med_1_all[0], [0.16,0.5,0.84])
e_med_n_qtls = np.quantile(e_med_n_all[0], [0.16,0.5,0.84], axis=0)
e_plfit_n_qtls_model1 = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all[0], alpha_e_all[0], n_norm=n_norm)
e_plfit_n_qtls_model2 = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all[1], alpha_e_all[1], n_norm=n_norm)
plt.errorbar(n_array, e_med_n_qtls[1,:], yerr=[e_med_n_qtls[1,:]-e_med_n_qtls[0,:], e_med_n_qtls[2,:]-e_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='Maximum AMD model')
plt.errorbar([1], [e_med_1_qtls[1]], yerr=[[e_med_1_qtls[1]-e_med_1_qtls[0]], [e_med_1_qtls[2]-e_med_1_qtls[1]]], fmt='o', capsize=5, ls='none', color='c', label='')
plt.plot(n_array, e_med_n_fcrits[0,:], 'o', lw=lw, color='k', label=r'$f_{\rm crit} = 0.5$')
plt.plot(n_array, e_med_n_fcrits[1,:], 'o', lw=lw, color='r', label=r'$f_{\rm crit} = 2$')
plt.fill_between(np.arange(10)+1, e_plfit_n_qtls_model1[0,:], e_plfit_n_qtls_model1[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.031_{-0.003}^{+0.004}$, $\alpha_e = -1.74_{-0.07}^{+0.10}$')
plt.fill_between(np.arange(10)+1, e_plfit_n_qtls_model2[0,:], e_plfit_n_qtls_model2[2,:], color='g', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.040_{-0.002}^{+0.003}$, $\alpha_e = -1.54_{-0.11}^{+0.10}$')
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
im_med_n_qtls = np.quantile(im_med_n_all[0], [0.16,0.5,0.84], axis=0)
im_plfit_n_qtls_model1 = compute_power_law_at_n_quantiles(n_array, mu_im_all[0], alpha_im_all[0], n_norm=n_norm)
im_plfit_n_qtls_model2 = compute_power_law_at_n_quantiles(n_array, mu_im_all[1], alpha_im_all[1], n_norm=n_norm)
plt.errorbar(n_array, im_med_n_qtls[1,:], yerr=[im_med_n_qtls[1,:]-im_med_n_qtls[0,:], im_med_n_qtls[2,:]-im_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='')
plt.plot(n_array, im_med_n_fcrits[0,:], 'o', lw=lw, color='k', label='')
plt.plot(n_array, im_med_n_fcrits[1,:], 'o', lw=lw, color='r', label='')
plt.fill_between(n_array, im_plfit_n_qtls_model1[0,:], im_plfit_n_qtls_model1[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.10_{-0.11}^{+0.15}$, $\alpha_i = -1.73_{-0.08}^{+0.09}$')
plt.fill_between(n_array, im_plfit_n_qtls_model2[0,:], im_plfit_n_qtls_model2[2,:], color='g', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.41_{-0.08}^{+0.10}$, $\alpha_i = -1.55_{-0.11}^{+0.11}$')
plt.plot(n_array, incl_mult_power_law_Zhu2018(n_array)*np.sqrt(2.*np.log(2.)), 'x', ls='-', lw=2, ms=5, color='orange', label=r'$\sigma_{i,5}[^\circ] = 0.8$, $\alpha_i = -3.5$')
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
handles = handles[1:] + [handles[0]]
labels = labels[1:] + [labels[0]]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_ecc_incl_vs_mult_fits.pdf')
    plt.close()
plt.show()
'''





##### To remake some figures for proposals:
'''
directory = '/Users/hematthi/Documents/GradSchool/Postdoctoral_Applications/Figures/'

fig_size = (6,3) #size of each panel (figure)
fig_lbrt = [0.2, 0.3, 0.95, 0.925]

n_bins = 100
lw = 3 #linewidth
alpha = 0.2

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

# Multiplicities:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(Mtot_bins_mid, Mtot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(Mtot_bins_mid + model_stagger_errorbars[m], Mtot_counts_qtls[m][:,1], yerr=[Mtot_counts_qtls[m][:,1]-Mtot_counts_qtls[m][:,0], Mtot_counts_qtls[m][:,2]-Mtot_counts_qtls[m][:,1]], fmt='none', color=model_colors[m], lw=lw, label='') #capsize=5 #label=r'16% and 84%' if m==0 else ''
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([-0.5, 10.5]) #[0, np.max(x)]
plt.ylim([0., 0.6])
plt.xlabel('Intrinsic planet multiplicity', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(directory + save_name + '_underlying_multiplicities.pdf')
    plt.close()

# Number of clusters:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(clustertot_bins_mid, clustertot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(clustertot_bins_mid + model_stagger_errorbars[m], clustertot_counts_qtls[m][:,1], yerr=[clustertot_counts_qtls[m][:,1]-clustertot_counts_qtls[m][:,0], clustertot_counts_qtls[m][:,2]-clustertot_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 5.5])
plt.ylim([0., 1.])
plt.xlabel(r'Clusters per system $N_c$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_clusters.pdf')
    plt.close()

# Number of planets per cluster:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(pl_per_cluster_bins_mid, pl_per_cluster_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(pl_per_cluster_bins_mid + model_stagger_errorbars[m], pl_per_cluster_counts_qtls[m][:,1], yerr=[pl_per_cluster_counts_qtls[m][:,1]-pl_per_cluster_counts_qtls[m][:,0], pl_per_cluster_counts_qtls[m][:,2]-pl_per_cluster_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 7.5])
plt.ylim([0., 0.7])
plt.xlabel(r'Planets per cluster $N_p$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_planets_per_cluster.pdf')
    plt.close()

# Periods:
plot_fig_pdf_simple([sssp['P_all'] for sssp in model_sssp], [], x_min=P_min, x_max=P_max, n_bins=n_bins, log_x=True, log_y=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    label_this = r'16%-84%' if m==0 else ''
    plt.fill_between(P_bins_mid, P_counts_qtls[m][:,0], P_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha, label=label_this)
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_periods.pdf')
    plt.close()

# Period ratios (all):
plot_fig_pdf_simple([sssp['Rm_all'] for sssp in model_sssp], [], x_min=Rm_bins[0], x_max=Rm_bins[-1], y_max=0.07, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(Rm_bins_mid, Rm_counts_qtls[m][:,0], Rm_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
plt.minorticks_off()
if savefigures:
    plt.savefig(directory + save_name + '_underlying_periodratios.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_simple([sssp['radii_all'] for sssp in model_sssp], [], x_min=radii_min, x_max=radii_max, y_max=0.025, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[0.5,1,2,4,10], xlabel_text=r'$R_p$ ($R_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_bins_mid, radii_counts_qtls[m][:,0], radii_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_radii.pdf')
    plt.close()

# Planet radii ratios:
plot_fig_pdf_simple([sssp['radii_ratio_all'] for sssp in model_sssp], [], x_min=radii_ratio_bins[0], x_max=radii_ratio_bins[-1], y_max=0.06, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$R_{p,i+1}/R_{p,i}$', afs=afs, tfs=tfs, lfs=lfs, fig_size=fig_size, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_ratio_bins_mid, radii_ratio_counts_qtls[m][:,0], radii_ratio_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_radii_ratios.pdf')
    plt.close()

plt.show()

# Eccentricity and mutual inclinations vs. intrinsic multiplicity:

fig = plt.figure(figsize=(12,6))
plot = GridSpec(1,2,left=0.1,bottom=0.15,right=0.98,top=0.95,wspace=0.3,hspace=0)

ax = plt.subplot(plot[0,0]) # ecc vs n
e_med_1_qtls = np.quantile(e_med_1_all[0], [0.16,0.5,0.84])
e_med_n_qtls = np.quantile(e_med_n_all[0], [0.16,0.5,0.84], axis=0)
e_plfit_n_qtls_model1 = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all[0], alpha_e_all[0], n_norm=n_norm)
e_plfit_n_qtls_model2 = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all[1], alpha_e_all[1], n_norm=n_norm)
plt.errorbar(n_array, e_med_n_qtls[1,:], yerr=[e_med_n_qtls[1,:]-e_med_n_qtls[0,:], e_med_n_qtls[2,:]-e_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='Maximum AMD model')
plt.errorbar([1], [e_med_1_qtls[1]], yerr=[[e_med_1_qtls[1]-e_med_1_qtls[0]], [e_med_1_qtls[2]-e_med_1_qtls[1]]], fmt='o', capsize=5, ls='none', color='c', label='')
plt.plot(n_array, e_med_n_fcrits[0,:], 'o', lw=lw, color='k', label=r'$f_{\rm crit} = 0.5$')
plt.plot(n_array, e_med_n_fcrits[1,:], 'o', lw=lw, color='r', label=r'$f_{\rm crit} = 2$')
plt.fill_between(np.arange(10)+1, e_plfit_n_qtls_model1[0,:], e_plfit_n_qtls_model1[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.031_{-0.003}^{+0.004}$,' + '\n' + r'$\alpha_e = -1.74_{-0.07}^{+0.10}$')
#plt.fill_between(np.arange(10)+1, e_plfit_n_qtls_model2[0,:], e_plfit_n_qtls_model2[2,:], color='g', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.040_{-0.002}^{+0.003}$, $\alpha_e = -1.54_{-0.11}^{+0.10}$')
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
im_med_n_qtls = np.quantile(im_med_n_all[0], [0.16,0.5,0.84], axis=0)
im_plfit_n_qtls_model1 = compute_power_law_at_n_quantiles(n_array, mu_im_all[0], alpha_im_all[0], n_norm=n_norm)
im_plfit_n_qtls_model2 = compute_power_law_at_n_quantiles(n_array, mu_im_all[1], alpha_im_all[1], n_norm=n_norm)
plt.errorbar(n_array, im_med_n_qtls[1,:], yerr=[im_med_n_qtls[1,:]-im_med_n_qtls[0,:], im_med_n_qtls[2,:]-im_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='')
plt.plot(n_array, im_med_n_fcrits[0,:], 'o', lw=lw, color='k', label='')
plt.plot(n_array, im_med_n_fcrits[1,:], 'o', lw=lw, color='r', label='')
plt.fill_between(n_array, im_plfit_n_qtls_model1[0,:], im_plfit_n_qtls_model1[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.10_{-0.11}^{+0.15}$,' + '\n' + r'$\alpha_i = -1.73_{-0.08}^{+0.09}$')
#plt.fill_between(n_array, im_plfit_n_qtls_model2[0,:], im_plfit_n_qtls_model2[2,:], color='g', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.41_{-0.08}^{+0.10}$, $\alpha_i = -1.55_{-0.11}^{+0.11}$')
plt.plot(n_array, incl_mult_power_law_Zhu2018(n_array)*np.sqrt(2.*np.log(2.)), 'x', ls='-', lw=2, ms=5, color='orange', label=r'$\sigma_{i,5}[^\circ] = 0.8$, $\alpha_i = -3.5$')
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
    plt.savefig(directory + save_name + '_ecc_incl_vs_mult_fits.pdf')
    plt.close()
plt.show()
'''
