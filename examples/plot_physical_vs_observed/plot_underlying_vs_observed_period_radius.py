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
from src.functions_compute_RVs import *





##### To load the underlying and observed populations:

savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/PR_grids/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number

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

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)
N_factor = N_sim/N_Kep

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)





##### To plot period-radius diagrams:

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

bins = 100



# Period-radius diagram (linear radius axis):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(6,10,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[1:,:9])
corner.hist2d(np.log10(sssp['P_all']), sssp['radii_all'], bins=50, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.scatter(np.log10(sss['P_obs']), sss['radii_obs'], s=5, marker='.', c='b')
plt.scatter(np.log10(ssk['P_obs']), ssk['radii_obs'], s=5, marker='.', c='r')
ax.tick_params(axis='both', labelsize=20)
xtick_vals = np.array([3,10,30,100,300])
plt.xticks(np.log10(xtick_vals), xtick_vals)
ax.set_yticks([0.5,2,4,6,8,10])
plt.xlim([np.log10(P_min), np.log10(P_max)])
plt.ylim([radii_min, radii_max])
plt.xlabel(r'Orbital Period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)

ax = plt.subplot(plot[0,:9]) # top histogram
plt.hist(sssp['P_all'], bins=np.logspace(np.log10(P_min), np.log10(P_max), bins+1), weights=np.ones(len(sssp['P_all']))/len(sssp['P_all']), histtype='step', color='k', ls='-', label='Simulated intrinsic')
plt.hist(sss['P_obs'], bins=np.logspace(np.log10(P_min), np.log10(P_max), bins+1), weights=np.ones(len(sss['P_obs']))/len(sss['P_obs']), histtype='step', color='b', ls='-', label='Simulated observed')
plt.hist(ssk['P_obs'], bins=np.logspace(np.log10(P_min), np.log10(P_max), bins+1), weights=np.ones(len(ssk['P_obs']))/len(ssk['P_obs']), histtype='step', color='r', ls='-', label='Kepler')
plt.gca().set_xscale("log")
plt.xlim([P_min, P_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='best', bbox_to_anchor=(0.99,0), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,9]) # side histogram
plt.hist(sssp['radii_all'], bins=np.linspace(radii_min, radii_max, bins+1), weights=np.ones(len(sssp['radii_all']))/len(sssp['radii_all']), histtype='step', orientation='horizontal', color='k', ls='-')
plt.hist(sss['radii_obs'], bins=np.linspace(radii_min, radii_max, bins+1), weights=np.ones(len(sss['radii_obs']))/len(sss['radii_obs']), histtype='step', orientation='horizontal', color='b', ls='-')
plt.hist(ssk['radii_obs'], bins=np.linspace(radii_min, radii_max, bins+1), weights=np.ones(len(ssk['radii_obs']))/len(ssk['radii_obs']), histtype='step', orientation='horizontal', color='r', ls='-')
plt.ylim([radii_min, radii_max])
plt.xticks([])
plt.yticks([])

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_loglinear.pdf')
    plt.close()
plt.show()



# Period-radius diagram (log radius axis):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(6,10,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[1:,:9])
corner.hist2d(np.log10(sssp['P_all']), np.log10(sssp['radii_all']), bins=50, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.scatter(np.log10(sss['P_obs']), np.log10(sss['radii_obs']), s=5, marker='.', c='b')
plt.scatter(np.log10(ssk['P_obs']), np.log10(ssk['radii_obs']), s=5, marker='.', c='r')
ax.tick_params(axis='both', labelsize=20)
xtick_vals = np.array([3,10,30,100,300])
ytick_vals = np.array([0.5,1,2,4,8])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(P_min), np.log10(P_max)])
plt.ylim([np.log10(radii_min), np.log10(radii_max)])
plt.xlabel(r'Orbital Period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)

ax = plt.subplot(plot[0,:9]) # top histogram
plt.hist(sssp['P_all'], bins=np.logspace(np.log10(P_min), np.log10(P_max), bins+1), weights=np.ones(len(sssp['P_all']))/len(sssp['P_all']), histtype='step', color='k', ls='-', label='Simulated intrinsic')
plt.hist(sss['P_obs'], bins=np.logspace(np.log10(P_min), np.log10(P_max), bins+1), weights=np.ones(len(sss['P_obs']))/len(sss['P_obs']), histtype='step', color='b', ls='-', label='Simulated observed')
plt.hist(ssk['P_obs'], bins=np.logspace(np.log10(P_min), np.log10(P_max), bins+1), weights=np.ones(len(ssk['P_obs']))/len(ssk['P_obs']), histtype='step', color='r', ls='-', label='Kepler')
plt.gca().set_xscale("log")
plt.xlim([P_min, P_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='best', bbox_to_anchor=(0.99,0), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,9]) # side histogram
plt.hist(sssp['radii_all'], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), weights=np.ones(len(sssp['radii_all']))/len(sssp['radii_all']), histtype='step', orientation='horizontal', color='k', ls='-')
plt.hist(sss['radii_obs'], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), weights=np.ones(len(sss['radii_obs']))/len(sss['radii_obs']), histtype='step', orientation='horizontal', color='b', ls='-')
plt.hist(ssk['radii_obs'], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), weights=np.ones(len(ssk['radii_obs']))/len(ssk['radii_obs']), histtype='step', orientation='horizontal', color='r', ls='-')
plt.gca().set_yscale("log")
plt.ylim([radii_min, radii_max])
plt.xticks([])
plt.yticks([])

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_loglog.pdf')
    plt.close()
plt.show()



# Period-radius grids (custom bins):

#P_bins = np.logspace(np.log10(P_min), np.log10(P_max), 5+1)
#R_bins = np.array([0.5, 1., 1.5, 2., 3., 5., 10.])
P_bins = np.array([4., 8., 16., 32., 64., 128., 256.])
R_bins = np.array([0.5, 1., 1.5, 2., 3., 4., 6.])
n_P_bins, n_R_bins = len(P_bins)-1, len(R_bins)-1

# Observed counts in each bin:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.85,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[:,:])
counts_sim_grid = np.zeros((n_R_bins, n_P_bins))
counts_Kep_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        counts_sim_cell = np.sum((sss['P_obs'] > P_bins[i]) & (sss['P_obs'] < P_bins[i+1]) & (sss['radii_obs'] > R_bins[j]) & (sss['radii_obs'] < R_bins[j+1]))
        counts_Kep_cell = np.sum((ssk['P_obs'] > P_bins[i]) & (ssk['P_obs'] < P_bins[i+1]) & (ssk['radii_obs'] > R_bins[j]) & (ssk['radii_obs'] < R_bins[j+1]))
        counts_sim_grid[j,i] = counts_sim_cell
        counts_Kep_grid[j,i] = counts_Kep_cell
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.025, s='%s' % np.round(counts_sim_cell/N_factor, 1), ha='right', va='top', color='k', fontsize=16, transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s='%s' % counts_Kep_cell, ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
        plt.text(x=0.02+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s='%s' % np.round((counts_sim_cell/N_factor)/float(counts_Kep_cell), 2), ha='left', va='center', color='b', fontsize=20, transform=ax.transAxes)
counts_normed_sim_grid = counts_sim_grid/N_factor
plt.imshow(counts_normed_sim_grid/counts_Kep_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", vmin=0., vmax=2., origin='lower') #cmap='coolwarm'
cbar = plt.colorbar()
cbar.set_label(r'$N_{\rm Sim}/N_{\rm Kep}$', rotation=270, va='bottom', fontsize=20)
#plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Orbital Period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)

plot = GridSpec(1,1,left=0.86,bottom=0.81,right=0.98,top=0.95,wspace=0,hspace=0) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.9, y=0.9, s=r'$N_{\rm Sim}$', ha='right', va='top', color='k', fontsize=16, transform=ax.transAxes)
plt.text(x=0.9, y=0.6, s=r'$N_{\rm Kep}$', ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
plt.text(x=0.1, y=0.5, s=r'$\frac{N_{\rm Sim}}{N_{\rm Kep}}$', ha='left', va='center', color='b', fontsize=20, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=20)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_counts_obs.pdf')
    plt.close()
plt.show()



PRK_obs = [] # 2D array to be filled with: [period, radius, K status, mean intrinsic multiplicity, mean multiplicity with K > 0.1m/s, mean multiplicity with K > 1m/s] for each observed planet, where K status = 1 (it is an intrinsic single), 2 (it is the largest K in a multiplanet system), 3 (it is NOT the largest K in a multiplanet system)

for i,det_sys in enumerate(sssp_per_sys['det_all']):
    n_pl_det = np.sum(det_sys)
    if n_pl_det > 0:
        P_sys = sssp_per_sys['P_all'][i]
        det_sys = det_sys[P_sys > 0]
        Mp_sys = sssp_per_sys['mass_all'][i][P_sys > 0]
        Rp_sys = sssp_per_sys['radii_all'][i][P_sys > 0]
        e_sys = sssp_per_sys['e_all'][i][P_sys > 0]
        incl_sys = sssp_per_sys['incl_all'][i][P_sys > 0]
        P_sys = P_sys[P_sys > 0]
        n_pl = len(P_sys)
        
        K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=sssp['Mstar_all'][i])
        n_pl_K0p1 = np.sum(K_sys > 0.1)
        n_pl_K1 = np.sum(K_sys > 1.)
        
        if n_pl == 1:
            PRK_obs.append([P_sys[0], Rp_sys[0], 1, n_pl, n_pl_K0p1, n_pl_K1, 0, 0, 0])
        else:
            j_Kmax = np.argsort(K_sys)[-1]
            for j in np.arange(n_pl)[det_sys == 1]:
                n_pl_miss_interior = np.sum(det_sys[:j] == 0)
                n_pl_miss_interior_K0p1 = np.sum((det_sys[:j] == 0) & (K_sys[:j] > 0.1))
                n_pl_miss_interior_K1 = np.sum((det_sys[:j] == 0) & (K_sys[:j] > 1.))
                if j == j_Kmax:
                    PRK_obs.append([P_sys[j], Rp_sys[j], 2, n_pl, n_pl_K0p1, n_pl_K1, n_pl_miss_interior, n_pl_miss_interior_K0p1, n_pl_miss_interior_K1])
                else:
                    PRK_obs.append([P_sys[j], Rp_sys[j], 3, n_pl, n_pl_K0p1, n_pl_K1, n_pl_miss_interior, n_pl_miss_interior_K0p1, n_pl_miss_interior_K1])
PRK_obs = np.array(PRK_obs)

# Fraction of time when observed planet is the maximum K:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.85,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[:,:])
counts_frac_K3_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
        counts_tot_cell = len(PRK_obs_cell) # number of observed planets in cell
        counts_K1_cell = np.sum(PRK_obs_cell[:,2] == 1) # number of observed planets in cell that are intrinsic singles
        counts_K2_cell = np.sum(PRK_obs_cell[:,2] == 2) # number of observed planets in cell that are the largest K in their multiplanet systems
        counts_K3_cell = np.sum(PRK_obs_cell[:,2] == 3) # number of observed planets in cell that are NOT the largest K in their multiplanet systems
        
        counts_frac_K3_grid[j,i] = counts_K3_cell/counts_tot_cell
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.025, s='%s' % counts_K1_cell, ha='right', va='top', color='k', fontsize=16, transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s='%s' % counts_K2_cell, ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.125, s='%s' % counts_K3_cell, ha='right', va='top', color='b', fontsize=16, transform=ax.transAxes)
        plt.text(x=0.02+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s='%s' % np.round(counts_K3_cell/counts_tot_cell, 2), ha='left', va='center', color='g', fontsize=20, fontweight='bold', transform=ax.transAxes)
plt.imshow(counts_frac_K3_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", vmin=0., vmax=1., origin='lower') #cmap='coolwarm'
cbar = plt.colorbar()
cbar.set_label(r'$\frac{N_{K<K_{\rm max}}}{N_{\rm tot}}$', rotation=270, va='bottom', fontsize=20)
#plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Orbital Period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)

plot = GridSpec(1,1,left=0.86,bottom=0.81,right=0.98,top=0.95,wspace=0,hspace=0) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.9, y=0.9, s=r'$N_{n=1}$', ha='right', va='top', color='k', fontsize=16, transform=ax.transAxes)
plt.text(x=0.9, y=0.6, s=r'$N_{K_{\rm max}}$', ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
plt.text(x=0.9, y=0.3, s=r'$N_{K<K_{\rm max}}$', ha='right', va='top', color='b', fontsize=16, transform=ax.transAxes)
plt.text(x=0.1, y=0.5, s=r'$\frac{N_{K<K_{\rm max}}}{N_{\rm tot}}$', ha='left', va='center', color='g', fontsize=20, fontweight='bold', transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=20)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_fraction_Kmax.pdf')
    plt.close()
#plt.show()

# Mean intrinsic multiplicities (all planets and with various K thresholds):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.85,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[:,:])
mean_n_pl_grid = np.zeros((n_R_bins, n_P_bins))
mean_n_pl_K0p1_grid = np.zeros((n_R_bins, n_P_bins))
mean_n_pl_K1_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
        mean_n_pl = np.mean(PRK_obs_cell[:,3]) # mean intrinsic multiplicity
        mean_n_pl_K0p1 = np.mean(PRK_obs_cell[:,4]) # mean intrinsic multiplicity of K > 0.1m/s
        mean_n_pl_K1 = np.mean(PRK_obs_cell[:,5]) # mean intrinsic multiplicity of K > 1m/s
        
        mean_n_pl_grid[j,i] = mean_n_pl
        mean_n_pl_K0p1_grid[j,i] = mean_n_pl_K0p1
        mean_n_pl_K1_grid[j,i] = mean_n_pl_K1
        plt.text(x=0.02+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s='%s' % np.round(mean_n_pl, 2), ha='left', va='center', color='k', fontsize=20, fontweight='bold', transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s='%s' % np.round(mean_n_pl_K0p1, 2), ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.125, s='%s' % np.round(mean_n_pl_K1, 2), ha='right', va='top', color='b', fontsize=16, transform=ax.transAxes)
plt.imshow(mean_n_pl_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower') #cmap='coolwarm'
cbar = plt.colorbar()
cbar.set_label(r'Mean intrinsic multiplicity $\bar{n}$', rotation=270, va='bottom', fontsize=20)
#plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Orbital Period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)

plot = GridSpec(1,1,left=0.86,bottom=0.81,right=0.98,top=0.95,wspace=0,hspace=0) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.1, y=0.5, s=r'$\bar{n}$', ha='left', va='center', color='k', fontsize=20, fontweight='bold', transform=ax.transAxes)
plt.text(x=0.9, y=0.6, s=r'$\bar{n}_{K > 0.1m/s}$', ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
plt.text(x=0.9, y=0.3, s=r'$\bar{n}_{K > 1m/s}$', ha='right', va='top', color='b', fontsize=16, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=20)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_mean_mult_true.pdf')
    plt.close()
#plt.show()

# Mean number of missed (undetected), interior planets (all planets and with various K thresholds):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.85,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[:,:])
mean_n_pl_miss_interior_grid = np.zeros((n_R_bins, n_P_bins))
mean_n_pl_miss_interior_K0p1_grid = np.zeros((n_R_bins, n_P_bins))
mean_n_pl_miss_interior_K1_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
        mean_n_pl_miss_interior = np.mean(PRK_obs_cell[:,6]) # mean number of missed planets interior
        mean_n_pl_miss_interior_K0p1 = np.mean(PRK_obs_cell[:,7]) # mean number of missed planets interior, with K > 0.1m/s
        mean_n_pl_miss_interior_K1 = np.mean(PRK_obs_cell[:,8]) # mean number of missed planets interior, with K > 1m/s
        
        mean_n_pl_miss_interior_grid[j,i] = mean_n_pl_miss_interior
        mean_n_pl_miss_interior_K0p1_grid[j,i] = mean_n_pl_miss_interior_K0p1
        mean_n_pl_miss_interior_K1_grid[j,i] = mean_n_pl_miss_interior_K1
        plt.text(x=0.02+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s='%s' % np.round(mean_n_pl_miss_interior, 2), ha='left', va='center', color='k', fontsize=20, fontweight='bold', transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s='%s' % np.round(mean_n_pl_miss_interior_K0p1, 2), ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.125, s='%s' % np.round(mean_n_pl_miss_interior_K1, 2), ha='right', va='top', color='b', fontsize=16, transform=ax.transAxes)
plt.imshow(mean_n_pl_miss_interior_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower') #cmap='coolwarm'
cbar = plt.colorbar()
cbar.set_label(r'$\bar{n}_{\rm missed,interior}$', rotation=270, va='bottom', fontsize=20)
#plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Orbital Period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)

plot = GridSpec(1,1,left=0.86,bottom=0.81,right=0.98,top=0.95,wspace=0,hspace=0) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.1, y=0.5, s='(1)', ha='left', va='center', color='k', fontsize=20, fontweight='bold', transform=ax.transAxes)
plt.text(x=0.9, y=0.6, s='(2)', ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
plt.text(x=0.9, y=0.3, s='(3)', ha='right', va='top', color='b', fontsize=16, transform=ax.transAxes)
plt.text(x=-0.5, y=-0.5, s=r'(1) $\bar{n}_{\rm missed,interior}$', ha='left', va='center', color='k', fontsize=16, transform=ax.transAxes)
plt.text(x=-0.5, y=-0.75, s=r'(2) $\bar{n}_{\rm missed,interior,K > 0.1m/s}$', ha='left', va='center', color='r', fontsize=16, transform=ax.transAxes)
plt.text(x=-0.5, y=-1, s=r'(3) $\bar{n}_{\rm missed,interior,K > 1m/s}$', ha='left', va='center', color='b', fontsize=16, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=20)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_mean_mult_missed_interior.pdf')
    plt.close()
#plt.show()

# Fraction of time when there are missed (undetected), interior planets (all planets and with various K thresholds):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.85,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[:,:])
f_miss_interior_grid = np.zeros((n_R_bins, n_P_bins))
f_miss_interior_K0p1_grid = np.zeros((n_R_bins, n_P_bins))
f_miss_interior_K1_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
        counts_tot_cell = len(PRK_obs_cell) # number of observed planets in cell
        counts_miss_interior = np.sum(PRK_obs_cell[:,6] > 0) # number of times there are missed planets interior
        counts_miss_interior_K0p1 = np.sum(PRK_obs_cell[:,7] > 0) # number of times there are missed planets interior, with K > 0.1m/s
        counts_miss_interior_K1 = np.sum(PRK_obs_cell[:,8] > 0) # number of times there are missed planets interior, with K > 1m/s
        
        f_miss_interior_grid[j,i] = counts_miss_interior/counts_tot_cell
        f_miss_interior_K0p1_grid[j,i] = counts_miss_interior_K0p1/counts_tot_cell
        f_miss_interior_K1_grid[j,i] = counts_miss_interior_K1/counts_tot_cell
        plt.text(x=0.02+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s='%s' % np.round(counts_miss_interior/counts_tot_cell, 2), ha='left', va='center', color='g', fontsize=20, fontweight='bold', transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.015, s='%s' % counts_tot_cell, ha='right', va='top', color='k', fontsize=16, transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.055, s='%s' % counts_miss_interior, ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.095, s='%s' % counts_miss_interior_K0p1, ha='right', va='top', color='b', fontsize=16, transform=ax.transAxes)
        plt.text(x=-0.02+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.135, s='%s' % counts_miss_interior_K1, ha='right', va='top', color='m', fontsize=16, transform=ax.transAxes)
plt.imshow(f_miss_interior_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower') #cmap='coolwarm'
cbar = plt.colorbar()
cbar.set_label(r'$N_{\rm missed,interior}/N_{\rm tot}$', rotation=270, va='bottom', fontsize=20)
#plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Orbital Period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)

plot = GridSpec(1,1,left=0.86,bottom=0.81,right=0.98,top=0.95,wspace=0,hspace=0) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.1, y=0.5, s='(1)', ha='left', va='center', color='g', fontsize=20, fontweight='bold', transform=ax.transAxes)
plt.text(x=0.9, y=0.9, s='(2)', ha='right', va='top', color='k', fontsize=16, transform=ax.transAxes)
plt.text(x=0.9, y=0.7, s='(3)', ha='right', va='top', color='r', fontsize=16, transform=ax.transAxes)
plt.text(x=0.9, y=0.5, s='(4)', ha='right', va='top', color='b', fontsize=16, transform=ax.transAxes)
plt.text(x=0.9, y=0.3, s='(5)', ha='right', va='top', color='m', fontsize=16, transform=ax.transAxes)
plt.text(x=-0.5, y=-0.5, s=r'(1) $N_{\rm missed,interior}/N_{\rm tot}$', ha='left', va='center', color='g', fontsize=16, transform=ax.transAxes)
plt.text(x=-0.5, y=-0.75, s=r'(2) $N_{\rm tot}$', ha='left', va='center', color='k', fontsize=16, transform=ax.transAxes)
plt.text(x=-0.5, y=-1, s=r'(3) $N_{\rm missed,interior}$', ha='left', va='center', color='r', fontsize=16, transform=ax.transAxes)
plt.text(x=-0.5, y=-1.25, s=r'(4) $N_{\rm missed,interior,K > 0.1m/s}$', ha='left', va='center', color='b', fontsize=16, transform=ax.transAxes)
plt.text(x=-0.5, y=-1.5, s=r'(5) $N_{\rm missed,interior,K > 1m/s}$', ha='left', va='center', color='m', fontsize=16, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=20)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_fraction_missed_interior.pdf')
    plt.close()
plt.show()
