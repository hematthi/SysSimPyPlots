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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/PR_grids/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number

compute_ratios = compute_ratios_adjacent
AD_mod = 'true' # 'true' or 'false'
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

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)





##### To load many catalogs:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'

runs = 100

PRK_obs_all = [] # 2D array to be filled with: [period, radius, K status, mean intrinsic multiplicity, mean multiplicity with K > 0.1m/s, mean multiplicity with K > 1m/s] for each observed planet, where K status = 1 (it is an intrinsic single), 2 (it is the largest K in a multiplanet system), 3 (it is NOT the largest K in a multiplanet system)
sssp_per_sys_P_all = []
sssp_per_sys_R_all = []
for i in range(runs): #range(1,runs+1)
    run_number = i+1
    print(i)
    N_sim_i = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)[0]
    param_vals_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

    PRK_obs = [] # 2D array to be filled with: [period, radius, K status, mean intrinsic multiplicity, mean multiplicity with K > 0.1m/s, mean multiplicity with K > 1m/s] for each observed planet, where K status = 1 (it is an intrinsic single), 2 (it is the largest K in a multiplanet system), 3 (it is NOT the largest K in a multiplanet system)
    for i,det_sys in enumerate(sssp_per_sys_i['det_all']):
        n_pl_det = np.sum(det_sys)
        if n_pl_det > 0:
            P_sys = sssp_per_sys_i['P_all'][i]
            det_sys = det_sys[P_sys > 0]
            Mp_sys = sssp_per_sys_i['mass_all'][i][P_sys > 0]
            Rp_sys = sssp_per_sys_i['radii_all'][i][P_sys > 0]
            e_sys = sssp_per_sys_i['e_all'][i][P_sys > 0]
            incl_sys = sssp_per_sys_i['incl_all'][i][P_sys > 0]
            P_sys = P_sys[P_sys > 0]
            n_pl = len(P_sys)
            
            K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=sssp_i['Mstar_all'][i])
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

    PRK_obs_all.append(PRK_obs)
    sssp_per_sys_P_all.append(sssp_per_sys_i['P_all'])
    sssp_per_sys_R_all.append(sssp_per_sys_i['radii_all'])





##### To plot period-radius diagrams:

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

mfs = 16 # main numbers font size
sfs = 12 # secondary numbers font size

# Period-radius grids (custom bins):

#P_bins = np.logspace(np.log10(P_min), np.log10(P_max), 5+1)
#R_bins = np.array([0.5, 1., 1.5, 2., 3., 5., 10.])
P_bins = np.array([4., 8., 16., 32., 64., 128., 256.])
R_bins = np.array([0.5, 1., 1.5, 2., 3., 4., 6.])
n_P_bins, n_R_bins = len(P_bins)-1, len(R_bins)-1

# Specify edges of GridSpec panels to ensure that the legend is the same size as a cell:
bgrid, tgrid = 0.1, 0.95 # bottom and top of grid
lgrid, rleg, wcb = 0.08, 0.97, 0.09 # left of grid, width of space for colorbar, and right of legend
rgrid = (rleg-lgrid-wcb)*(n_P_bins/(n_P_bins+1)) + lgrid
lleg = rgrid + wcb
bleg, tleg = tgrid - (tgrid-bgrid)/n_R_bins, tgrid



# Occurrence rates (intrinsic mean number of planets per star and fraction of stars with planets) in each bin:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)

ax = plt.subplot(plot[:,:])
mean_pl_grid = np.zeros((n_R_bins, n_P_bins))
fswp_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        pl_tot_cell_all = []
        sys_tot_cell_all = []
        for k in range(runs):
            pl_cell_bools = (sssp_per_sys_P_all[k] > P_bins[i]) & (sssp_per_sys_P_all[k] < P_bins[i+1]) & (sssp_per_sys_R_all[k] > R_bins[j]) & (sssp_per_sys_R_all[k] < R_bins[j+1])
            sys_cell_bools = np.any(pl_cell_bools, axis=1)
            pl_tot_cell_all.append(np.sum(pl_cell_bools))
            sys_tot_cell_all.append(np.sum(sys_cell_bools))
        pl_tot_cell_all = np.array(pl_tot_cell_all)
        sys_tot_cell_all = np.array(sys_tot_cell_all)
        
        mean_pl_cell_all = pl_tot_cell_all/N_sim_i # mean number of such planets per star
        mean_pl_sys_cell_all = pl_tot_cell_all/sys_tot_cell_all # mean number of such planets per system with at least one such planet
        fswp_cell_all = sys_tot_cell_all/N_sim_i # fraction of stars with such planets
        
        mean_pl_cell_qtls = np.quantile(mean_pl_cell_all, [0.16,0.5,0.84])
        mean_pl_sys_cell_qtls = np.quantile(mean_pl_sys_cell_all, [0.16,0.5,0.84])
        fswp_cell_qtls = np.quantile(fswp_cell_all, [0.16,0.5,0.84])
        
        mean_pl_grid[j,i] = mean_pl_cell_qtls[1]
        fswp_grid[j,i] = fswp_cell_qtls[1]
        
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.025, s=r'${:.2f}$'.format(np.round(mean_pl_cell_qtls[1], 2)), ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s=r'${:.2f}$'.format(np.round(mean_pl_sys_cell_qtls[1], 2)), ha='right', va='top', color='r', fontsize=sfs, transform=ax.transAxes)
        snum = r'${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'.format(np.round(fswp_cell_qtls[1], 2), np.round(fswp_cell_qtls[1]-fswp_cell_qtls[0], 2), np.round(fswp_cell_qtls[2]-fswp_cell_qtls[1], 2))
        plt.text(x=0.01+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s=snum, ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(fswp_grid, cmap='coolwarm', norm=matplotlib.colors.LogNorm(), aspect='auto', interpolation="nearest", origin='lower') #cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Period of conditioned planet, $P_{\rm cond}$ (days)', fontsize=tfs)
plt.ylabel(r'Radius of conditioned planet, $R_{p,\rm cond}$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$f_{\rm swp} = \frac{\bar{n}_{\rm star}}{\bar{n}_{\rm sys}}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.95, y=0.9, s=r'$\bar{n}_{\rm star}$', ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.95, y=0.6, s=r'$\bar{n}_{\rm sys}$', ha='right', va='top', color='r', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.05, y=0.5, s=r'$f_{\rm swp} = \frac{\bar{n}_{\rm star}}{\bar{n}_{\rm sys}}$', ha='left', va='center', color='k', fontsize=mfs, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_mean_pl_fswp.pdf')
    plt.close()
#plt.show()



# Observed counts in each bin:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)

ax = plt.subplot(plot[:,:])
counts_sim_grid = np.zeros((n_R_bins, n_P_bins))
counts_Kep_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        counts_cell_all = []
        for PRK_obs in PRK_obs_all:
            PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
            counts_cell = len(PRK_obs_cell) # number of observed planets in cell
            counts_cell_all.append(counts_cell)
        counts_cell_all = np.array(counts_cell_all)
        
        counts_sim_cell_qtls = np.quantile(counts_cell_all, [0.16,0.5,0.84])
        counts_Kep_cell = np.sum((ssk['P_obs'] > P_bins[i]) & (ssk['P_obs'] < P_bins[i+1]) & (ssk['radii_obs'] > R_bins[j]) & (ssk['radii_obs'] < R_bins[j+1]))
        counts_ratio_cell_qtls = np.quantile(counts_cell_all/counts_Kep_cell, [0.16,0.5,0.84])
        counts_sim_grid[j,i] = counts_sim_cell_qtls[1]
        counts_Kep_grid[j,i] = counts_Kep_cell
        
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.025, s=r'${:.1f}$'.format(np.round(counts_sim_cell_qtls[1], 1)), ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s='%s' % counts_Kep_cell, ha='right', va='top', color='r', fontsize=sfs, transform=ax.transAxes)
        snum = r'${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'.format(np.round(counts_ratio_cell_qtls[1], 2), np.round(counts_ratio_cell_qtls[1]-counts_ratio_cell_qtls[0], 2), np.round(counts_ratio_cell_qtls[2]-counts_ratio_cell_qtls[1], 2)) if counts_Kep_cell > 0 else r'$-$'
        plt.text(x=0.01+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s=snum, ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(counts_sim_grid/counts_Kep_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", vmin=0., vmax=2., origin='lower') #cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Period of conditioned planet, $P_{\rm cond}$ (days)', fontsize=tfs)
plt.ylabel(r'Radius of conditioned planet, $R_{p,\rm cond}$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$N_{\rm Sim}/N_{\rm Kep}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.9, y=0.9, s=r'$N_{\rm Sim}$', ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.9, y=0.6, s=r'$N_{\rm Kep}$', ha='right', va='top', color='r', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.1, y=0.5, s=r'$\frac{N_{\rm Sim}}{N_{\rm Kep}}$', ha='left', va='center', color='k', fontsize=mfs, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_counts_obs.pdf')
    plt.close()
#plt.show()



# Fraction of time when observed planet is the maximum K:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)

ax = plt.subplot(plot[:,:])
frac_K3_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        counts_cell_all = []
        counts_K1_cell_all = []
        counts_K2_cell_all = []
        counts_K3_cell_all = []
        for PRK_obs in PRK_obs_all:
            PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
            counts_cell = len(PRK_obs_cell) # number of observed planets in cell
            counts_K1_cell = np.sum(PRK_obs_cell[:,2] == 1) # number of observed planets in cell that are intrinsic singles
            counts_K2_cell = np.sum(PRK_obs_cell[:,2] == 2) # number of observed planets in cell that are the largest K in their multiplanet systems
            counts_K3_cell = np.sum(PRK_obs_cell[:,2] == 3) # number of observed planets in cell that are NOT the largest K in their multiplanet systems
            counts_cell_all.append(counts_cell)
            counts_K1_cell_all.append(counts_K1_cell)
            counts_K2_cell_all.append(counts_K2_cell)
            counts_K3_cell_all.append(counts_K3_cell)
        counts_cell_all = np.array(counts_cell_all)
        counts_K1_cell_all = np.array(counts_K1_cell_all)
        counts_K2_cell_all = np.array(counts_K2_cell_all)
        counts_K3_cell_all = np.array(counts_K3_cell_all)

        counts_cell_qtls = np.quantile(counts_cell_all, [0.16,0.5,0.84])
        counts_K1_cell_qtls = np.quantile(counts_K1_cell_all, [0.16,0.5,0.84])
        counts_K2_cell_qtls = np.quantile(counts_K2_cell_all, [0.16,0.5,0.84])
        counts_K3_cell_qtls = np.quantile(counts_K3_cell_all, [0.16,0.5,0.84])

        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.025, s=r'${:.2f}$'.format(np.round(np.sum(counts_K1_cell_all)/np.sum(counts_cell_all), 2)), ha='right', va='top', color='darkblue', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s=r'${:.2f}$'.format(np.round(np.sum(counts_K2_cell_all)/np.sum(counts_cell_all), 2)), ha='right', va='top', color='maroon', fontsize=sfs, transform=ax.transAxes)
        #plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.125, s=r'${:.1f}$'.format(np.round(counts_K3_cell_qtls[1], 1)), ha='right', va='top', color='g', fontsize=sfs, transform=ax.transAxes)

        frac_K3_cell_qtls = np.nanquantile(counts_K3_cell_all/counts_cell_all, [0.16,0.5,0.84])
        frac_K3_grid[j,i] = frac_K3_cell_qtls[1]
        snum = r'${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'.format(np.round(frac_K3_cell_qtls[1], 2), np.round(frac_K3_cell_qtls[1]-frac_K3_cell_qtls[0], 2), np.round(frac_K3_cell_qtls[2]-frac_K3_cell_qtls[1], 2))
        plt.text(x=0.01+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s=snum, ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(frac_K3_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", vmin=0., vmax=1., origin='lower') #cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Period of conditioned planet, $P_{\rm cond}$ (days)', fontsize=tfs)
plt.ylabel(r'Radius of conditioned planet, $R_{p,\rm cond}$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$f_{K<K_{\rm max}}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.9, y=0.9, s=r'$f_{n=1}$', ha='right', va='top', color='darkblue', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.9, y=0.6, s=r'$f_{K_{\rm max}}$', ha='right', va='top', color='maroon', fontsize=sfs, transform=ax.transAxes)
#plt.text(x=0.9, y=0.3, s=r'$N_{K<K_{\rm max}}$', ha='right', va='top', color='g', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.1, y=0.5, s=r'$f_{K<K_{\rm max}}$', ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_fraction_Kmax.pdf')
    plt.close()
#plt.show()



# Mean intrinsic multiplicities (all planets and with various K thresholds):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)

ax = plt.subplot(plot[:,:])
nbar_grid = np.zeros((n_R_bins, n_P_bins))
nbar_K0p1_grid = np.zeros((n_R_bins, n_P_bins))
nbar_K1_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        nbar_cell_all = []
        nbar_K0p1_cell_all = []
        nbar_K1_cell_all = []
        for PRK_obs in PRK_obs_all:
            PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
            if len(PRK_obs_cell) > 0:
                nbar_cell = np.mean(PRK_obs_cell[:,3]) # mean intrinsic multiplicity
                nbar_K0p1_cell = np.mean(PRK_obs_cell[:,4]) # mean intrinsic multiplicity of K > 0.1m/s
                nbar_K1_cell = np.mean(PRK_obs_cell[:,5]) # mean intrinsic multiplicity of K > 1m/s
                nbar_cell_all.append(nbar_cell)
                nbar_K0p1_cell_all.append(nbar_K0p1_cell)
                nbar_K1_cell_all.append(nbar_K1_cell)
        nbar_cell_all = np.array(nbar_cell_all)
        nbar_K0p1_cell_all = np.array(nbar_K0p1_cell_all)
        nbar_K1_cell_all = np.array(nbar_K1_cell_all)
        
        nbar_cell_qtls = np.quantile(nbar_cell_all, [0.16,0.5,0.84])
        nbar_K0p1_cell_qtls = np.quantile(nbar_K0p1_cell_all, [0.16,0.5,0.84])
        nbar_K1_cell_qtls = np.quantile(nbar_K1_cell_all, [0.16,0.5,0.84])
        
        nbar_grid[j,i] = nbar_cell_qtls[1]
        nbar_K0p1_grid[j,i] = nbar_K0p1_cell_qtls[1]
        nbar_K1_grid[j,i] = nbar_K1_cell_qtls[1]
        snum = r'${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'.format(np.round(nbar_cell_qtls[1], 2), np.round(nbar_cell_qtls[1]-nbar_cell_qtls[0], 2), np.round(nbar_cell_qtls[2]-nbar_cell_qtls[1], 2))
        plt.text(x=0.01+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s=snum, ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s=r'${:.2f}$'.format(np.round(np.mean(nbar_K0p1_cell_all), 2)), ha='right', va='top', color='maroon', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.125, s=r'${:.2f}$'.format(np.round(np.mean(nbar_K1_cell_all), 2)), ha='right', va='top', color='darkblue', fontsize=sfs, transform=ax.transAxes)
img = plt.imshow(nbar_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower') #cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Period of conditioned planet, $P_{\rm cond}$ (days)', fontsize=tfs)
plt.ylabel(r'Radius of conditioned planet, $R_{p,\rm cond}$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$\bar{n}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.1, y=0.5, s=r'$\bar{n}$', ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
plt.text(x=0.9, y=0.6, s=r'$\bar{n}_{K > 0.1m/s}$', ha='right', va='top', color='maroon', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.9, y=0.3, s=r'$\bar{n}_{K > 1m/s}$', ha='right', va='top', color='darkblue', fontsize=sfs, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_mean_mult_true.pdf')
    plt.close()
#plt.show()



# Mean number of missed (undetected), interior planets (all planets and with various K thresholds):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)

ax = plt.subplot(plot[:,:])
nbar_missin_grid = np.zeros((n_R_bins, n_P_bins))
nbar_missin_K0p1_grid = np.zeros((n_R_bins, n_P_bins))
nbar_missin_K1_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        nbar_missin_cell_all = []
        nbar_missin_K0p1_cell_all = []
        nbar_missin_K1_cell_all = []
        for PRK_obs in PRK_obs_all:
            PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
            if len(PRK_obs_cell) > 0:
                nbar_missin_cell = np.mean(PRK_obs_cell[:,6]) # mean number of missed planets interior
                nbar_missin_K0p1_cell = np.mean(PRK_obs_cell[:,7]) # mean number of missed planets interior, with K > 0.1m/s
                nbar_missin_K1_cell = np.mean(PRK_obs_cell[:,8]) # mean number of missed planets interior, with K > 1m/s
                nbar_missin_cell_all.append(nbar_missin_cell)
                nbar_missin_K0p1_cell_all.append(nbar_missin_K0p1_cell)
                nbar_missin_K1_cell_all.append(nbar_missin_K1_cell)
        nbar_missin_cell_all = np.array(nbar_missin_cell_all)
        nbar_missin_K0p1_cell_all = np.array(nbar_missin_K0p1_cell_all)
        nbar_missin_K1_cell_all = np.array(nbar_missin_K1_cell_all)
        
        nbar_missin_cell_qtls = np.quantile(nbar_missin_cell_all, [0.16,0.5,0.84])
        nbar_missin_K0p1_cell_qtls = np.quantile(nbar_missin_K0p1_cell_all, [0.16,0.5,0.84])
        nbar_missin_K1_cell_qtls = np.quantile(nbar_missin_K1_cell_all, [0.16,0.5,0.84])
        
        nbar_missin_grid[j,i] = nbar_missin_cell_qtls[1]
        nbar_missin_K0p1_grid[j,i] = nbar_missin_K0p1_cell_qtls[1]
        nbar_missin_K1_grid[j,i] = nbar_missin_K1_cell_qtls[1]
        snum = r'${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'.format(np.round(nbar_missin_cell_qtls[1], 2), np.round(nbar_missin_cell_qtls[1]-nbar_missin_cell_qtls[0], 2), np.round(nbar_missin_cell_qtls[2]-nbar_missin_cell_qtls[1], 2))
        plt.text(x=0.01+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s=snum, ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s=r'${:.2f}$'.format(np.round(np.mean(nbar_missin_K0p1_cell_all), 2)), ha='right', va='top', color='maroon', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.125, s=r'${:.2f}$'.format(np.round(np.mean(nbar_missin_K1_cell_all), 2)), ha='right', va='top', color='darkblue', fontsize=sfs, transform=ax.transAxes)
img = plt.imshow(nbar_missin_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower') #cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Period of conditioned planet, $P_{\rm cond}$ (days)', fontsize=tfs)
plt.ylabel(r'Radius of conditioned planet, $R_{p,\rm cond}$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$\bar{n}_{\rm miss,in}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.1, y=0.5, s='(1)', ha='left', va='center', color='k', fontsize=mfs, transform=ax.transAxes)
plt.text(x=0.9, y=0.6, s='(2)', ha='right', va='top', color='maroon', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.9, y=0.3, s='(3)', ha='right', va='top', color='darkblue', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0, y=-0.5, s=r'(1) $\bar{n}_{\rm miss,in}$', ha='left', va='center', color='k', fontsize=lfs, transform=ax.transAxes)
plt.text(x=0, y=-0.75, s=r'(2) $\bar{n}_{\rm miss,in,K > 0.1m/s}$', ha='left', va='center', color='maroon', fontsize=lfs, transform=ax.transAxes)
plt.text(x=0, y=-1, s=r'(3) $\bar{n}_{\rm miss,in,K > 1m/s}$', ha='left', va='center', color='darkblue', fontsize=lfs, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_mean_mult_missed_interior.pdf')
    plt.close()
#plt.show()



# Fraction of time when there are missed (undetected), interior planets (all planets and with various K thresholds):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)

ax = plt.subplot(plot[:,:])
f_missin_grid = np.zeros((n_R_bins, n_P_bins))
f_missin_K0p1_grid = np.zeros((n_R_bins, n_P_bins))
f_missin_K1_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        counts_cell_all = []
        counts_missin_cell_all = []
        counts_missin_K0p1_cell_all = []
        counts_missin_K1_cell_all = []
        for PRK_obs in PRK_obs_all:
            PRK_obs_cell = PRK_obs[(PRK_obs[:,0] > P_bins[i]) & (PRK_obs[:,0] < P_bins[i+1]) & (PRK_obs[:,1] > R_bins[j]) & (PRK_obs[:,1] < R_bins[j+1])]
            counts_cell = len(PRK_obs_cell)
            counts_missin_cell = np.sum(PRK_obs_cell[:,6] > 0) # number of times there are missed planets interior
            counts_missin_K0p1_cell = np.sum(PRK_obs_cell[:,7] > 0) # number of times there are missed planets interior, with K > 0.1m/s
            counts_missin_K1_cell = np.sum(PRK_obs_cell[:,8] > 0) # number of times there are missed planets interior, with K > 1m/s
            counts_cell_all.append(counts_cell)
            counts_missin_cell_all.append(counts_missin_cell)
            counts_missin_K0p1_cell_all.append(counts_missin_K0p1_cell)
            counts_missin_K1_cell_all.append(counts_missin_K1_cell)
        counts_cell_all = np.array(counts_cell_all)
        counts_missin_cell_all = np.array(counts_missin_cell_all)
        counts_missin_K0p1_cell_all = np.array(counts_missin_K0p1_cell_all)
        counts_missin_K1_cell_all = np.array(counts_missin_K1_cell_all)
        
        counts_cell_qtls = np.quantile(counts_cell_all, [0.16,0.5,0.84])
        counts_missin_cell_qtls = np.quantile(counts_missin_cell_all, [0.16,0.5,0.84])
        counts_missin_K0p1_cell_qtls = np.quantile(counts_missin_K0p1_cell_all, [0.16,0.5,0.84])
        counts_missin_K1_cell_qtls = np.quantile(counts_missin_K1_cell_all, [0.16,0.5,0.84])
        
        f_missin_cell_qtls = np.nanquantile(counts_missin_cell_all/counts_cell_all, [0.16,0.5,0.84])
        f_missin_K0p1_cell_qtls = np.nanquantile(counts_missin_K0p1_cell_all/counts_cell_all, [0.16,0.5,0.84])
        f_missin_K1_cell_qtls = np.nanquantile(counts_missin_K1_cell_all/counts_cell_all, [0.16,0.5,0.84])
        f_missin_grid[j,i] = f_missin_cell_qtls[1]
        f_missin_K0p1_grid[j,i] = f_missin_K0p1_cell_qtls[1]
        f_missin_K1_grid[j,i] = f_missin_K1_cell_qtls[1]
        
        #snum = r'${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'.format(np.round(f_missin_cell_qtls[1], 2), np.round(f_missin_cell_qtls[1]-f_missin_cell_qtls[0], 2), np.round(f_missin_cell_qtls[2]-f_missin_cell_qtls[1], 2))
        snum = r'${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'.format(np.round(f_missin_K1_cell_qtls[1], 2), np.round(f_missin_K1_cell_qtls[1]-f_missin_K1_cell_qtls[0], 2), np.round(f_missin_K1_cell_qtls[2]-f_missin_K1_cell_qtls[1], 2))
        plt.text(x=0.01+i*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.09, s=snum, ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
        #plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.015, s=r'${:.1f}$'.format(np.round(counts_cell_qtls[1], 1)), ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.025, s=r'${:.2f}$'.format(np.round(np.sum(counts_missin_cell_all)/np.sum(counts_cell_all), 2)), ha='right', va='top', color='maroon', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.075, s=r'${:.2f}$'.format(np.round(np.sum(counts_missin_K0p1_cell_all)/np.sum(counts_cell_all), 2)), ha='right', va='top', color='darkblue', fontsize=sfs, transform=ax.transAxes)
        #plt.text(x=-0.01+(i+1)*(1./n_P_bins), y=(j+1)*(1./n_R_bins)-0.135, s=r'${:.2f}$'.format(np.round(np.sum(counts_missin_K1_cell_all)/np.sum(counts_cell_all), 2)), ha='right', va='top', color='m', fontsize=sfs, transform=ax.transAxes)
img = plt.imshow(f_missin_K1_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower') #cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Period of conditioned planet, $P_{\rm cond}$ (days)', fontsize=tfs)
plt.ylabel(r'Radius of conditioned planet, $R_{p,\rm cond}$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$f_{\rm miss,in,K > 1m/s}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.1, y=0.5, s='(1)', ha='left', va='center', color='k', fontsize=mfs, transform=ax.transAxes)
plt.text(x=0.9, y=0.9, s='(2)', ha='right', va='top', color='maroon', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.9, y=0.5, s='(3)', ha='right', va='center', color='darkblue', fontsize=sfs, transform=ax.transAxes)
#plt.text(x=0.9, y=0.5, s='(4)', ha='right', va='top', color='g', fontsize=sfs, transform=ax.transAxes)
#plt.text(x=0.9, y=0.3, s='(5)', ha='right', va='top', color='m', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0, y=-0.5, s=r'(1) $f_{\rm miss,in,K > 1m/s}$', ha='left', va='center', color='k', fontsize=lfs, transform=ax.transAxes)
plt.text(x=0, y=-0.8, s=r'(2) $f_{\rm miss,in}$', ha='left', va='center', color='maroon', fontsize=lfs, transform=ax.transAxes)
plt.text(x=0, y=-1.1, s=r'(3) $f_{\rm miss,in,K > 0.1m/s}$', ha='left', va='center', color='darkblue', fontsize=lfs, transform=ax.transAxes)
#plt.text(x=0, y=-1.25, s=r'(4) $N_{\rm miss,in,K > 0.1m/s}$', ha='left', va='center', color='g', fontsize=lfs, transform=ax.transAxes)
#plt.text(x=0, y=-1.5, s=r'(5) $N_{\rm miss,in,K > 1m/s}$', ha='left', va='center', color='m', fontsize=lfs, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_fraction_missed_interior.pdf')
    plt.close()
plt.show()
