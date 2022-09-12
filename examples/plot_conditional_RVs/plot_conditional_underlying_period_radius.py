# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
from syssimpyplots.compute_RVs import *





##### To load the underlying and observed populations:

savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Conditional_P8_12d_R1p5_2_transiting/' #'Conditional_Venus/' #'Conditional_P8_12d_R1p5_2_transiting/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Systems_conditional/Conditional_P8_12d_R1p5_2_transiting/' #'Conditional_Venus/' #'Conditional_P8_12d_R1p5_2_transiting/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [8.,11.3137], [1.5874,2.0], [0.,np.inf]
#P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [8.,12.], [1.8,2.0], [0.,np.inf]
#P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [8.,12.], [0.9,1.1], [0.,np.inf]
#P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [8.,12.], [3.,4.], [0.,np.inf]
#P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [215.,235.], [0.9,1.0], [0.77,0.86] # Venus
det = True # set False for Venus
conds = conditionals_dict(P_cond_bounds=P_cond_bounds, Rp_cond_bounds=Rp_cond_bounds, Mp_cond_bounds=Mp_cond_bounds, det=det)

n_per_sys = sssp_per_sys['Mtot_all']
bools_cond_per_sys = condition_planets_bools_per_sys(sssp_per_sys, conds)
i_cond = condition_systems_indices(sssp_per_sys, conds)

n_sys_cond = len(i_cond)
P_all_cond = sssp_per_sys['P_all'][i_cond]
Rp_all_cond = sssp_per_sys['radii_all'][i_cond]
det_all_cond = sssp_per_sys['det_all'][i_cond]
bools_cond_all_cond = bools_cond_per_sys[i_cond]

# To also load in a full simulated catalog for normalizing the relative occurrence of planets in each bin:
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number = ''

param_vals_all_full = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys_full, sssp_full = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

n_sys_full = len(sssp_per_sys_full['Mtot_all'])
P_all_full = sssp_per_sys_full['P_all']
Rp_all_full = sssp_per_sys_full['radii_all']





##### To plot period-radius diagrams:

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

mfs = 12 # main numbers font size
sfs = 12 # secondary numbers font size

bins = 100





##### Scatter plot of period vs radius for the conditioned systems:

#P_max = 256. # if want to further truncate periods for plot
n_sys_plot = 1000
P_all_cond_plot = P_all_cond[:n_sys_plot]
Rp_all_cond_plot = Rp_all_cond[:n_sys_plot]
det_all_cond_plot = det_all_cond[:n_sys_plot]
bools_cond_all_cond_plot = bools_cond_all_cond[:n_sys_plot]

fig = plt.figure(figsize=(16,8))
plot = GridSpec(6,10,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[1:,:9])
# Contour plot:
#cmap = cm.get_cmap('viridis', 256)
corner.hist2d(np.log10(P_all_cond[(~bools_cond_all_cond) & (P_all_cond > 0) & (P_all_cond < P_max)]), np.log10(Rp_all_cond[(~bools_cond_all_cond) & (P_all_cond > 0) & (P_all_cond < P_max)]), bins=30, plot_datapoints=False, plot_density=False, fill_contours=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'}) #{'colors': ['0.6','0.4','0.2','0']}; {'colors': [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8)]}
# Scatter dots:
#plt.scatter(np.log10(P_all_cond_plot[bools_cond_all_cond_plot]), np.log10(Rp_all_cond_plot[bools_cond_all_cond_plot]), s=5, marker='.', c='g', label='Conditioned planets')
#plt.scatter(np.log10(P_all_cond_plot[(~bools_cond_all_cond_plot) & (P_all_cond_plot > 0) & (det_all_cond_plot == 1)]), np.log10(Rp_all_cond_plot[(~bools_cond_all_cond_plot) & (P_all_cond_plot > 0) & (det_all_cond_plot == 1)]), s=5, marker='.', c='b', label='Other planets in the conditioned systems')
# Scatter circles with/without outlines for detected/undetected planets:
sc11 = plt.scatter(np.log10(P_all_cond_plot[(bools_cond_all_cond_plot) & (det_all_cond_plot == 1)]), np.log10(Rp_all_cond_plot[(bools_cond_all_cond_plot) & (det_all_cond_plot == 1)]), s=10, marker='o', facecolors='g', edgecolors='g', label='Conditioned planets')
sc12 = plt.scatter(np.log10(P_all_cond_plot[(bools_cond_all_cond_plot) & (det_all_cond_plot == 0)]), np.log10(Rp_all_cond_plot[(bools_cond_all_cond_plot) & (det_all_cond_plot == 0)]), s=10, marker='o', facecolors='none', edgecolors='g')
sc21 = plt.scatter(np.log10(P_all_cond_plot[(~bools_cond_all_cond_plot) & (P_all_cond_plot > 0) & (det_all_cond_plot == 1)]), np.log10(Rp_all_cond_plot[(~bools_cond_all_cond_plot) & (P_all_cond_plot > 0) & (det_all_cond_plot == 1)]), s=10, marker='o', facecolors='b', edgecolors='b', label='Other planets in the conditioned systems')
sc22 = plt.scatter(np.log10(P_all_cond_plot[(~bools_cond_all_cond_plot) & (P_all_cond_plot > 0) & (det_all_cond_plot == 0)]), np.log10(Rp_all_cond_plot[(~bools_cond_all_cond_plot) & (P_all_cond_plot > 0) & (det_all_cond_plot == 0)]), s=10, marker='o', facecolors='none', edgecolors='b')
ax.tick_params(axis='both', labelsize=20)
xtick_vals = np.array([3,10,30,100,300])
ytick_vals = np.array([0.5,1,2,4,8])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(P_min), np.log10(P_max)])
plt.ylim([np.log10(radii_min), np.log10(radii_max)])
plt.xlabel(r'Orbital period $P$ (days)', fontsize=20)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=20)
legend1 = plt.legend([sc11, sc21], ['Conditioned planets', 'Other planets in the conditioned systems'], loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs) # for conditioned/other planets (colors)
#plt.legend([sc21, sc22], ['Kepler-detected', 'Kepler-undetected'], loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs) # for detected/undetected planets (markers)
#plt.gca().add_artist(legend1)

ax = plt.subplot(plot[0,:9]) # top histogram
x_cond = P_all_cond[(~bools_cond_all_cond) & (P_all_cond > 0)]
plt.hist(x_cond, bins=np.logspace(np.log10(P_min), np.log10(P_max), bins+1), weights=np.ones(len(x_cond))/len(x_cond), histtype='step', color='b', ls='-', label='Other planets in the conditioned systems')
plt.hist(sssp_full['P_all'], bins=np.logspace(np.log10(P_min), np.log10(P_max), bins+1), weights=np.ones(len(sssp_full['P_all']))/len(sssp_full['P_all']), histtype='step', color='k', ls='-', label='Underlying distribution (all systems)')
plt.gca().set_xscale("log")
plt.xlim([P_min, P_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,9]) # side histogram
x_cond = Rp_all_cond[(~bools_cond_all_cond) & (P_all_cond > 0)]
plt.hist(x_cond, bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), weights=np.ones(len(x_cond))/len(x_cond), histtype='step', orientation='horizontal', color='b', ls='-')
plt.hist(sssp_full['radii_all'], bins=np.logspace(np.log10(radii_min), np.log10(radii_max), bins+1), weights=np.ones(len(sssp_full['radii_all']))/len(sssp_full['radii_all']), histtype='step', orientation='horizontal', color='k', ls='-')
plt.gca().set_yscale("log")
plt.ylim([radii_min, radii_max])
plt.xticks([])
plt.yticks([])

if savefigures:
    fig_name = savefigures_directory + model_name + '_P%s_%s_R%s_%s_cond_PR_loglog.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1])
    plt.savefig(fig_name)
    plt.close()





##### Period-radius grids (custom bins):

#P_bins = np.logspace(np.log10(P_min), np.log10(P_max), 5+1)
#R_bins = np.array([0.5, 1., 1.5, 2., 3., 5., 10.])
#P_bins = np.array([4., 8., 16., 32., 64., 128., 256.])
#R_bins = np.array([0.5, 1., 1.5, 2., 3., 4., 6.])
P_bins = np.logspace(np.log10(4.), np.log10(256.), 12+1)
R_bins = np.logspace(np.log10(0.5), np.log10(8.), 12+1) #np.arange(0.5, 6.1, 0.5)
n_P_bins, n_R_bins = len(P_bins)-1, len(R_bins)-1

# Specify edges of GridSpec panels to ensure that the legend is the same size as a cell:
bgrid, tgrid = 0.1, 0.9 # bottom and top of grid
lgrid, rleg, wcb = 0.08, 0.97, 0.09 # left of grid, width of space for colorbar, and right of legend
rgrid = (rleg-lgrid-wcb)*(n_P_bins/(n_P_bins+1)) + lgrid
lleg = rgrid + wcb
bleg, tleg = tgrid - (tgrid-bgrid)/n_R_bins, tgrid



# First, plot overall occurrence rates (all systems):

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)
plt.figtext(0.5, 0.95, r'Occurrence rates', va='center', ha='center', fontsize=tfs)

ax = plt.subplot(plot[:,:])
mpps_grid = np.zeros((n_R_bins, n_P_bins))
mpps_dlnPR_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        dlnP, dlnR = np.log(P_bins[i+1]/P_bins[i]), np.log(R_bins[j+1]/R_bins[j])
        pl_cell_bools_full = (P_all_full > P_bins[i]) & (P_all_full < P_bins[i+1]) & (Rp_all_full > R_bins[j]) & (Rp_all_full < R_bins[j+1])
        pl_tot_cell_full = np.sum(pl_cell_bools_full)
        sys_cell_bools_full = np.any(pl_cell_bools_full, axis=1)
        sys_tot_cell_full = np.sum(sys_cell_bools_full)
        print('(i=%s,j=%s): n_pl (all) = %s/%s' % (i,j,pl_tot_cell_full,n_sys_full))

        mpps_cell_full = pl_tot_cell_full/n_sys_full # mean number of planets in bin per star, for all systems
        mpps_dlnPR_cell_full = mpps_cell_full/(dlnP*dlnR)
        mpps_grid[j,i] = mpps_cell_full
        mpps_dlnPR_grid[j,i] = mpps_dlnPR_cell_full

        plt.text(x=(i+0.95)*(1./n_P_bins), y=(j+0.95)*(1./n_R_bins), s=r'${:.3f}$'.format(np.round(mpps_cell_full, 3)), ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=(i+0.05)*(1./n_P_bins), y=(j+0.5)*(1./n_R_bins), s=r'${:.2f}$'.format(np.round(mpps_dlnPR_cell_full, 2)), ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(mpps_dlnPR_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower', extent=np.log10((P_bins[0], P_bins[-1], R_bins[0], R_bins[-1]))) #cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.log10(P_bins[::2]), ['{:.1f}'.format(x) for x in P_bins[::2]])
plt.yticks(np.log10(R_bins[::3]), ['{:.1f}'.format(x) for x in R_bins[::3]])
plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$\frac{\bar{n}_{\rm bin}}{d(\ln{R_p}) d(\ln{P})}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.95, y=0.9, s='(2)', ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.05, y=0.5, s='(1)', ha='left', va='center', color='k', fontsize=mfs, transform=ax.transAxes)
plt.text(x=-0.3, y=-1.5, s=r'(1) $\frac{\bar{n}_{\rm bin}}{d(\ln{R_p}) d(\ln{P})}$', ha='left', va='center', color='k', fontsize=lfs, transform=ax.transAxes)
plt.text(x=-0.3, y=-2.25, s=r'(2) $\bar{n}_{\rm bin}$', ha='left', va='center', color='b', fontsize=lfs, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    fig_name = savefigures_directory + model_name + '_all_PR_grid_rates.pdf'
    plt.savefig(fig_name)
    plt.close()

##### Remake for defense talk:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=bgrid,right=0.865,top=tgrid,wspace=0,hspace=0)
plt.figtext(0.5, 0.95, r'Occurrence rates over all systems', va='center', ha='center', fontsize=tfs)

ax = plt.subplot(plot[:,:])
mpps_grid = np.zeros((n_R_bins, n_P_bins))
mpps_dlnPR_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        dlnP, dlnR = np.log(P_bins[i+1]/P_bins[i]), np.log(R_bins[j+1]/R_bins[j])
        pl_cell_bools_full = (P_all_full > P_bins[i]) & (P_all_full < P_bins[i+1]) & (Rp_all_full > R_bins[j]) & (Rp_all_full < R_bins[j+1])
        pl_tot_cell_full = np.sum(pl_cell_bools_full)
        sys_cell_bools_full = np.any(pl_cell_bools_full, axis=1)
        sys_tot_cell_full = np.sum(sys_cell_bools_full)
        print('(i=%s,j=%s): n_pl (all) = %s/%s' % (i,j,pl_tot_cell_full,n_sys_full))

        mpps_cell_full = pl_tot_cell_full/n_sys_full # mean number of planets in bin per star, for all systems
        mpps_dlnPR_cell_full = mpps_cell_full/(dlnP*dlnR)
        mpps_grid[j,i] = mpps_cell_full
        mpps_dlnPR_grid[j,i] = mpps_dlnPR_cell_full

        plt.text(x=(i+0.5)*(1./n_P_bins), y=(j+0.5)*(1./n_R_bins), s=r'${:.3f}$'.format(np.round(mpps_cell_full, 3)), ha='center', va='center', color='k', fontsize=16, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(mpps_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", vmax=0.072, origin='lower', extent=np.log10((P_bins[0], P_bins[-1], R_bins[0], R_bins[-1]))) #cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.log10(P_bins[::2]), ['{:.1f}'.format(x) for x in P_bins[::2]])
plt.yticks(np.log10(R_bins[::3]), ['{:.1f}'.format(x) for x in R_bins[::3]])
plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=0.89,bottom=bgrid,right=0.92,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$\bar{n}_{\rm bin,all}$', rotation=270, va='bottom', fontsize=tfs)

if savefigures:
    fig_name = savefigures_directory + model_name + '_all_PR_grid_rates_simple.pdf'
    plt.savefig(fig_name)
    plt.close()



# Occurrence rates in the conditioned systems:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)
plt.figtext(0.5, 0.95, r'Occurrence rates conditioned on a planet in $P = [{:.1f},{:.1f}]$d, $R_p = [{:.2f},{:.2f}] R_\oplus$'.format(conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper']), va='center', ha='center', fontsize=tfs)
#plt.figtext(0.5, 0.95, r'Occurrence rates conditioned on a Venus-like planet', va='center', ha='center', fontsize=tfs)

ax = plt.subplot(plot[:,:])
mpps_grid = np.zeros((n_R_bins, n_P_bins))
mpps_dlnPR_grid = np.zeros((n_R_bins, n_P_bins))
fswp_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        dlnP, dlnR = np.log(P_bins[i+1]/P_bins[i]), np.log(R_bins[j+1]/R_bins[j])
        pl_cell_bools = (P_all_cond > P_bins[i]) & (P_all_cond < P_bins[i+1]) & (Rp_all_cond > R_bins[j]) & (Rp_all_cond < R_bins[j+1]) & (~bools_cond_all_cond) # last condition is to NOT count the conditioned planets themselves
        pl_tot_cell = np.sum(pl_cell_bools)
        sys_cell_bools = np.any(pl_cell_bools, axis=1)
        sys_tot_cell = np.sum(sys_cell_bools)

        mpps_cell = pl_tot_cell/n_sys_cond # mean number of planets in bin per star, for conditioned systems
        mpps_dlnPR_cell = mpps_cell/(dlnP*dlnR)
        fswp_cell = sys_tot_cell/n_sys_cond # fraction of stars with planets in bin, for conditioned systems
        mpps_grid[j,i] = mpps_cell
        mpps_dlnPR_grid[j,i] = mpps_dlnPR_cell
        fswp_grid[j,i] = fswp_cell

        plt.text(x=(i+0.95)*(1./n_P_bins), y=(j+0.95)*(1./n_R_bins), s=r'${:.3f}$'.format(np.round(mpps_cell, 3)), ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=(i+0.05)*(1./n_P_bins), y=(j+0.5)*(1./n_R_bins), s=r'${:.2f}$'.format(np.round(mpps_dlnPR_cell, 2)), ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(mpps_dlnPR_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower', extent=np.log10((P_bins[0], P_bins[-1], R_bins[0], R_bins[-1]))) #cmap='coolwarm'
box_cond = patches.Rectangle(np.log10((conds['P_lower'], conds['Rp_lower'])), np.log10(conds['P_upper']/conds['P_lower']), np.log10(conds['Rp_upper']/conds['Rp_lower']), linewidth=2, edgecolor='g', facecolor='none')
ax.add_patch(box_cond)
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.log10(P_bins[::2]), ['{:.1f}'.format(x) for x in P_bins[::2]])
plt.yticks(np.log10(R_bins[::3]), ['{:.1f}'.format(x) for x in R_bins[::3]])
plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$\frac{\bar{n}_{\rm bin}}{d(\ln{R_p}) d(\ln{P})}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.95, y=0.9, s='(2)', ha='right', va='top', color='b', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.05, y=0.5, s='(1)', ha='left', va='center', color='k', fontsize=mfs, transform=ax.transAxes)
plt.text(x=-0.3, y=-1.5, s=r'(1) $\frac{\bar{n}_{\rm bin}}{d(\ln{R_p}) d(\ln{P})}$', ha='left', va='center', color='k', fontsize=lfs, transform=ax.transAxes)
plt.text(x=-0.3, y=-2.25, s=r'(2) $\bar{n}_{\rm bin}$', ha='left', va='center', color='b', fontsize=lfs, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    fig_name = savefigures_directory + model_name + '_P%s_%s_R%s_%s_cond_PR_grid_rates.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1])
    plt.savefig(fig_name)
    plt.close()

##### Remake for defense talk:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=bgrid,right=0.865,top=tgrid,wspace=0,hspace=0)
plt.figtext(0.5, 0.95, r'Occurrence rates conditioned on a planet in $P = [{:.1f},{:.1f}]$d, $R_p = [{:.2f},{:.2f}] R_\oplus$'.format(conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper']), va='center', ha='center', fontsize=tfs)
#plt.figtext(0.5, 0.95, r'Occurrence rates conditioned on a Venus-like planet', va='center', ha='center', fontsize=tfs)

ax = plt.subplot(plot[:,:])
mpps_grid = np.zeros((n_R_bins, n_P_bins))
mpps_dlnPR_grid = np.zeros((n_R_bins, n_P_bins))
fswp_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        dlnP, dlnR = np.log(P_bins[i+1]/P_bins[i]), np.log(R_bins[j+1]/R_bins[j])
        pl_cell_bools = (P_all_cond > P_bins[i]) & (P_all_cond < P_bins[i+1]) & (Rp_all_cond > R_bins[j]) & (Rp_all_cond < R_bins[j+1]) & (~bools_cond_all_cond) # last condition is to NOT count the conditioned planets themselves
        pl_tot_cell = np.sum(pl_cell_bools)
        sys_cell_bools = np.any(pl_cell_bools, axis=1)
        sys_tot_cell = np.sum(sys_cell_bools)

        mpps_cell = pl_tot_cell/n_sys_cond # mean number of planets in bin per star, for conditioned systems
        mpps_dlnPR_cell = mpps_cell/(dlnP*dlnR)
        fswp_cell = sys_tot_cell/n_sys_cond # fraction of stars with planets in bin, for conditioned systems
        mpps_grid[j,i] = mpps_cell
        mpps_dlnPR_grid[j,i] = mpps_dlnPR_cell
        fswp_grid[j,i] = fswp_cell

        plt.text(x=(i+0.5)*(1./n_P_bins), y=(j+0.5)*(1./n_R_bins), s=r'${:.3f}$'.format(np.round(mpps_cell, 3)), ha='center', va='center', color='k', fontsize=16, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(mpps_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", vmax=0.072, origin='lower', extent=np.log10((P_bins[0], P_bins[-1], R_bins[0], R_bins[-1]))) #cmap='coolwarm'
box_cond = patches.Rectangle(np.log10((conds['P_lower'], conds['Rp_lower'])), np.log10(conds['P_upper']/conds['P_lower']), np.log10(conds['Rp_upper']/conds['Rp_lower']), linewidth=2, edgecolor='g', facecolor='none')
ax.add_patch(box_cond)
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.log10(P_bins[::2]), ['{:.1f}'.format(x) for x in P_bins[::2]])
plt.yticks(np.log10(R_bins[::3]), ['{:.1f}'.format(x) for x in R_bins[::3]])
plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=0.89,bottom=bgrid,right=0.92,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$\bar{n}_{\rm bin,cond}$', rotation=270, va='bottom', fontsize=tfs)

if savefigures:
    fig_name = savefigures_directory + model_name + '_P%s_%s_R%s_%s_cond_PR_grid_rates_simple.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1])
    plt.savefig(fig_name)
    plt.close()





# Relative occurrence rates (intrinsic mean numbers of planets in conditioned systems vs. in general) in each bin:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)
plt.figtext(0.5, 0.95, r'Relative occurrence rates conditioned on a planet in $P = [{:.1f},{:.1f}]$d, $R_p = [{:.2f},{:.2f}] R_\oplus$'.format(conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper']), va='center', ha='center', fontsize=tfs)
#plt.figtext(0.5, 0.95, r'Relative occurrence rates conditioned on a Venus-like planet', va='center', ha='center', fontsize=tfs)

ax = plt.subplot(plot[:,:])
rel_mpps_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        dlnP, dlnR = np.log(P_bins[i+1]/P_bins[i]), np.log(R_bins[j+1]/R_bins[j])
        pl_cell_bools = (P_all_cond > P_bins[i]) & (P_all_cond < P_bins[i+1]) & (Rp_all_cond > R_bins[j]) & (Rp_all_cond < R_bins[j+1]) & (~bools_cond_all_cond) # last condition is to NOT count the conditioned planets themselves
        pl_tot_cell = np.sum(pl_cell_bools)
        sys_cell_bools = np.any(pl_cell_bools, axis=1)
        sys_tot_cell = np.sum(sys_cell_bools)

        pl_cell_bools_full = (P_all_full > P_bins[i]) & (P_all_full < P_bins[i+1]) & (Rp_all_full > R_bins[j]) & (Rp_all_full < R_bins[j+1])
        pl_tot_cell_full = np.sum(pl_cell_bools_full)
        sys_cell_bools_full = np.any(pl_cell_bools_full, axis=1)
        sys_tot_cell_full = np.sum(sys_cell_bools_full)
        print('(i=%s,j=%s): n_pl (cond) = %s/%s, n_pl (all) = %s/%s' % (i,j,pl_tot_cell,n_sys_cond,pl_tot_cell_full,n_sys_full))

        mpps_cell = pl_tot_cell/n_sys_cond # mean number of planets in bin per star, for conditioned systems
        mpps_dlnPR_cell = mpps_cell/(dlnP*dlnR)
        mpps_cell_full = pl_tot_cell_full/n_sys_full # mean number of planets in bin per star, for all systems
        mpps_dlnPR_cell_full = mpps_cell_full/(dlnP*dlnR)
        rel_mpps_grid[j,i] = mpps_cell/mpps_cell_full

        plt.text(x=(i+0.95)*(1./n_P_bins), y=(j+0.7)*(1./n_R_bins), s=r'${:.2f}$'.format(np.round(mpps_dlnPR_cell, 2)), ha='right', va='center', color='b', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=(i+0.95)*(1./n_P_bins), y=(j+0.3)*(1./n_R_bins), s=r'${:.2f}$'.format(np.round(mpps_dlnPR_cell_full, 2)), ha='right', va='center', color='r', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=(i+0.05)*(1./n_P_bins), y=(j+0.5)*(1./n_R_bins), s=r'${:.2f}$'.format(np.round(mpps_cell/mpps_cell_full, 2)), ha='left', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(rel_mpps_grid, cmap='coolwarm', norm=MidPointLogNorm(vmin=0.1,vmax=10.,midpoint=1.), aspect='auto', interpolation="nearest", origin='lower', extent=np.log10((P_bins[0], P_bins[-1], R_bins[0], R_bins[-1]))) # log colorscale; norm=matplotlib.colors.LogNorm(vmin=0.1,vmax=10.); MidPointLogNorm(vmin=0.1,vmax=10.,midpoint=1.)
#img = plt.imshow(rel_mpps_grid, cmap='coolwarm', norm=matplotlib.colors.TwoSlopeNorm(vcenter=1.), aspect='auto', interpolation="nearest", origin='lower', extent=np.log10((P_bins[0], P_bins[-1], R_bins[0], R_bins[-1]))) # linear colorscale
box_cond = patches.Rectangle(np.log10((conds['P_lower'], conds['Rp_lower'])), np.log10(conds['P_upper']/conds['P_lower']), np.log10(conds['Rp_upper']/conds['Rp_lower']), linewidth=2, edgecolor='g', facecolor='none')
ax.add_patch(box_cond)
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.log10(P_bins[::2]), ['{:.1f}'.format(x) for x in P_bins[::2]])
plt.yticks(np.log10(R_bins[::3]), ['{:.1f}'.format(x) for x in R_bins[::3]])
plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [0.1,0.2,0.5,1.,2.,5.,10.] # for LogNorm color scale only
cbar = plt.colorbar(img, cax=cax, ticks=cticks_custom, format=ticker.ScalarFormatter())
#cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$\bar{n}_{\rm bin,cond}/\bar{n}_{\rm bin,all}$', rotation=270, va='bottom', fontsize=tfs)

plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
ax = plt.subplot(plot[:,:])
plt.text(x=0.95, y=0.7, s='(2)', ha='right', va='center', color='b', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.95, y=0.3, s='(3)', ha='right', va='center', color='r', fontsize=sfs, transform=ax.transAxes)
plt.text(x=0.05, y=0.5, s='(1)', ha='left', va='center', color='k', fontsize=mfs, transform=ax.transAxes)
plt.text(x=-0.3, y=-1.5, s=r'(1) $\frac{\bar{n}_{\rm bin,cond}}{\bar{n}_{\rm bin,all}}$', ha='left', va='center', color='k', fontsize=lfs, transform=ax.transAxes)
plt.text(x=-0.3, y=-2.25, s=r'(2) $\frac{\bar{n}_{\rm bin,cond}}{d(\ln{R_p}) d(\ln{P})}$', ha='left', va='center', color='b', fontsize=lfs, transform=ax.transAxes)
plt.text(x=-0.3, y=-3, s=r'(3) $\frac{\bar{n}_{\rm bin,all}}{d(\ln{R_p}) d(\ln{P})}$', ha='left', va='center', color='r', fontsize=lfs, transform=ax.transAxes)
plt.xticks([])
plt.yticks([])
plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    fig_name = savefigures_directory + model_name + '_P%s_%s_R%s_%s_cond_PR_grid_rates_ratios_logscale_extra.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1])
    plt.savefig(fig_name)
    plt.close()
plt.show()

##### Remake relative occurrence rates for paper version:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=bgrid,right=0.865,top=tgrid,wspace=0,hspace=0)
plt.figtext(0.5, 0.95, r'Relative occurrence rates conditioned on a planet in $P = [{:.1f},{:.1f}]$d, $R_p = [{:.2f},{:.2f}] R_\oplus$'.format(conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper']), va='center', ha='center', fontsize=tfs)
#plt.figtext(0.5, 0.95, r'Relative occurrence rates conditioned on a Venus-like planet', va='center', ha='center', fontsize=tfs)

ax = plt.subplot(plot[:,:])
rel_mpps_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        dlnP, dlnR = np.log(P_bins[i+1]/P_bins[i]), np.log(R_bins[j+1]/R_bins[j])
        pl_cell_bools = (P_all_cond > P_bins[i]) & (P_all_cond < P_bins[i+1]) & (Rp_all_cond > R_bins[j]) & (Rp_all_cond < R_bins[j+1]) & (~bools_cond_all_cond) # last condition is to NOT count the conditioned planets themselves
        pl_tot_cell = np.sum(pl_cell_bools)
        sys_cell_bools = np.any(pl_cell_bools, axis=1)
        sys_tot_cell = np.sum(sys_cell_bools)

        pl_cell_bools_full = (P_all_full > P_bins[i]) & (P_all_full < P_bins[i+1]) & (Rp_all_full > R_bins[j]) & (Rp_all_full < R_bins[j+1])
        pl_tot_cell_full = np.sum(pl_cell_bools_full)
        sys_cell_bools_full = np.any(pl_cell_bools_full, axis=1)
        sys_tot_cell_full = np.sum(sys_cell_bools_full)
        print('(i=%s,j=%s): n_pl (cond) = %s/%s, n_pl (all) = %s/%s' % (i,j,pl_tot_cell,n_sys_cond,pl_tot_cell_full,n_sys_full))

        mpps_cell = pl_tot_cell/n_sys_cond # mean number of planets in bin per star, for conditioned systems
        mpps_dlnPR_cell = mpps_cell/(dlnP*dlnR)
        mpps_cell_full = pl_tot_cell_full/n_sys_full # mean number of planets in bin per star, for all systems
        mpps_dlnPR_cell_full = mpps_cell_full/(dlnP*dlnR)
        rel_mpps_grid[j,i] = mpps_cell/mpps_cell_full

        mpps_ratio_cell = mpps_cell/mpps_cell_full
        #snum = r'${:.3f}$'.format(np.round(mpps_ratio_cell, 3)) if mpps_ratio_cell < 0.005 else r'${:.2f}$'.format(np.round(mpps_ratio_cell, 2))
        snum = r'${:.2g}$'.format(mpps_ratio_cell) if mpps_ratio_cell < 0.1 else r'${:.2f}$'.format(np.round(mpps_ratio_cell, 2))
        plt.text(x=(i+0.5)*(1./n_P_bins), y=(j+0.5)*(1./n_R_bins), s=snum, ha='center', va='center', color='k', fontsize=16, fontweight='bold', transform=ax.transAxes)
img = plt.imshow(rel_mpps_grid, cmap='coolwarm', norm=MidPointLogNorm(vmin=0.1,vmax=10.,midpoint=1.), aspect='auto', interpolation="nearest", origin='lower', extent=np.log10((P_bins[0], P_bins[-1], R_bins[0], R_bins[-1]))) # log colorscale; norm=matplotlib.colors.LogNorm(vmin=0.1,vmax=10.); MidPointLogNorm(vmin=0.1,vmax=10.,midpoint=1.)
#img = plt.imshow(rel_mpps_grid, cmap='coolwarm', norm=matplotlib.colors.TwoSlopeNorm(vcenter=1.), aspect='auto', interpolation="nearest", origin='lower', extent=np.log10((P_bins[0], P_bins[-1], R_bins[0], R_bins[-1]))) # linear colorscale
box_cond = patches.Rectangle(np.log10((conds['P_lower'], conds['Rp_lower'])), np.log10(conds['P_upper']/conds['P_lower']), np.log10(conds['Rp_upper']/conds['Rp_lower']), linewidth=2, edgecolor='g', facecolor='none')
ax.add_patch(box_cond)
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.log10(P_bins[::2]), ['{:.1f}'.format(x) for x in P_bins[::2]])
plt.yticks(np.log10(R_bins[::3]), ['{:.1f}'.format(x) for x in R_bins[::3]])
plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=0.89,bottom=bgrid,right=0.92,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [0.1,0.2,0.5,1.,2.,5.,10.] # for LogNorm color scale only
cbar = plt.colorbar(img, cax=cax, ticks=cticks_custom, format=ticker.ScalarFormatter())
#cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'$\bar{n}_{\rm bin,cond}/\bar{n}_{\rm bin,all}$', rotation=270, va='bottom', fontsize=tfs)

if savefigures:
    fig_name = savefigures_directory + model_name + '_P%s_%s_R%s_%s_cond_PR_grid_rates_ratios_logscale.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1])
    plt.savefig(fig_name)
    plt.close()
plt.show()

# To make a machine-readable table file with the relative occurrence rates:
'''
fname = savefigures_directory + 'Venus_relrates_Fig11.txt'
file_comments = '# Conditioned on systems with planets in: P=[%s,%s] days, R=[%s,%s] R_earth, M=[%s,%s] M_earth\n' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], Mp_cond_bounds[0], Mp_cond_bounds[1])
file_comments += '# periods in days; radii in Earth units\n'
file_comments += '# mpps_cond_dlnPR = mean number of planets per star in conditioned systems, divided by natural logs of the bin widths\n'
file_comments += '# mpps_dlnPR = mean number of planets per star in all systems, divided by natural logs of the bin widths\n'
file_comments += '# mpps_cond_ratio = mpps_cond_dlnPR / mpps_dlnPR\n\n'
table_headers = ['period_min', 'period_max', 'radius_min', 'radius_max', 'mpps_cond_dlnPR', 'mpps_dlnPR', 'mpps_cond_ratio']
table_fmts = ['%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.4f', '%1.4f', '%1.4f']
table_array = []
for i in range(n_P_bins): # NOTE that order of for-loops are switched compared to the above! (But i,j labels are the same)
    for j in range(n_R_bins):
        dlnP, dlnR = np.log(P_bins[i+1]/P_bins[i]), np.log(R_bins[j+1]/R_bins[j])
        pl_cell_bools = (P_all_cond > P_bins[i]) & (P_all_cond < P_bins[i+1]) & (Rp_all_cond > R_bins[j]) & (Rp_all_cond < R_bins[j+1]) & (~bools_cond_all_cond) # last condition is to NOT count the conditioned planets themselves
        pl_tot_cell = np.sum(pl_cell_bools)
        sys_cell_bools = np.any(pl_cell_bools, axis=1)
        sys_tot_cell = np.sum(sys_cell_bools)

        pl_cell_bools_full = (P_all_full > P_bins[i]) & (P_all_full < P_bins[i+1]) & (Rp_all_full > R_bins[j]) & (Rp_all_full < R_bins[j+1])
        pl_tot_cell_full = np.sum(pl_cell_bools_full)
        sys_cell_bools_full = np.any(pl_cell_bools_full, axis=1)
        sys_tot_cell_full = np.sum(sys_cell_bools_full)
        print('(i=%s,j=%s): n_pl (cond) = %s/%s, n_pl (all) = %s/%s' % (i,j,pl_tot_cell,n_sys_cond,pl_tot_cell_full,n_sys_full))

        mpps_cell = pl_tot_cell/n_sys_cond # mean number of planets in bin per star, for conditioned systems
        mpps_dlnPR_cell = mpps_cell/(dlnP*dlnR)
        mpps_cell_full = pl_tot_cell_full/n_sys_full # mean number of planets in bin per star, for all systems
        mpps_dlnPR_cell_full = mpps_cell_full/(dlnP*dlnR)
        mpps_ratio_cell = mpps_cell/mpps_cell_full

        table_array.append([P_bins[i], P_bins[i+1], R_bins[j], R_bins[j+1], mpps_dlnPR_cell, mpps_dlnPR_cell_full, mpps_ratio_cell])
table_array = np.array(table_array)
###np.savetxt(fname, table_array, fmt=table_fmts, header=file_comments + ' '.join(table_headers))
'''
