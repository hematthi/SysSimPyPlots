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
from src.functions_compute_RVs import *





##### To load the underlying and observed populations:

savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Conditional_transiting_PR_grid/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/PR_grids/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number





##### To plot a period-radius diagram with the number of observations required to measure K_cond in each bin:

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

mfs = 16 # main numbers font size
sfs = 12 # secondary numbers font size

# Period-radius grids (custom bins):
P_bins = np.array([4., 8., 16., 32., 64., 128., 256.])
R_bins = np.array([0.5, 1., 1.5, 2., 3., 4., 6.])
n_P_bins, n_R_bins = len(P_bins)-1, len(R_bins)-1

# Specify edges of GridSpec panels to ensure that the legend is the same size as a cell:
bgrid, tgrid = 0.1, 0.95 # bottom and top of grid
lgrid, rleg, wcb = 0.08, 0.98, 0.1 # left of grid, width of space for colorbar, and right of legend
#rgrid = (rleg-lgrid-wcb)*(n_P_bins/(n_P_bins+1)) + lgrid
rgrid = (rleg-lgrid-wcb) + lgrid
lleg = rgrid + wcb
bleg, tleg = tgrid - (tgrid-bgrid)/n_R_bins, tgrid



def compute_string_quantiles(x, nan_above=1000, qtls=[0.16,0.5,0.84]):
    lenx = len(x)
    x_sorted = np.sort(x)
    
    x_qtls = x_sorted[[int(np.round(q*lenx)) for q in qtls]]
    if np.isnan(x_qtls[1]): # Case 1: median > nan_above
        if np.isnan(x_qtls[0]): # Case 1a: all quantiles > nan_above
            snum = r'$>{:d}$'.format(nan_above)
        else: # Case 1b: lower quantile <= nan_above
            snum = r'$>{:d}_{{-{:d}}}$'.format(nan_above, int(nan_above-x_qtls[0]))
    else: # Case 2: median <= nan_above
        if np.isnan(x_qtls[2]): # Case 2a: upper quantile > nan_above
            snum = r'${:d}_{{-{:d}}}$'.format(int(x_qtls[1]), int(x_qtls[1]-x_qtls[0]))
        else: # Case 2b: all quantiles <= nan_above
            snum = r'${:d}_{{-{:d}}}^{{+{:d}}}$'.format(int(x_qtls[1]), int(x_qtls[1]-x_qtls[0]), int(x_qtls[2]-x_qtls[1]))
    
    return snum, x_qtls

N_sample, repeat = 1000, 100
σ_1obs = 0.3

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=lgrid,bottom=bgrid,right=rgrid,top=tgrid)

ax = plt.subplot(plot[:,:])
N_obs_med_grid = np.zeros((n_R_bins, n_P_bins))
for j in range(n_R_bins):
    for i in range(n_P_bins):
        subdir = 'Conditional_P%s_%sd_R%s_%s_transiting/' % (P_bins[i], P_bins[i+1], R_bins[j], R_bins[j+1])
        try:
            fname = 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p3_cases.txt' % (N_sample, repeat)
            outputs = np.genfromtxt(loadfiles_directory + subdir + fname, names=True)
            print('Loaded (1000): P_cond = [%s,%s], Rp_cond = [%s,%s]' % (P_bins[i], P_bins[i+1], R_bins[j], R_bins[j+1]))
            
            has_ideal = True
        except:
            try:
                fname = 'RV_obs_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (N_sample, repeat)
                outputs = np.genfromtxt(loadfiles_directory + subdir + fname, names=True)
                print('Loaded (300): P_cond = [%s,%s], Rp_cond = [%s,%s]' % (P_bins[i], P_bins[i+1], R_bins[j], R_bins[j+1]))
                
                has_ideal = False
            except:
                print('No simulation file: P_cond = [%s,%s], Rp_cond = [%s,%s]' % (P_bins[i], P_bins[i+1], R_bins[j], R_bins[j+1]))
                continue
        
        #N_obs_qtls = np.nanquantile(outputs['N_obs_min_20p'], [0.16,0.5,0.84])
        #N_obs_med_grid[j,i] = N_obs_qtls[1]
        #snum = r'${:d}_{{-{:d}}}^{{+{:d}}}$'.format(int(N_obs_qtls[1]), int(N_obs_qtls[1]-N_obs_qtls[0]), int(N_obs_qtls[2]-N_obs_qtls[1]))
        
        snum, N_obs_qtls = compute_string_quantiles(outputs['N_obs_min_20p'])
        N_obs_med_grid[j,i] = N_obs_qtls[1]

        if has_ideal:
            snum_ideal, N_obs_ideal_qtls = compute_string_quantiles(outputs['N_obs_min_20p_ideal'])
            #N_obs_med_grid[j,i] = N_obs_ideal_qtls[1]
            plt.text(x=(i+0.95)*(1./n_P_bins), y=(j+0.8)*(1./n_R_bins), s=snum_ideal, ha='right', va='center', color='b', fontsize=sfs, transform=ax.transAxes)
        plt.text(x=(i+0.4)*(1./n_P_bins), y=(j+0.5)*(1./n_R_bins), s=snum, ha='center', va='center', color='k', fontsize=mfs, fontweight='bold', transform=ax.transAxes)
cmap = matplotlib.cm.YlOrRd
cmap.set_bad('w')
img = plt.imshow(N_obs_med_grid, cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=5., vmax=1000.), aspect='auto', interpolation="nearest", origin='lower') # cmap='coolwarm'
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.linspace(-0.5, n_P_bins-0.5, n_P_bins+1), P_bins)
plt.yticks(np.linspace(-0.5, n_R_bins-0.5, n_R_bins+1), R_bins)
plt.xlabel(r'Period of conditioned planet, $P_{\rm cond}$ (days)', fontsize=tfs)
plt.ylabel(r'Radius of conditioned planet, $R_{p,\rm cond}$ ($R_\oplus$)', fontsize=tfs)

plot = GridSpec(1,1,left=rgrid+0.01,bottom=bgrid,right=rgrid+0.03,top=tgrid) # colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [5,10,20,50,100,200,300,500,1000]
cbar = plt.colorbar(img, cax=cax, ticks=cticks_custom, format=ticker.ScalarFormatter())
cbar.ax.tick_params(labelsize=lfs)
cbar.set_label(r'Median $N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', rotation=270, va='bottom', fontsize=tfs)

#plot = GridSpec(1,1,left=lleg,bottom=bleg,right=rleg,top=tleg) # legend
#ax = plt.subplot(plot[:,:])
#plt.text(x=0.5, y=0.5, s=r'$N_{\rm obs}$', ha='center', va='center', color='k', fontsize=mfs, transform=ax.transAxes) # fontweight='bold'
#plt.xticks([])
#plt.yticks([])
#plt.xlabel('Legend', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_PR_grid_Nobs_sigma0p3.pdf')
    plt.close()
plt.show()
