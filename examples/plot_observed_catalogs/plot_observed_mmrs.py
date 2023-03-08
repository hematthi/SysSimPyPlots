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

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
#savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/'
savefigures_directory = loadfiles_directory
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

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)





##### To plot the simulated and Kepler catalogs:

subdirectory = ''

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 3 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





##### To plot a histogram of the zeta statistic (similar to Fig 5 in Fabrycky et al. 2014):

Rm_max = 2.5 # should be 2.5 for zeta1 and 4 for zeta2
pratios_small_sim = sss['Rm_obs'][sss['Rm_obs'] < Rm_max]
pratios_small_Kep = ssk['Rm_obs'][ssk['Rm_obs'] < Rm_max]
zeta1_small_sim = zeta(pratios_small_sim)
zeta1_small_Kep = zeta(pratios_small_Kep)
zeta2_small_sim = zeta(pratios_small_sim, order=2) # DON'T TRUST THESE YET!
zeta2_small_Kep = zeta(pratios_small_Kep, order=2) # DON'T TRUST THESE YET!
plot_fig_pdf_simple([zeta1_small_sim], [zeta1_small_Kep], x_min=-1., x_max=1., n_bins=40, normalize=True, lw=lw, labels_sim=[r'Simulated (all $\mathcal{P} < %s$)' % Rm_max], labels_Kep=[r'Kepler (all $\mathcal{P} < %s$)' % Rm_max], xlabel_text=r'$\zeta_1$', ylabel_text='Normalized fraction', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=(8,5), fig_lbrt=[0.15,0.2,0.95,0.925], save_name=savefigures_directory + subdirectory + model_name + '_zeta1_small_compare.pdf', save_fig=savefigures)
plot_fig_pdf_simple([zeta2_small_sim], [zeta2_small_Kep], x_min=-1., x_max=1., n_bins=40, normalize=True, lw=lw, labels_sim=[r'Simulated (all $\mathcal{P} < %s$)' % Rm_max], labels_Kep=[r'Kepler (all $\mathcal{P} < %s$)' % Rm_max], xlabel_text=r'$\zeta_2$', ylabel_text='Normalized fraction', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=(8,5), fig_lbrt=[0.15,0.2,0.95,0.925], save_name=savefigures_directory + subdirectory + model_name + '_zeta2_small_compare.pdf', save_fig=savefigures)
plt.show()
#plt.close()





##### To plot galleries marking the planets near resonances:

ss_per_sys = ssk_per_sys # sss_per_sys

# For 2-planet systems:
bools_mult = ss_per_sys['Mtot_obs'] == 2
bools_pr = ss_per_sys['Rm_obs'][:,0] <= 2.5 # <-- this works only for 2-planet systems; in general use --> #[True if any((ss_per_sys['Rm_obs'][i] > 1.) & (ss_per_sys['Rm_obs'][i] < 2.5)) else False for i in range(len(ss_per_sys['Rm_obs']))] # select small period ratios
idx_sys_selected = np.where(bools_mult & bools_pr)[0]

zeta1_selected = zeta(ss_per_sys['Rm_obs'][idx_sys_selected,0])
idx_sys_selected = idx_sys_selected[np.argsort(zeta1_selected)][50:100]

x_min, x_max = 0.5*P_min, P_max
s_norm = 2.
legend_fmt, s_units = r'${:.0f}$', r'$R_\oplus$'

fig = plt.figure(figsize=(5,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.9,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.title('Systems with 2 observed planets', fontsize=16)
for j,idx in enumerate(idx_sys_selected):
    P_sys = ss_per_sys['P_obs'][idx]
    Pr_sys = ss_per_sys['Rm_obs'][idx] #P_sys[1:]/P_sys[:-1]
    Rp_sys = ss_per_sys['radii_obs'][idx]
    
    Rp_sys = Rp_sys[P_sys > 0]
    P_sys = P_sys[P_sys > 0]
    Pr_sys = Pr_sys[Pr_sys > 0]
    
    zeta1_sys = zeta(Pr_sys, order=1)
    c_sys = 'r' if np.abs(zeta1_sys[0]) <= 0.25 else 'k'
    #c_sys = np.array(['k']*len(P_sys))
    #for i,zeta1 in enumerate(zeta1_sys):
    #    if np.abs(zeta1) <= 0.25:
    #        c_sys[i:i+2] = 'r'
    sc = plt.scatter(P_sys, np.ones(len(P_sys))+j, s=s_norm*Rp_sys**2., c=c_sys)
    plt.text(x=x_min, y=j+1, s='{:.2f}'.format(Pr_sys[0]), va='center', ha='right', c=c_sys, fontsize=8)
    plt.text(x=x_max, y=j+1, s='{:.2f}'.format(zeta1_sys[0]), va='center', ha='left', c=c_sys, fontsize=8)
    plt.axhline(y=j+1, lw=0.05, color='k')
# Legend:
s_examples = [1., 3., 10.]
x_examples = np.logspace(np.log10(x_min), np.log10(x_max), len(s_examples)+2)[1:-1] # first and last elements are for buffer zones
plt.scatter(x_examples, np.ones(len(s_examples))+j+2, c='k', s=s_norm*np.array(s_examples)**2.)
for s,size in enumerate(s_examples):
    plt.annotate(legend_fmt.format(size) + s_units, (1.2*x_examples[s], j+3), ha='left', va='center', fontsize=12)
plt.text(1.1*x_min, j+3, s='Legend:', ha='left', va='center', fontsize=12)
plt.fill_between([x_min, x_max], j+2, j+4, color='b', alpha=0.2)
plt.text(x=0.9*x_min, y=j+2, s=r'$\mathcal{P}$', va='center', ha='right', fontsize=8)
plt.text(x=1.1*x_max, y=j+2, s=r'$\zeta_1$', va='center', ha='left', fontsize=8)
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=16)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([])
plt.xlim([x_min, x_max])
plt.ylim([0., len(idx_sys_selected)+3])
plt.xlabel('Period (days)', fontsize=16)
plt.show()





##### TESTING PURPOSES: visualizing the 1st and 2nd order MMRs and their neighborhoods (3rd order MMRs)

tt = np.linspace(1.00001, 4., 1000)
test = pratio_is_in_1st_order_mmr_neighborhood(tt)
colors = np.array(['k']*len(tt))
colors[test[0]] = 'r'

fig = plt.figure(figsize=(8,5))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.9,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(tt, np.ones(len(tt)), marker='.', c=colors)
for i in range(1,6):
    mmr1_i = (i+1)/i
    print('%s:%s =' % (i+1, i), mmr1_i)
    plt.axvline(mmr1_i, color='r')
for i in range(1,10,2):
    mmr2_i = (i+2)/i
    print('%s:%s =' % (i+2, i), mmr2_i)
    plt.axvline(mmr2_i, color='b')
plt.show()
