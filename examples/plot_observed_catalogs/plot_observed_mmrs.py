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

pratio_max_1, pratio_max_2 = 2.5, 4. # should be 2.5 for zeta_{2,1} and 4 for zeta_{2,2}
pratios_small_sim = sss['Rm_obs'][sss['Rm_obs'] < pratio_max_2]
pratios_small_Kep = ssk['Rm_obs'][ssk['Rm_obs'] < pratio_max_2]

bools_in_1st_sim, _ = pratio_is_in_any_1st_order_mmr_neighborhood(pratios_small_sim)
bools_in_1st_Kep, _ = pratio_is_in_any_1st_order_mmr_neighborhood(pratios_small_Kep)
bools_in_2nd_sim, _ = pratio_is_in_any_2nd_order_mmr_neighborhood(pratios_small_sim)
bools_in_2nd_Kep, _ = pratio_is_in_any_2nd_order_mmr_neighborhood(pratios_small_Kep)
zeta1_small_sim = zeta(pratios_small_sim[bools_in_1st_sim])
zeta1_small_Kep = zeta(pratios_small_Kep[bools_in_1st_Kep])
zeta2_small_sim = zeta(pratios_small_sim[bools_in_2nd_sim], order=2)
zeta2_small_Kep = zeta(pratios_small_Kep[bools_in_2nd_Kep], order=2)

plot_fig_pdf_simple([zeta1_small_sim], [zeta1_small_Kep], x_min=-1., x_max=1., n_bins=40, normalize=True, lw=lw, labels_sim=[r'Simulated (all $\mathcal{P} < %s$)' % pratio_max_1], labels_Kep=[r'Kepler (all $\mathcal{P} < %s$)' % pratio_max_1], xlabel_text=r'$\zeta_1$', ylabel_text='Normalized fraction', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=(8,5), fig_lbrt=[0.15,0.2,0.95,0.925], save_name=savefigures_directory + subdirectory + model_name + '_zeta1_small_compare.pdf', save_fig=savefigures)
plot_fig_pdf_simple([zeta2_small_sim], [zeta2_small_Kep], x_min=-1., x_max=1., n_bins=40, normalize=True, lw=lw, labels_sim=[r'Simulated (all $\mathcal{P} < %s$)' % pratio_max_2], labels_Kep=[r'Kepler (all $\mathcal{P} < %s$)' % pratio_max_2], xlabel_text=r'$\zeta_2$', ylabel_text='Normalized fraction', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=(8,5), fig_lbrt=[0.15,0.2,0.95,0.925], save_name=savefigures_directory + subdirectory + model_name + '_zeta2_small_compare.pdf', save_fig=savefigures)
plt.show()
#plt.close()





##### To plot galleries marking the planets near resonances:

ss_per_sys = ssk_per_sys # sss_per_sys, ssk_per_sys

n_mmr = 1 # 1 => only consider the 1st order MMRs; 2 => consider both 1st and 2nd order MMRs
pratio_max = 3. if n_mmr==1 else 4.

zeta_in_mmr_lim = 0.25 # all |zeta1| <= zeta_in_mmr_lim are considered "in a 1st order MMR"

x_min, x_max = 0.5*P_min, P_max
s_norm = 2.
s_examples = [1., 3., 10.]
x_examples = np.logspace(np.log10(x_min), np.log10(x_max), len(s_examples)+2)[1:-1] # first and last elements are for buffer zones
legend_fmt, s_units = r'${:.0f}$', r'$R_\oplus$'



### For 2-planet systems:
bools_mult = ss_per_sys['Mtot_obs'] == 2
bools_pr = ss_per_sys['Rm_obs'][:,0] <= pratio_max # <-- this works only for 2-planet systems; in general use --> #[True if any((ss_per_sys['Rm_obs'][i] > 1.) & (ss_per_sys['Rm_obs'][i] < pratio_max)) else False for i in range(len(ss_per_sys['Rm_obs']))] # select small period ratios
idx_sys_selected = np.where(bools_mult & bools_pr)[0]

# Sort by zeta:
if n_mmr == 1:
    zeta_selected = zeta(ss_per_sys['Rm_obs'][idx_sys_selected,0]) # zeta_{1,1}
elif n_mmr == 2:
    zeta_selected, _1, _2 = zeta_2_order(ss_per_sys['Rm_obs'][idx_sys_selected,0]) # zeta_{2,1} and zeta_{2,2}
else:
    print('ERROR: must set n_mmr = 1 or 2')

idx_sys_selected = idx_sys_selected[np.argsort(zeta_selected)]

n_figs = 4 # divide the total number of systems into this many figures as evenly as possible
sys_per_fig = int(np.ceil(len(idx_sys_selected)/n_figs))

for n in range(n_figs):
    fig = plt.figure(figsize=(5,8))
    plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.9,top=0.95,wspace=0,hspace=0)
    ax = plt.subplot(plot[0,0])
    plt.title('Systems with 2 observed planets', fontsize=16)
    for j,idx in enumerate(idx_sys_selected[n*sys_per_fig:(n+1)*sys_per_fig]):
        P_sys = ss_per_sys['P_obs'][idx]
        Pr_sys = ss_per_sys['Rm_obs'][idx] #P_sys[1:]/P_sys[:-1]
        Rp_sys = ss_per_sys['radii_obs'][idx]
        
        Rp_sys = Rp_sys[P_sys > 0]
        P_sys = P_sys[P_sys > 0]
        Pr_sys = Pr_sys[Pr_sys > 0]
        
        if n_mmr == 1:
            zeta_sys = zeta(Pr_sys, n=1, order=1) # zeta_{1,1}
        elif n_mmr == 2:
            zeta_sys, in_1st, in_2nd = zeta_2_order(Pr_sys) # zeta_{2,1} or zeta_{2,2}; can also be NaN if did not check enough indices for period ratios -> 1
        #c_sys = np.array(['k']*len(P_sys))
        #for i,zeta1 in enumerate(zeta_sys):
        #    if np.abs(zeta1) <= 0.25:
        #        c_sys[i:i+2] = 'r'
        c_sys = 'k'
        if np.abs(zeta_sys[0]) <= zeta_in_mmr_lim:
            if n_mmr == 1:
                c_sys = 'r'
                in_1st, i_1st = pratio_is_in_any_1st_order_mmr_neighborhood(Pr_sys)
                in_1st, i_1st = in_1st[0], i_1st[0]
                #assert in_1st and i_1st >= 1
                mmr = '%s:%s' % (i_1st+1,i_1st)
            elif n_mmr == 2: # if get here, also means 'zeta_sys' is not a NaN
                assert ~(in_1st & in_2nd) # one must be True and the other is False
                if in_1st:
                    c_sys = 'r'
                    _, i_1st = pratio_is_in_any_1st_order_mmr_neighborhood(Pr_sys)
                    i_1st = i_1st[0]
                    mmr = '%s:%s' % (i_1st+1,i_1st)
                elif in_2nd:
                    c_sys = 'b'
                    _, i_2nd = pratio_is_in_any_2nd_order_mmr_neighborhood(Pr_sys)
                    n_2nd = 2*i_2nd[0] - 1
                    mmr = '%s:%s' % (n_2nd+2,n_2nd)
            plt.text(x=np.mean(P_sys), y=j+1.2, s=mmr, va='center', ha='center', c=c_sys, fontsize=8)
        sc = plt.scatter(P_sys, np.ones(len(P_sys))+j, s=s_norm*Rp_sys**2., c=c_sys)
        plt.text(x=x_min, y=j+1, s='{:.2f}'.format(Pr_sys[0]), va='center', ha='right', c=c_sys, fontsize=8)
        plt.text(x=x_max, y=j+1, s='{:.2f}'.format(zeta_sys[0]), va='center', ha='left', c=c_sys, fontsize=8)
        plt.axhline(y=j+1, lw=0.05, color='k')
    # Legend:
    plt.scatter(x_examples, np.ones(len(s_examples))+j+2, c='k', s=s_norm*np.array(s_examples)**2.)
    for s,size in enumerate(s_examples):
        plt.annotate(legend_fmt.format(size) + s_units, (1.2*x_examples[s], j+3), ha='left', va='center', fontsize=12)
    plt.text(1.1*x_min, j+3, s='Legend:', ha='left', va='center', fontsize=12)
    plt.fill_between([x_min, x_max], j+2, j+4, color='b', alpha=0.2)
    plt.text(x=0.9*x_min, y=j+2, s=r'$\mathcal{P}$', va='center', ha='right', fontsize=8)
    if n_mmr == 1:
        plt.text(x=1.1*x_max, y=j+2, s=r'$\zeta_{1,1}$', va='center', ha='left', fontsize=8)
    elif n_mmr == 2:
        plt.text(x=1.1*x_max, y=j+3, s=r'$\zeta_{2,1}$ or', va='center', ha='left', fontsize=8)
        plt.text(x=1.1*x_max, y=j+2, s=r'$\zeta_{2,2}$', va='center', ha='left', fontsize=8)
    plt.gca().set_xscale("log")
    ax.tick_params(axis='both', labelsize=16)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([])
    plt.xlim([x_min, x_max])
    plt.ylim([0., sys_per_fig+3])
    plt.xlabel('Period (days)', fontsize=16)
plt.show()



### For 3+ planet systems:

bools_mult = ss_per_sys['Mtot_obs'] >= 3
bools_pr = [True if any((ss_per_sys['Rm_obs'][i] > 1.) & (ss_per_sys['Rm_obs'][i] < pratio_max)) else False for i in range(len(ss_per_sys['Rm_obs']))] # select small period ratios
idx_sys_selected = np.where(bools_mult & bools_pr)[0]
pratios_per_sys_selected = [ss_per_sys['Rm_obs'][idx, ss_per_sys['Rm_obs'][idx] > 1] for idx in idx_sys_selected]

# Sort by the smallest |zeta| (i.e. pair closest to an MMR) in each system:
# NOTE: must only consider period ratios less than 'pratio_max', as larger period ratios can also lead to small values of zeta_{1,1}
if n_mmr == 1:
    abs_zeta_min_selected = [np.min(np.abs(zeta(pratios_sys[pratios_sys < pratio_max]))) for pratios_sys in pratios_per_sys_selected] # min|zeta_{1,1}|
elif n_mmr == 2:
    abs_zeta_min_selected = [np.min(np.abs(zeta_2_order(pratios_sys[pratios_sys < pratio_max])[0])) for pratios_sys in pratios_per_sys_selected] # min of any |zeta_{2,1}| or |zeta_{2,2}|
else:
    print('ERROR: must set n_mmr = 1 or 2')

idx_sys_selected = idx_sys_selected[np.argsort(abs_zeta_min_selected)]

n_figs = 3 # divide the total number of systems into this many figures as evenly as possible
sys_per_fig = int(np.ceil(len(idx_sys_selected)/n_figs))

for n in range(n_figs):
    fig = plt.figure(figsize=(5,8))
    plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.9,top=0.95,wspace=0,hspace=0)
    ax = plt.subplot(plot[0,0])
    plt.title('Systems with 3+ observed planets', fontsize=16)
    for j,idx in enumerate(idx_sys_selected[n*sys_per_fig:(n+1)*sys_per_fig]):
        P_sys = ss_per_sys['P_obs'][idx]
        Pr_sys = ss_per_sys['Rm_obs'][idx] #P_sys[1:]/P_sys[:-1]
        Rp_sys = ss_per_sys['radii_obs'][idx]
        
        Rp_sys = Rp_sys[P_sys > 0]
        P_sys = P_sys[P_sys > 0]
        Pr_sys = Pr_sys[Pr_sys > 0]
        
        if n_mmr == 1:
            zeta_sys = zeta(Pr_sys, n=1, order=1) # zeta_{1,1}
        elif n_mmr == 2:
            zeta_sys, in_1st_sys, in_2nd_sys = np.full(len(Pr_sys), np.nan), np.full(len(Pr_sys), False), np.full(len(Pr_sys), False)
            zeta_sys_small, in_1st_sys_small, in_2nd_sys_small = zeta_2_order(Pr_sys[Pr_sys < pratio_max]) # zeta_{2,1} or zeta_{2,2}; can also be NaN if did not check enough indices for period ratios -> 1
            zeta_sys[Pr_sys < pratio_max] = zeta_sys_small
            in_1st_sys[Pr_sys < pratio_max] = in_1st_sys_small
            in_2nd_sys[Pr_sys < pratio_max] = in_2nd_sys_small
        abs_zeta_min = np.min(np.abs(zeta_sys[Pr_sys < pratio_max]))
        c_sys = np.array(['k']*len(P_sys)) # colors for each planet
        num_mmr = 0 # count the number of MMRs
        for i,zeta_i in enumerate(zeta_sys):
            if (np.abs(zeta_i) <= zeta_in_mmr_lim) and (Pr_sys[i] < pratio_max):
                c_sys[i:i+2] = 'r' ##### how to color planets that are part of different MMRs in a chain?
                num_mmr += 1
                if n_mmr == 1:
                    c_pair = 'r'
                    in_1st, i_1st = pratio_is_in_any_1st_order_mmr_neighborhood(Pr_sys[i:i+1])
                    in_1st, i_1st = in_1st[0], i_1st[0]
                    #assert in_1st and i_1st >= 1
                    mmr = '%s:%s' % (i_1st+1,i_1st)
                elif n_mmr == 2: # if get here, also means 'zeta_sys' is not a NaN
                    assert ~(in_1st_sys[i] & in_2nd_sys[i]) # one must be True and the other is False
                    if in_1st_sys[i]:
                        c_pair = 'r'
                        _, i_1st = pratio_is_in_any_1st_order_mmr_neighborhood(Pr_sys[i:i+1])
                        i_1st = i_1st[0]
                        mmr = '%s:%s' % (i_1st+1,i_1st)
                    elif in_2nd_sys[i]:
                        c_pair = 'b'
                        _, i_2nd = pratio_is_in_any_2nd_order_mmr_neighborhood(Pr_sys[i:i+1])
                        n_2nd = 2*i_2nd[0] - 1
                        mmr = '%s:%s' % (n_2nd+2,n_2nd)
                plt.text(x=np.mean(P_sys[i:i+2]), y=j+1.2, s=mmr, va='center', ha='center', c=c_pair, fontsize=8)
        c_wholesys = 'r' if np.any(c_sys=='r') else 'k' # if want a single color for the whole system
        if num_mmr == 2:
            c_wholesys = 'orange'
        elif num_mmr == 3:
            c_wholesys = 'm'
        sc = plt.scatter(P_sys, np.ones(len(P_sys))+j, s=s_norm*Rp_sys**2., c=c_sys)
        #<[1] Display one of the following left-labels:
        #plt.text(x=x_min, y=j+1, s='{:.2f}'.format(np.mean(Pr_sys)), va='center', ha='right', c=c_wholesys, fontsize=8) # display mean period ratio
        plt.text(x=x_min, y=j+1, s='%s (%s)' % (len(c_sys), np.sum(c_sys=='r')), va='center', ha='right', c=c_wholesys, fontsize=8) # display multiplicity (total vs. near MMR)
        #plt.text(x=x_min, y=j+1, s='{:.2f}'.format(gen.gap_complexity_GF2020(P_sys)), va='center', ha='right', c=c_wholesys, fontsize=8) # display gap complexity
        #plt.text(x=x_min, y=j+1, s='{:.3f}'.format(gen.dispersion_W2022(Rp_sys)), va='center', ha='right', c=c_wholesys, fontsize=8) # display radius dispersion
        #[1]>
        plt.text(x=x_max, y=j+1, s='{:.2f}'.format(abs_zeta_min), va='center', ha='left', c=c_wholesys, fontsize=8) # display min|zeta|
        plt.axhline(y=j+1, lw=max(0.05, 0.05*num_mmr), color=c_wholesys)
    # Legend:
    plt.scatter(x_examples, np.ones(len(s_examples))+j+2, c='k', s=s_norm*np.array(s_examples)**2.)
    for s,size in enumerate(s_examples):
        plt.annotate(legend_fmt.format(size) + s_units, (1.2*x_examples[s], j+3), ha='left', va='center', fontsize=12)
    plt.text(1.1*x_min, j+3, s='Legend:', ha='left', va='center', fontsize=12)
    plt.fill_between([x_min, x_max], j+2, j+4, color='b', alpha=0.2)
    #<[1] Display the appropriate left-label legend:
    #plt.text(x=0.9*x_min, y=j+2, s=r'$\bar{\mathcal{P}}$', va='center', ha='right', fontsize=8) # for displaying mean period ratio
    plt.text(x=1.*x_min, y=j+2, s=r'$n$ ($n_{\rm MMR}$)', va='center', ha='right', fontsize=8) # for displaying multiplicity (total vs. near MMR)
    #plt.text(x=0.9*x_min, y=j+2, s=r'$\mathcal{C}$', va='center', ha='right', fontsize=8) # for displaying gap complexity
    #plt.text(x=0.9*x_min, y=j+2, s=r'$\sigma_R^2$', va='center', ha='right', fontsize=8) # for displaying radius partitioning
    #[1]>
    if n_mmr == 1:
        plt.text(x=1.*x_max, y=j+2, s=r'min$|\zeta_{1,1}|$', va='center', ha='left', fontsize=8)
    elif n_mmr == 2:
        #plt.text(x=1.*x_max, y=j+3, s=r'min$|\zeta_{2,1}|$', va='center', ha='left', fontsize=8)
        plt.text(x=1.*x_max, y=j+2, s=r'min$|\zeta_{2,i}|$', va='center', ha='left', fontsize=8)
    plt.gca().set_xscale("log")
    ax.tick_params(axis='both', labelsize=16)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([])
    plt.xlim([x_min, x_max])
    plt.ylim([0., sys_per_fig+3])
    plt.xlabel('Period (days)', fontsize=16)
plt.show()





##### TESTING PURPOSES: visualizing the 1st and 2nd order MMRs and their neighborhoods (3rd order MMRs)

pr_all = np.linspace(1.001, 5., 1000)
i_max = 5
bools_in_1st, i_of_1st = pratio_is_in_any_1st_order_mmr_neighborhood(pr_all, i_max=i_max)
bools_in_2nd, i_of_2nd = pratio_is_in_any_2nd_order_mmr_neighborhood(pr_all, i_max=i_max)
colors_all = np.array(['k']*len(pr_all))
colors_all[bools_in_1st] = 'r'
colors_all[bools_in_2nd] = 'b'

fig = plt.figure(figsize=(8,5))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.9,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(pr_all, np.ones(len(pr_all)), marker='.', c=colors_all)
for i in range(1,i_max+1):
    # 1st order MMRs:
    mmr1_i = (i+1)/i
    print('%s:%s =' % (i+1, i), mmr1_i)
    plt.axvline(mmr1_i, color='r')
    
    # 2nd order MMRs:
    n = 2*i-1
    mmr2_i = (n+2)/n
    print('%s:%s =' % (n+2, n), mmr2_i)
    plt.axvline(mmr2_i, color='b')
plt.show()
