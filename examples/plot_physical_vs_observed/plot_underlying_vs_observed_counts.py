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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/'
save_name = 'Models_Compare'





##### To load the underlying populations:

# Model 1:
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

model_name = 'Maximum AMD model'





##### To count the observed and intrinsic multiplicities:

mult_obs_all = []
mult_obs_1earth_all = []
mult_obs_2earth_all = []
mult_true_all = []
mult_true_1earth_all = []
mult_true_2earth_all = []
for i,dets in enumerate(sssp_per_sys['det_all']):
    radii_sys = sssp_per_sys['radii_all'][i]
    m_obs = int(np.sum((dets == 1) & (radii_sys > 0.)))
    m_obs_1earth = int(np.sum((dets == 1) & (radii_sys > 1.)))
    m_obs_2earth = int(np.sum((dets == 1) & (radii_sys > 2.)))
    if m_obs > 0:
        n_true = np.sum(sssp_per_sys['radii_all'][i] > 0.)
        mult_obs_all.append(m_obs)
        mult_true_all.append(n_true)
        print('{}: m_obs = {}, n_true = {}'.format(i, m_obs, n_true))
    if m_obs_1earth > 0:
        n_true_1earth = np.sum(sssp_per_sys['radii_all'][i] > 1.)
        mult_obs_1earth_all.append(m_obs_1earth)
        mult_true_1earth_all.append(n_true_1earth)
        print('{}: m_obs = {}, n_true = {}'.format(i, m_obs_1earth, n_true_1earth))
    if m_obs_2earth > 0:
        n_true_2earth = np.sum(sssp_per_sys['radii_all'][i] > 2.)
        mult_obs_2earth_all.append(m_obs_2earth)
        mult_true_2earth_all.append(n_true_2earth)
        print('{}: m_obs = {}, n_true = {}'.format(i, m_obs_2earth, n_true_2earth))
mult_obs_all = np.array(mult_obs_all)
mult_obs_1earth_all = np.array(mult_obs_1earth_all)
mult_obs_2earth_all = np.array(mult_obs_2earth_all)
mult_true_all = np.array(mult_true_all)
mult_true_1earth_all = np.array(mult_true_1earth_all)
mult_true_2earth_all = np.array(mult_true_2earth_all)

n_m1 = mult_true_all[mult_obs_all == 1]
n_m1_1earth = mult_true_1earth_all[mult_obs_1earth_all == 1]
n_m1_2earth = mult_true_2earth_all[mult_obs_2earth_all == 1]
print('All planets:   p(n=1 | m=1) = %s' % (np.sum(n_m1 == 1)/len(n_m1)))
print('R > 1 R_earth: p(n=1 | m=1) = %s' % (np.sum(n_m1_1earth == 1)/len(n_m1_1earth)))
print('R > 2 R_earth: p(n=1 | m=1) = %s' % (np.sum(n_m1_2earth == 1)/len(n_m1_2earth)))





##### To load and compute the same statistics for a large number of models:
'''
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'

runs = 100
f_n1_m1_all = []
f_n1_m1_1earth_all = []
f_n1_m1_2earth_all = []
for i in range(1,runs+1): #range(1,runs+1)
    run_number = i
    print(i)
    N_sim_i = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)[0]
    param_vals_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

    mult_obs_all_i = []
    mult_obs_1earth_all_i = []
    mult_obs_2earth_all_i = []
    mult_true_all_i = []
    mult_true_1earth_all_i = []
    mult_true_2earth_all_i = []
    for j,dets in enumerate(sssp_per_sys_i['det_all']):
        radii_sys = sssp_per_sys_i['radii_all'][j]
        m_obs = int(np.sum((dets == 1) & (radii_sys > 0.)))
        m_obs_1earth = int(np.sum((dets == 1) & (radii_sys > 1.)))
        m_obs_2earth = int(np.sum((dets == 1) & (radii_sys > 2.)))
        if m_obs > 0:
            n_true = np.sum(sssp_per_sys_i['radii_all'][j] > 0.)
            mult_obs_all_i.append(m_obs)
            mult_true_all_i.append(n_true)
            #print('{}: m_obs = {}, n_true = {}'.format(j, m_obs, n_true))
        if m_obs_1earth > 0:
            n_true_1earth = np.sum(sssp_per_sys_i['radii_all'][j] > 1.)
            mult_obs_1earth_all_i.append(m_obs_1earth)
            mult_true_1earth_all_i.append(n_true_1earth)
            #print('{}: m_obs = {}, n_true = {}'.format(j, m_obs_1earth, n_true_1earth))
        if m_obs_2earth > 0:
            n_true_2earth = np.sum(sssp_per_sys_i['radii_all'][j] > 2.)
            mult_obs_2earth_all_i.append(m_obs_2earth)
            mult_true_2earth_all_i.append(n_true_2earth)
            #print('{}: m_obs = {}, n_true = {}'.format(j, m_obs_2earth, n_true_2earth))
    mult_obs_all_i = np.array(mult_obs_all_i)
    mult_obs_1earth_all_i = np.array(mult_obs_1earth_all_i)
    mult_obs_2earth_all_i = np.array(mult_obs_2earth_all_i)
    mult_true_all_i = np.array(mult_true_all_i)
    mult_true_1earth_all_i = np.array(mult_true_1earth_all_i)
    mult_true_2earth_all_i = np.array(mult_true_2earth_all_i)

    n_m1 = mult_true_all_i[mult_obs_all_i == 1]
    n_m1_1earth = mult_true_1earth_all_i[mult_obs_1earth_all_i == 1]
    n_m1_2earth = mult_true_2earth_all_i[mult_obs_2earth_all_i == 1]

    f_n1_m1_all.append(np.sum(n_m1 == 1)/len(n_m1))
    f_n1_m1_1earth_all.append(np.sum(n_m1_1earth == 1)/len(n_m1_1earth))
    f_n1_m1_2earth_all.append(np.sum(n_m1_2earth == 1)/len(n_m1_2earth))
f_n1_m1_all = np.array(f_n1_m1_all)
f_n1_m1_1earth_all = np.array(f_n1_m1_1earth_all)
f_n1_m1_2earth_all = np.array(f_n1_m1_2earth_all)
'''





##### To plot the intrinsic vs. observed multiplicity:

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

m_mults = [5,4,3,2,1] # will bin 5+

fig = plt.figure(figsize=(8,8))
plot = GridSpec(len(m_mults), 1, left=0.15, bottom=0.1, right=0.975, top=0.975, wspace=0, hspace=0)

for i,m in enumerate(m_mults):
    ax = plt.subplot(plot[i,0])

    if m == 5:
        n_mults = mult_true_all[mult_obs_all >= m]
        n_mults_1earth = mult_true_1earth_all[mult_obs_1earth_all >= m]
        n_mults_2earth = mult_true_2earth_all[mult_obs_2earth_all >= m]
        m_label = r'$m = 5+$'
    else:
        n_mults = mult_true_all[mult_obs_all == m]
        n_mults_1earth = mult_true_1earth_all[mult_obs_1earth_all == m]
        n_mults_2earth = mult_true_2earth_all[mult_obs_2earth_all == m]
        m_label = r'$m = %s$' % m

    xlabel = r'Intrinsic multiplicity $n$' if m == 1 else None
    xticks = range(1,11) if m == 1 else []
    plot_panel_counts_hist_simple(ax, [n_mults], [], x_min=0, x_max=11, x_llim=0.5, x_ulim=10.5, normalize=False, labels_sim=['All'], xticks_custom=xticks, xlabel_text=xlabel, ylabel_text=None, afs=afs, tfs=tfs, lfs=lfs)
    #plot_panel_counts_hist_simple(ax, [n_mults, n_mults_1earth, n_mults_2earth], [], x_min=0, x_max=11, x_llim=0.5, x_ulim=10.5, normalize=False, c_sim=['k','b','g'], ls_sim=['-','-','-'], labels_sim=['All', r'$R_p > 1 R_\plus$', r'$R_p > 2 R_\plus$'], xticks_custom=xticks, xlabel_text=xlabel, ylabel_text=None, afs=afs, tfs=tfs, lfs=lfs)
    plt.text(x=0.02, y=0.8, s=m_label, ha='left', fontsize=lfs, transform=ax.transAxes)

plt.show()





##### To plot the eccentricities of observed planets vs. observed multiplicity:

m_mults = [5,4,3,2,1] # will bin 5+

fig = plt.figure(figsize=(8,8))
plot = GridSpec(len(m_mults), 1, left=0.15, bottom=0.1, right=0.975, top=0.975, wspace=0, hspace=0)

for i,m in enumerate(m_mults):
    ax = plt.subplot(plot[i,0])

    ecc_m = []
    if m == 5:
        m_label = r'$m = 5+$'
        for i,dets in enumerate(sssp_per_sys['det_all']):
            m_obs = int(np.sum(dets))
            if m_obs >= m:
                ecc_m = ecc_m + list(sssp_per_sys['e_all'][i][dets == 1])
    else:
        m_label = r'$m = %s$' % m
        for i,dets in enumerate(sssp_per_sys['det_all']):
            m_obs = int(np.sum(dets))
            if m_obs == m:
                ecc_m = ecc_m + list(sssp_per_sys['e_all'][i][dets == 1])

    print(m, ': ', len(ecc_m))

    xlabel = r'Eccentricity $e$' if m == 1 else None
    xticks = [0.001, 0.01, 0.1, 1.] if m == 1 else []
    plot_panel_pdf_simple(ax, [ecc_m], [], x_min=1e-3, x_max=1., n_bins=20, normalize=False, log_x=True, xticks_custom=xticks, xlabel_text=xlabel, ylabel_text=None, afs=afs, tfs=tfs, lfs=lfs)
    plt.text(x=0.02, y=0.8, s=m_label, ha='left', fontsize=lfs, transform=ax.transAxes)

plt.show()
