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

from syssimpyplots.compute_RVs import *





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/RV/'
save_name = 'Maximum_AMD_model'





##### To load the underlying populations:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)





##### To compute and plot RV semi-amplitudes:

n_bins = 50
lw = 2 #linewidth
alpha = 0.2

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size






##### To compute RV semi-amplitudes:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'

runs = 100

# Aggregate = 1% of the points from each catalog:
factor = 5
pr_all_aggregate = []
pr_obs_aggregate = []
Kr_all_aggregate = []
Kr_obs_aggregate = []

pr_min, pr_max = 3./10., 300./5.
pr_bins = np.logspace(np.log10(pr_min), np.log10(pr_max), n_bins+1)
pr_bins_mid = (pr_bins[:-1] + pr_bins[1:])/2.
pr_counts_obs = []
pr_counts_obs_above1 = [] # K_max/K_[5-10]d > 1
pr_counts_obs_below1 = [] # K_2ndmax/K_[5-10]d < 1

Kr_bins = np.logspace(-3., 3., n_bins+1)
Kr_bins_mid = (Kr_bins[:-1] + Kr_bins[1:])/2.
Kr_counts_all = []
Kr_counts_obs = []

# To store some other statistics:
f_all_kr_above1 = [] # fractions of all 5-10d planets where K_max/K_[5-10]d > 1
f_obs_kr_above1 = [] # fractions of observed 5-10d planets where K_max/K_[5-10]d > 1
f_obs_kr_above1_given_pr_above1 = [] # fractions of observed 5-10d planets where K_max/K_[5-10]d > 1 given that the K_max or K_2ndmax planet is outer
f_obs_kr_above1_given_pr_below1 = [] # fractions of observed 5-10d planets where K_max/K_[5-10]d > 1 given that the K_max or K_2ndmax planet is inner
f_obs_pr_above1_given_kr_above1 = [] # fractions of observed 5-10d planets where P_Kmax is an outer planet to P_[5-10]d given that K_max/K_[5-10]d > 1
f_obs_pr_above1_given_kr_below1 = [] # fractions of observed 5-10d planets where P_Kmax is an outer planet to P_[5-10]d given that K_2ndmax/K_[5-10]d < 1

for i in range(runs): #range(1,runs+1)
    run_number = i+1
    print(i)
    N_sim_i = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)[0]
    param_vals_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

    # For multi-planet systems where at least one planet is in the period range:
    det_inrange_all = []
    P_inrange_all = []
    Rp_inrange_all = []
    Mp_inrange_all = []
    K_K_max_inrange_all = []
    K_max_or_2ndmax_all = [] # either K_max (when planet in range is not the max) or the 2nd K_max (when planet in range is the max); also includes repeats, for systems where more than one planet is in the range
    P_K_max_or_2ndmax_all = [] # period of either the planet with K_max (when planet in range is not the max) or of the planet with the 2nd K_max (when planet in range is the max)

    # For single systems where the planet is in the period range:
    P_inrange_singles = []
    Rp_inrange_singles = []
    Mp_inrange_singles = []
    K_max_singles = []

    for i,P_sys in enumerate(sssp_per_sys_i['P_all']):
        det_sys = sssp_per_sys_i['det_all'][i]
        Mp_sys = sssp_per_sys_i['mass_all'][i][P_sys > 0]
        Rp_sys = sssp_per_sys_i['radii_all'][i][P_sys > 0]
        e_sys = sssp_per_sys_i['e_all'][i][P_sys > 0]
        incl_sys = sssp_per_sys_i['incl_all'][i][P_sys > 0]
        P_sys = P_sys[P_sys > 0]
        n_pl =  len(P_sys)
        if any((P_sys > 5.) & (P_sys < 10.)):
            if n_pl == 1: # singles
                K_single = rv_K(Mp_sys, P_sys, Mstar=sssp_i['Mstar_all'][i])
                P_inrange_singles.append(P_sys[0])
                Rp_inrange_singles.append(Rp_sys[0])
                Mp_inrange_singles.append(Mp_sys[0])
                K_max_singles.append(K_single[0])
            else: # multi-planet systems
                j_inrange = np.arange(len(P_sys))[(P_sys > 5.) & (P_sys < 10.)]
                #print(i, ': planets in [5,10]d = ', len(j_inrange))

                #K_sys = rv_K(Mp_sys, P_sys, Mstar=sssp_i['Mstar_all'][i])
                K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=sssp_i['Mstar_all'][i])
                idsort_K_sys = np.argsort(K_sys)
                K_max, K_2ndmax = K_sys[idsort_K_sys[-1]], K_sys[idsort_K_sys[-2]]
                P_K_max, P_K_2ndmax = P_sys[idsort_K_sys[-1]], P_sys[idsort_K_sys[-2]]
                K_K_max_sys = K_sys/K_max
                for j in j_inrange:
                    det_inrange_all.append(det_sys[j])
                    P_inrange_all.append(P_sys[j])
                    Rp_inrange_all.append(Rp_sys[j])
                    Mp_inrange_all.append(Mp_sys[j])
                    #K_max_all.append(K_max)
                    if K_K_max_sys[j] == 1:
                        K_K_max_inrange_all.append(K_max/K_2ndmax)
                        K_max_or_2ndmax_all.append(K_2ndmax)
                        P_K_max_or_2ndmax_all.append(P_K_2ndmax)
                    else:
                        K_K_max_inrange_all.append(K_K_max_sys[j])
                        K_max_or_2ndmax_all.append(K_max)
                        P_K_max_or_2ndmax_all.append(P_K_max)

    det_inrange_all = np.array(det_inrange_all)
    P_inrange_all = np.array(P_inrange_all)
    Rp_inrange_all = np.array(Rp_inrange_all)
    Mp_inrange_all = np.array(Mp_inrange_all)
    K_K_max_inrange_all = np.array(K_K_max_inrange_all)
    K_max_or_2ndmax_all = np.array(K_max_or_2ndmax_all)
    P_K_max_or_2ndmax_all = np.array(P_K_max_or_2ndmax_all)

    P_inrange_singles = np.array(P_inrange_singles)
    Rp_inrange_singles = np.array(Rp_inrange_singles)
    Mp_inrange_singles = np.array(Mp_inrange_singles)
    K_max_singles = np.array(K_max_singles)



    # Collect a sample of points from each catalog:
    kr_all = 1./K_K_max_inrange_all
    pr_all = P_K_max_or_2ndmax_all/P_inrange_all
    n_all, n_obs = len(kr_all), np.sum(det_inrange_all == 1)

    pr_all_aggregate = pr_all_aggregate + list(pr_all[:int(n_all*factor/runs)])
    pr_obs_aggregate = pr_obs_aggregate + list(pr_all[det_inrange_all == 1][:int(n_obs*factor/runs)])
    Kr_all_aggregate = Kr_all_aggregate + list(kr_all[:int(n_all*factor/runs)])
    Kr_obs_aggregate = Kr_obs_aggregate + list(kr_all[det_inrange_all == 1][:int(n_obs*factor/runs)])

    # Compute the histograms:
    counts, bins = np.histogram(pr_all[det_inrange_all == 1], bins=pr_bins)
    pr_counts_obs.append(counts)

    counts, bins = np.histogram(pr_all[(kr_all > 1.) & (det_inrange_all == 1)], bins=pr_bins)
    pr_counts_obs_above1.append(counts)

    counts, bins = np.histogram(pr_all[(kr_all < 1.) & (det_inrange_all == 1)], bins=pr_bins)
    pr_counts_obs_below1.append(counts)

    counts, bins = np.histogram(kr_all, bins=Kr_bins)
    Kr_counts_all.append(counts/float(np.sum(counts)))

    counts, bins = np.histogram(kr_all[det_inrange_all == 1], bins=Kr_bins)
    Kr_counts_obs.append(counts/float(np.sum(counts)))

    # Compute other statistics:
    n_kr_obs = len(kr_all[det_inrange_all == 1])
    n_kr_obs_above1 = np.sum(kr_all[det_inrange_all == 1] > 1.)
    n_kr_obs_below1 = np.sum(kr_all[det_inrange_all == 1] < 1.)
    n_pr_obs_above1 = np.sum(pr_all[det_inrange_all == 1] > 1.)
    n_pr_obs_below1 = np.sum(pr_all[det_inrange_all == 1] < 1.)
    n_kr_obs_above1_pr_obs_above1 = np.sum(kr_all[(det_inrange_all == 1) & (pr_all > 1.)] > 1.)
    n_kr_obs_above1_pr_obs_below1 = np.sum(kr_all[(det_inrange_all == 1) & (pr_all < 1.)] > 1.)
    n_pr_obs_above1_kr_obs_above1 = np.sum(pr_all[(det_inrange_all == 1) & (kr_all > 1.)] > 1.)
    n_pr_obs_above1_kr_obs_below1 = np.sum(pr_all[(det_inrange_all == 1) & (kr_all < 1.)] > 1.)

    f_all_kr_above1.append(np.sum(kr_all > 1.)/len(kr_all))
    f_obs_kr_above1.append(n_kr_obs_above1/n_kr_obs)
    f_obs_kr_above1_given_pr_above1.append(n_kr_obs_above1_pr_obs_above1/n_pr_obs_above1)
    f_obs_kr_above1_given_pr_below1.append(n_kr_obs_above1_pr_obs_below1/n_pr_obs_below1)
    f_obs_pr_above1_given_kr_above1.append(n_pr_obs_above1_kr_obs_above1/n_kr_obs_above1)
    f_obs_pr_above1_given_kr_below1.append(n_pr_obs_above1_kr_obs_below1/n_kr_obs_below1)

pr_all_aggregate = np.array(pr_all_aggregate)
pr_obs_aggregate = np.array(pr_obs_aggregate)
Kr_all_aggregate = np.array(Kr_all_aggregate)
Kr_obs_aggregate = np.array(Kr_obs_aggregate)

pr_counts_obs = np.array(pr_counts_obs)
pr_counts_obs_above1 = np.array(pr_counts_obs_above1)
pr_counts_obs_below1 = np.array(pr_counts_obs_below1)

Kr_counts_all = np.array(Kr_counts_all)
Kr_counts_obs = np.array(Kr_counts_obs)

f_all_kr_above1 = np.array(f_all_kr_above1)
f_obs_kr_above1 = np.array(f_obs_kr_above1)
f_obs_kr_above1_given_pr_above1 = np.array(f_obs_kr_above1_given_pr_above1)
f_obs_kr_above1_given_pr_below1 = np.array(f_obs_kr_above1_given_pr_below1)
f_obs_pr_above1_given_kr_above1 = np.array(f_obs_pr_above1_given_kr_above1)
f_obs_pr_above1_given_kr_below1 = np.array(f_obs_pr_above1_given_kr_below1)



pr_counts_obs_qtls = np.zeros((n_bins,3))
pr_counts_obs_above1_qtls = np.zeros((n_bins,3))
pr_counts_obs_below1_qtls = np.zeros((n_bins,3))
Kr_counts_all_qtls = np.zeros((n_bins,3))
Kr_counts_obs_qtls = np.zeros((n_bins,3))
for b in range(n_bins):
    pr_counts_obs_qtls[b] = np.quantile(pr_counts_obs[:,b], [0.16,0.5,0.84])
    pr_counts_obs_above1_qtls[b] = np.quantile(pr_counts_obs_above1[:,b], [0.16,0.5,0.84])
    pr_counts_obs_below1_qtls[b] = np.quantile(pr_counts_obs_below1[:,b], [0.16,0.5,0.84])
    Kr_counts_all_qtls[b] = np.quantile(Kr_counts_all[:,b], [0.16,0.5,0.84])
    Kr_counts_obs_qtls[b] = np.quantile(Kr_counts_obs[:,b], [0.16,0.5,0.84])






##### To compute some statistics and plot RV semi-amplitudes:

q16, q50, q84 = np.quantile(f_all_kr_above1, [0.16,0.5,0.84])
print('Fraction of all 5-10d planets where another planet has a larger K: %s_{%s}^{+%s}' % (np.round(q50,3), np.round(q16-q50,3), np.round(q84-q50,3)))

q16, q50, q84 = np.quantile(f_obs_kr_above1, [0.16,0.5,0.84])
print('Fraction of observed 5-10d planets where another planet has a larger K: %s_{%s}^{+%s}' % (np.round(q50,3), np.round(q16-q50,3), np.round(q84-q50,3)))

q16, q50, q84 = np.quantile(f_obs_kr_above1_given_pr_above1, [0.16,0.5,0.84])
print('Given a planet exterior to the observed 5-10d planet, the fraction of the time where that planet has a larger K: %s_{%s}^{+%s}' % (np.round(q50,3), np.round(q16-q50,3), np.round(q84-q50,3)))

q16, q50, q84 = np.quantile(f_obs_kr_above1_given_pr_below1, [0.16,0.5,0.84])
print('Given a planet interior to the observed 5-10d planet, the fraction of the time where that planet has a larger K: %s_{%s}^{+%s}' % (np.round(q50,3), np.round(q16-q50,3), np.round(q84-q50,3)))

q16, q50, q84 = np.quantile(f_obs_pr_above1_given_kr_above1, [0.16,0.5,0.84])
print('Given another planet has a larger K than the observed 5-10d planet, the fraction of the time where that planet is exterior: %s_{%s}^{+%s}' % (np.round(q50,3), np.round(q16-q50,3), np.round(q84-q50,3)))

q16, q50, q84 = np.quantile(f_obs_pr_above1_given_kr_below1, [0.16,0.5,0.84])
print('Given that the observed 5-10d planet has the largest K, the fraction of the time where the second largest K planet is exterior: %s_{%s}^{+%s}' % (np.round(q50,3), np.round(q16-q50,3), np.round(q84-q50,3)))


# K_max/K vs. period ratio (all planets):

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5, 5, left=0.18, bottom=0.13, right=0.98, top=0.98, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4])
corner.hist2d(np.log10(pr_all_aggregate), np.log10(Kr_all_aggregate), bins=[30,60], plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
plt.scatter(np.log10(pr_obs_aggregate), np.log10(Kr_obs_aggregate), marker='.', s=2, c='m')
plt.axhline(y=0., color='g', ls='--', lw=3)
plt.axvline(x=0., color='g', ls='--', lw=3)
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([0.5, 1., 2., 4., 8., 16., 32.])
ytick_vals = np.array([0.01, 0.1, 1., 10., 100.])
plt.xticks(np.log10(xtick_vals), xtick_vals)
plt.yticks(np.log10(ytick_vals), ytick_vals)
plt.xlim([np.log10(pr_min), np.log10(pr_max)])
plt.ylim([-2., 2.]) #[1e-3, 1e3]
plt.xlabel(r'$P_{K_{\rm max\;or\;2nd\;max}}/P_{[5,10]d}$', fontsize=tfs)
plt.ylabel(r'$K_{\rm 2nd\;max}/K_{[5,10]d}$         $K_{\rm max}/K_{[5,10]d}$     ', fontsize=tfs)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.plot(pr_bins_mid, pr_counts_obs_qtls[:,1], drawstyle='steps-mid', color='m', ls='-', lw=lw, label='')
plt.plot(pr_bins_mid, pr_counts_obs_above1_qtls[:,1], drawstyle='steps-mid', color='b', ls='-', lw=lw, label=r'$\frac{K_{\rm max}}{K_{[5,10]d}} > 1$')
plt.plot(pr_bins_mid, pr_counts_obs_below1_qtls[:,1], drawstyle='steps-mid', color='r', ls='-', lw=lw, label=r'$\frac{K_{\rm 2nd\;max}}{K_{[5,10]d}} < 1$')
plt.fill_between(pr_bins_mid, pr_counts_obs_qtls[:,0], pr_counts_obs_qtls[:,2], color='m', alpha=alpha)
plt.fill_between(pr_bins_mid, pr_counts_obs_above1_qtls[:,0], pr_counts_obs_above1_qtls[:,2], color='b', alpha=alpha)
plt.fill_between(pr_bins_mid, pr_counts_obs_below1_qtls[:,0], pr_counts_obs_below1_qtls[:,2], color='r', alpha=alpha)
plt.axvline(x=1., color='g', ls='--', lw=3)
plt.gca().set_xscale("log")
plt.xlim([pr_min, pr_max])
plt.ylim([0,40])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(1.02,1.05), ncol=2, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4]) # side histogram
plt.plot(Kr_counts_all_qtls[:,1], Kr_bins_mid, drawstyle='steps-mid', color='k', ls='-', lw=lw, label='All')
plt.plot(Kr_counts_obs_qtls[:,1], Kr_bins_mid, drawstyle='steps-mid', color='m', ls='-', lw=lw, label=r'Observed $P_{[5,10]d}$')
plt.fill_betweenx(Kr_bins_mid, Kr_counts_all_qtls[:,0], Kr_counts_all_qtls[:,2], color='k', alpha=alpha)
plt.fill_betweenx(Kr_bins_mid, Kr_counts_obs_qtls[:,0], Kr_counts_obs_qtls[:,2], color='m', alpha=alpha)
plt.axhline(y=1., color='g', ls='--', lw=3)
plt.gca().set_yscale("log")
plt.ylim([1e-2, 1e2])
plt.xticks([])
plt.yticks([])
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + save_name + '_RVratio_pratio_5-10d_credible.pdf')
    plt.close()

plt.show()
