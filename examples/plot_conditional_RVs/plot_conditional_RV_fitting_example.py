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
from syssimpyplots.compute_RVs import *





##### To load the underlying and observed populations:

savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Conditional_P8_12d_R0p9_1p1_transiting/' #'Conditional_P8_12d_R3_4_transiting/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Systems_conditional/Conditional_P8_12d_R0p9_1p1_transiting/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [8.,12.], [0.9,1.1], None
conds = conditionals_dict(P_cond_bounds=P_cond_bounds, Rp_cond_bounds=Rp_cond_bounds, Mp_cond_bounds=Mp_cond_bounds, det=True)





##### To simulate and fit RV observations of a system conditioned on a given planet, plotting the observations and fit to make a GIF:

afs=12
tfs=12
lfs=12

#N_obs_all = np.array([int(round(x)) for x in np.logspace(np.log10(10), np.log10(300), 20)])
N_obs_all = np.array([i for i in range(10,301,5)])
t_obs_σ, σ_1obs = 0.2, 0.3
repeat = 100
cond_only, fit_all_planets = False, False

i_cond = condition_systems_indices(sssp_per_sys, conds)
id_sys = i_cond[0] # choose a custom system # 6

start = time.time()

Mstar_sys = sssp['Mstar_all'][id_sys]
P_sys = sssp_per_sys['P_all'][id_sys]
det_sys = sssp_per_sys['det_all'][id_sys]
Mp_sys = sssp_per_sys['mass_all'][id_sys]
Rp_sys = sssp_per_sys['radii_all'][id_sys]
e_sys = sssp_per_sys['e_all'][id_sys]
incl_sys = sssp_per_sys['incl_all'][id_sys]
clusterids_sys = sssp_per_sys['clusterids_all'][id_sys]

det_sys = det_sys[P_sys > 0]
Mp_sys = Mp_sys[P_sys > 0]
Rp_sys = Rp_sys[P_sys > 0]
e_sys = e_sys[P_sys > 0]
incl_sys = incl_sys[P_sys > 0]
clusterids_sys = clusterids_sys[P_sys > 0]
P_sys = P_sys[P_sys > 0]

T0_sys = P_sys*np.random.random(len(P_sys)) # reference epochs for each planet
omega_sys = 2.*np.pi*np.random.random(len(P_sys)) # WARNING: need to get from simulated physical catalogs, NOT re-drawn

K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=Mstar_sys)
id_pl_cond = np.arange(len(P_sys))[(P_sys > conds['P_lower']) & (P_sys < conds['P_upper']) & (Rp_sys > conds['Rp_lower']) & (Rp_sys < conds['Rp_upper'])][0] # index of conditioned planet
P_cond = P_sys[id_pl_cond] # period of conditioned planet (days)
Rp_cond = Rp_sys[id_pl_cond] # radius of conditioned planet (Earth radii)
K_cond = K_sys[id_pl_cond] # K of the conditioned planet (m/s)
K_max = np.max(K_sys)

# To compute the true RV time series:
t_end = 150. # days
t_array = np.linspace(0., t_end, 1001)
RV_sys_per_pl = np.zeros((len(t_array),len(P_sys)))
for i in range(len(P_sys)):
    RV_sys_per_pl[:,i] = [RV_true(t, K_sys[i], P_sys[i], T0=T0_sys[i], e=e_sys[i], w=omega_sys[i]) for t in t_array]
RV_sys = np.sum(RV_sys_per_pl, axis=1)
RV_absmax = np.max([-np.min(RV_sys), np.max(RV_sys)])

# To simulate and fit RV observations for measuring K of the conditioned planet:
K_cond_hat_random = np.zeros((len(N_obs_all), repeat))
K_cond_hat_daily = np.zeros((len(N_obs_all), repeat))
sigma_K_cond_random = np.zeros((len(N_obs_all), repeat))
sigma_K_cond_daily = np.zeros((len(N_obs_all), repeat))
sigma_K_cond_ideal_all = σ_1obs * np.sqrt(2./N_obs_all)
#print('##### P_sys: ', P_sys)
for n in range(repeat):
    t_obs_daily = []
    for i,N_obs in enumerate(N_obs_all):
        covarsc = σ_1obs**2. * np.identity(N_obs)
        t_obs_random = np.sort(t_end*np.random.random(N_obs))
        N_obs_add = N_obs - len(t_obs_daily) # number of observations to add
        t_obs_daily = t_obs_daily + list(len(t_obs_daily) + np.arange(N_obs_add) + t_obs_σ*np.random.random(N_obs_add))

        if cond_only:
            #RV_obs_random = np.array([RV_true(t, K_cond, P_cond, T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond]) for t in t_obs_random]) + σ_1obs*np.random.randn(N_obs)
            RV_obs_daily = np.array([RV_true(t, K_cond, P_cond, T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond]) for t in t_obs_daily]) + σ_1obs*np.random.randn(N_obs)
        else:
            #RV_obs_random = np.array([RV_true_sys(t, K_sys, P_sys, T0_sys, e_sys, omega_sys) for t in t_obs_random]) + σ_1obs*np.random.randn(N_obs)
            RV_obs_daily = np.array([RV_true_sys(t, K_sys, P_sys, T0_sys, e_sys, omega_sys) for t in t_obs_daily]) + σ_1obs*np.random.randn(N_obs)

        if fit_all_planets:
            bools_fit = K_sys > σ_1obs/2.
            id_pl_cond_of_fits = np.where(np.arange(len(K_sys))[bools_fit] == id_pl_cond)[0][0] # index of conditioned planet counting only fitted planets
            # Do not attempt fitting if fewer observations than free parameters (planets)
            if N_obs < 2*np.sum(bools_fit): #2*len(P_sys):
                #print('Skipping N_obs = ', N_obs)
                K_cond_hat_random[i,n] = np.inf
                K_cond_hat_daily[i,n] = np.inf
                continue
            else:
                try:
                    #K_hat_all, sigma_K_all = fit_rv_Ks_multi_planet_model_GLS(t_obs_random, RV_obs_random, covarsc, P_sys[bools_fit], T0_sys[bools_fit], e_sys[bools_fit], omega_sys[bools_fit])
                    #K_cond_hat_random[i,n], sigma_K_cond_random[i,n] = K_hat_all[id_pl_cond_of_fits], sigma_K_all[id_pl_cond_of_fits]
                    K_hat_all, sigma_K_all = fit_rv_Ks_multi_planet_model_GLS(t_obs_daily, RV_obs_daily, covarsc, P_sys[bools_fit], T0_sys[bools_fit], e_sys[bools_fit], omega_sys[bools_fit])
                    K_cond_hat_daily[i,n], sigma_K_cond_daily[i,n] = K_hat_all[id_pl_cond_of_fits], sigma_K_all[id_pl_cond_of_fits]
                except:
                    print('##### Possible singular/non-positive-definite matrices; skipping N_obs = ', N_obs)
                    K_cond_hat_random[i,n] = np.inf
                    K_cond_hat_daily[i,n] = np.inf
        else:
            #K_cond_hat_random[i,n], sigma_K_cond_random[i,n] = fit_rv_K_single_planet_model_GLS(t_obs_random, RV_obs_random, covarsc, P_sys[id_pl_cond], T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond])
            K_cond_hat_daily[i,n], sigma_K_cond_daily[i,n] = fit_rv_K_single_planet_model_GLS(t_obs_daily, RV_obs_daily, covarsc, P_sys[id_pl_cond], T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond])

        # To plot the simulated observations:
        if n==0:
            pass
        else:
            continue

        fig = plt.figure(figsize=(16,8))
        plot = GridSpec(10,1,left=0.1,bottom=0.1,right=0.95,top=0.95,hspace=5)

        ax = plt.subplot(plot[1:3,0]) # gallery
        sc = plt.scatter(P_sys[det_sys == 1], np.ones(np.sum(det_sys == 1)), c=K_sys[det_sys == 1], s=50.*Rp_sys[det_sys == 1]**2., vmin=0., vmax=1.)
        plt.scatter(P_sys[det_sys == 0], np.ones(np.sum(det_sys == 0)), c=K_sys[det_sys == 0], edgecolors='r', s=50.*Rp_sys[det_sys == 0]**2., vmin=0., vmax=1.) #facecolors='none'
        plt.gca().set_xscale("log")
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xticks([3,10,30,100,300])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_yticks([])
        plt.xlim([2., 400.])
        plt.ylim([0.5, 1.5])
        plt.xlabel(r'Orbital period $P$ (days)', fontsize=16)

        cax = plt.subplot(plot[0,0])
        cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
        cax.tick_params(labelsize=16)
        cbar.set_label(r'RV semi-amplitude $K$ (m/s)', fontsize=16)

        ax = plt.subplot(plot[3:,0]) # RV time series
        plt.text(x=0.02, y=0.9, s=r'$K_{{\rm cond}} = {:0.2f}$ m/s'.format(K_cond), fontsize=20, transform=ax.transAxes)
        plt.errorbar(t_obs_daily, RV_obs_daily, yerr=np.sqrt(np.diag(covarsc)), fmt='.', capsize=5)
        #plt.plot(t_array, RV_sys, color='k', label='Total system')
        #plt.plot(t_array, RV_sys_per_pl[:,id_pl_cond], color='r', label=r'Conditioned planet ($K_{{\rm cond}} = {:0.2f}$ m/s)'.format(np.round(K_cond,2)))
        #plt.plot(t_array, [RV_true(t, K_cond_hat_daily[i,n], P_sys[id_pl_cond], T0_sys[id_pl_cond], e_sys[id_pl_cond], omega_sys[id_pl_cond]) for t in t_array], color='b', label=r'Conditioned planet fit (daily obs): $\hat{{K}}_{{\rm cond}} = {:0.2f}\pm{:0.2f}$ m/s'.format(np.round(K_cond_hat_daily[i,n], 2), np.round(sigma_K_cond_daily[i,n], 2)))
        plt.plot(t_obs_daily, [RV_true(t, K_cond_hat_daily[i,n], P_sys[id_pl_cond], T0_sys[id_pl_cond], e_sys[id_pl_cond], omega_sys[id_pl_cond]) for t in t_obs_daily], color='b', label=r'Conditioned planet fit: $\hat{{K}}_{{\rm cond}} = {:0.2f}\pm{:0.2f}$ m/s'.format(np.round(K_cond_hat_daily[i,n], 2), np.round(sigma_K_cond_daily[i,n], 2)))
        ax.tick_params(axis='both', labelsize=16)
        plt.xlim([0, t_obs_daily[-1]]) #[0, t_end]
        plt.ylim([-2.,2.5])
        plt.xlabel(r'Time $t$ (days)', fontsize=16)
        plt.ylabel(r'Radial velocity $v$ (m/s)', fontsize=16)
        plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=20)

        #plt.savefig(savefigures_directory + model_name + '_example_RV_fitting%s.png' % i)
        plt.show()
plt.close()


# To plot all the results:
fig = plt.figure(figsize=(16,8))

plot = GridSpec(10,1,left=0.075,bottom=0.1,right=0.475,top=0.95,hspace=2)

ax = plt.subplot(plot[1:3,0]) # gallery
sc = plt.scatter(P_sys[det_sys == 1], np.ones(np.sum(det_sys == 1)), c=K_sys[det_sys == 1], s=50.*Rp_sys[det_sys == 1]**2., vmin=0., vmax=5.)
plt.scatter(P_sys[det_sys == 0], np.ones(np.sum(det_sys == 0)), c=K_sys[det_sys == 0], edgecolors='r', s=50.*Rp_sys[det_sys == 0]**2., vmin=0., vmax=5.) #facecolors='none'
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=afs)
ax.set_xticks([3,10,30,100,300])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([])
plt.xlim([2., 400.])
plt.ylim([0.5, 1.5])
plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)

cax = plt.subplot(plot[0,0])
cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
cbar.set_label(r'RV semi-amplitude $K$ (m/s)', fontsize=10)

ax = plt.subplot(plot[3:,0]) # RV time series
plt.errorbar(t_obs_daily, RV_obs_daily, yerr=np.sqrt(np.diag(covarsc)), fmt='.', capsize=5)
plt.plot(t_array, RV_sys, color='k', label='Total system')
plt.plot(t_array, RV_sys_per_pl[:,id_pl_cond], color='r', label=r'Conditioned planet ($K_{{\rm cond}} = {:0.2f}$ m/s)'.format(np.round(K_cond,2)))
plt.plot(t_array, [RV_true(t, K_cond_hat_daily[-1,0], P_sys[id_pl_cond], T0_sys[id_pl_cond], e_sys[id_pl_cond], omega_sys[id_pl_cond]) for t in t_array], color='b', label=r'Conditioned planet fit (daily obs): $\hat{{K}}_{{\rm cond}} = {:0.2f}\pm{:0.2f}$ m/s'.format(np.round(K_cond_hat_daily[-1,0], 2), np.round(sigma_K_cond_daily[-1,0], 2)))
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0, t_end])
plt.xlabel(r'Time $t$ (days)', fontsize=tfs)
plt.ylabel(r'Radial velocity $v$ (m/s)', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize=lfs)

plot = GridSpec(4,1,left=0.55,bottom=0.1,right=0.975,top=0.95,hspace=0)

ax = plt.subplot(plot[0,0]) # Estimated K with error bars vs. N_obs
plt.axhline(K_cond, ls='--', color='k', label='True value')
plt.errorbar(N_obs_all, K_cond_hat_random[:,0], yerr=sigma_K_cond_random[:,0], fmt='o', color='r', label='Random obs')
plt.errorbar(N_obs_all, K_cond_hat_daily[:,0], yerr=sigma_K_cond_daily[:,0], fmt='o', color='b', label='Daily obs')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0, N_obs_all[-1]])
plt.xticks([])
plt.ylabel(r'$K_{\rm cond}$ (m/s)', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize=lfs)

ax = plt.subplot(plot[1,0]) # RMS deviation in K vs N_obs
rmsd_K_random_all = np.sqrt(np.mean((K_cond_hat_random - K_cond)**2., axis=1))
rmsd_K_daily_all = np.sqrt(np.mean((K_cond_hat_daily - K_cond)**2., axis=1))
rms_sigma_K_daily_all = np.sqrt(np.mean(sigma_K_cond_daily**2., axis=1))
plt.plot(N_obs_all, rmsd_K_random_all, 'o', color='r', label='Random obs')
plt.plot(N_obs_all, rmsd_K_daily_all, 'o', color='b', label='Daily obs')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0, N_obs_all[-1]])
plt.xticks([])
plt.ylabel(r'$\sqrt{\sum{(\hat{K} - K_{\rm true})^2}/N}$', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize=lfs)

ax = plt.subplot(plot[2,0]) # Mean uncertainty in K vs. N_obs
plt.plot(N_obs_all, sigma_K_cond_ideal_all, color='k', label='Ideal')
plt.plot(N_obs_all, np.mean(sigma_K_cond_random, axis=1), 'o', color='r', label='Random obs')
plt.plot(N_obs_all, np.mean(sigma_K_cond_daily, axis=1), 'o', color='b', label='Daily obs')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0, N_obs_all[-1]])
plt.xticks([])
plt.ylabel(r'$\sigma_{K_{\rm cond}}$ (m/s)', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize=lfs)

ax = plt.subplot(plot[3,0]) # Mean fractional uncertainty in K vs. N_obs
plt.plot(N_obs_all, sigma_K_cond_ideal_all/K_sys[id_pl_cond], color='k', label='Ideal')
plt.plot(N_obs_all, np.mean(sigma_K_cond_random/K_cond_hat_random, axis=1), 'o', color='r', label='Random obs')
plt.plot(N_obs_all, np.mean(sigma_K_cond_daily/K_cond_hat_daily, axis=1), 'o', color='b', label='Daily obs')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0, N_obs_all[-1]])
plt.xlabel(r'Number of observations', fontsize=tfs)
plt.ylabel(r'$\sigma_{K_{\rm cond}}/\hat{K}_{\rm cond}$', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize=lfs)

plt.show()

stop = time.time()

i_N_obs_50p = rmsd_K_daily_all/K_cond < 0.5
i_N_obs_30p = rmsd_K_daily_all/K_cond < 0.3
i_N_obs_20p = rmsd_K_daily_all/K_cond < 0.2
i_N_obs_10p = rmsd_K_daily_all/K_cond < 0.1
i_N_obs_5p = rmsd_K_daily_all/K_cond < 0.05
N_obs_min_50p = N_obs_all[i_N_obs_50p][0] if np.sum(i_N_obs_50p) > 0 else np.nan
N_obs_min_30p = N_obs_all[i_N_obs_30p][0] if np.sum(i_N_obs_30p) > 0 else np.nan
N_obs_min_20p = N_obs_all[i_N_obs_20p][0] if np.sum(i_N_obs_20p) > 0 else np.nan
N_obs_min_10p = N_obs_all[i_N_obs_10p][0] if np.sum(i_N_obs_10p) > 0 else np.nan
N_obs_min_5p = N_obs_all[i_N_obs_5p][0] if np.sum(i_N_obs_5p) > 0 else np.nan
rms_sigma_K_50p = rms_sigma_K_daily_all[i_N_obs_50p][0] if np.sum(i_N_obs_50p) > 0 else np.nan
rms_sigma_K_30p = rms_sigma_K_daily_all[i_N_obs_30p][0] if np.sum(i_N_obs_30p) > 0 else np.nan
rms_sigma_K_20p = rms_sigma_K_daily_all[i_N_obs_20p][0] if np.sum(i_N_obs_20p) > 0 else np.nan
rms_sigma_K_10p = rms_sigma_K_daily_all[i_N_obs_10p][0] if np.sum(i_N_obs_10p) > 0 else np.nan
rms_sigma_K_5p = rms_sigma_K_daily_all[i_N_obs_5p][0] if np.sum(i_N_obs_5p) > 0 else np.nan

print('{:d} ({:0.1f}s): K_cond = {:0.3f} m/s --- K_max = {:0.3f} m/s --- K_cond/sum(K) = {:0.3f} --- N_obs for RMSD(K_cond)/K_cond <50%, <30%, <20%, <10%, <5%: {:.0f}, {:.0f}, {:.0f}, {:.0f}, {:.0f} --- best error = {:0.3f}'.format(id_sys, stop-start, K_cond, K_max, K_cond/np.sum(K_sys), N_obs_min_50p, N_obs_min_30p, N_obs_min_20p, N_obs_min_10p, N_obs_min_5p, np.min(rmsd_K_daily_all/K_cond)))
