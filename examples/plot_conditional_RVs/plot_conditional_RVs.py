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
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Conditional_P8_12d_R1p8_2_transiting/' #'Conditional_P8_12d_R1p8_2_transiting/'; 'Conditional_Venus_transiting/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Systems_conditional/Conditional_P8_12d_R1p8_2_transiting/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [8.,12.], [1.8,2.0], [0.,np.inf]
#P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [215.,235.], [0.9,1.0], [0.77,0.86] # Venus
det = True
conds = conditionals_dict(P_cond_bounds=P_cond_bounds, Rp_cond_bounds=Rp_cond_bounds, Mp_cond_bounds=Mp_cond_bounds, det=det)

# To load and combine multiple simulated catalogs:
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
#sssp_per_sys, sssp = load_cat_phys_multiple_and_compute_combine_summary_stats(loadfiles_directory, run_numbers=range(1,101), load_full_tables=True)





##### To simulate and fit RV observations of systems conditioned on a given planet, to see how the measured K varies with number of observations:

N_obs_all = np.array([int(round(x)) for x in np.logspace(np.log10(5), np.log10(300), 20)])
σ_1obs = 0.3
N_sample, repeat = 1000, 100

#'''
fname = 'RV_obs_N%s_repeat%s_20Nobs5to300_sigma0p3_cases.txt' % (N_sample, repeat)
outputs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('i4','i4')+('f8',)*33)

# To also simulate and fit RV observations of single planet systems, to see how the measured K varies with number of observations:
K_array = np.logspace(np.log10(0.05), np.log10(10.), 100) # m/s
alpha_P, sigma_ecc = 0., 0.25
fname = 'RV_obs_P8_12d_singles_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (len(K_array), repeat)
outputs_single_planet_RVs = np.genfromtxt(loadfiles_directory + '../RV_obs_singles/' + fname, names=True, dtype=('f8','f8','f8'))
#'''
#'''
# K_cond/sum(K) vs. K_cond plot:
fig = plt.figure(figsize=(10,8))
plot = GridSpec(1,1,left=0.12,bottom=0.12,right=0.8,top=0.95) # main plot
ax = plt.subplot(plot[:,:])
plt.title(r'$P_{\rm cond} = %s$d, $R_{p,\rm cond} = %s R_\oplus$' % (P_cond_bounds, Rp_cond_bounds), fontsize=20)
cmap = matplotlib.cm.viridis
cmap.set_bad('r')
sc = plt.scatter(outputs['K_cond'], outputs['K_cond']/outputs['K_sum'], c=outputs['N_obs_min_20p'], cmap=cmap, norm=matplotlib.colors.LogNorm(), plotnonfinite=True)
plt.axvline(x=σ_1obs, ls=':', color='r', label=r'Single measurement noise ($\sigma_{1,\rm obs}$)')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.set_xticks([0.1,0.2,0.3,0.5,1.,2.,3.,5.,10.]) # [0.5, 1., 2., 4., 8., 16.]
ax.set_yticks([0.1,0.2,0.3,0.4,0.5,1.])
plt.minorticks_off()
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlim([0.05, 10.])
plt.ylim([0.05, 1.])
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$K_{\rm cond}$ (m/s)', fontsize=20)
plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=20)
plot = GridSpec(1,1,left=0.81,bottom=0.12,right=0.85,top=0.95) # side panel colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [5,10,20,50,100,200,300,500]
cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter())
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=20)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_R%s_%s_%sNobs%sto%s_sigma%s_Kcondfrac_vs_Kcond.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()
plt.show()

# N_obs_min vs K_cond plot:
log_Nobs_norm, slope = fit_line_loglog_Nobs_K_single_planets(outputs_single_planet_RVs, σ_1obs, [2., -2.]) # fit a line to the ideal case simulations
Nobs_norm = 10.**log_Nobs_norm

Nobs_ideal_Karray = linear_logNobs_logK(K_array, σ_1obs, Nobs_norm, slope, round_to_ints=False) # ideal N_obs at K_array points
Nobs_ideal_Kcond = linear_logNobs_logK(outputs['K_cond'], σ_1obs, Nobs_norm, slope, round_to_ints=True) # ideal N_obs at K_cond points

fig = plt.figure(figsize=(10,8))
plot = GridSpec(1,5,left=0.15,bottom=0.12,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:-1]) # main plot
plt.title(r'$P_{\rm cond} = %s$d, $R_{p,\rm cond} = %s R_\oplus$' % (P_cond_bounds, Rp_cond_bounds), fontsize=20)
plt.loglog(outputs['K_cond'], outputs['N_obs_min_20p'], 'o', color='k', label='Conditioned\nplanets')
#plt.loglog(outputs_single_planet_RVs['K'], outputs_single_planet_RVs['N_obs_min_20p'], 'o', color='r')
plt.plot(K_array[Nobs_ideal_Karray > 5], Nobs_ideal_Karray[Nobs_ideal_Karray > 5], '-', lw=3, color='r', label='Ideal')
plt.axvline(x=σ_1obs, ls=':', color='r', label=r'$\sigma_{1,\rm obs}$')
yticks_custom = [5,10,20,50,100,200,500]
ax.set_xticks([0.1,0.2,0.5,1.,2.,5.,10.])
ax.set_yticks(yticks_custom)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlim([0.05,20.])
plt.ylim([4,400])
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$K_{\rm cond}$ (m/s)', fontsize=20)
plt.ylabel(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=20)
plt.legend(loc='lower left', bbox_to_anchor=(0,0), ncol=1, frameon=False, fontsize=16)
ax = plt.subplot(plot[:,-1]) # side panel CDFs
K_cond_bins = [0.,0.5,1.,2.,4.,8.,np.inf]
N_obs_min_20p_per_K_cond_bin = [outputs['N_obs_min_20p'][(K_cond_bins[i] <= outputs['K_cond']) & (outputs['K_cond'] <= K_cond_bins[i+1])] for i in range(len(K_cond_bins)-1)]
N_obs_min_20p_per_K_cond_bin_single_planet = [Nobs_ideal_Kcond[(K_cond_bins[i] <= outputs['K_cond']) & (outputs['K_cond'] <= K_cond_bins[i+1])] for i in range(len(K_cond_bins)-1)]
color_bins = ['k','b','g','r','orange','m']
label_bins = [r'$K_{\rm cond} < %s$' % K_cond_bins[1]] + [r'$%s \leq K_{\rm cond} < %s$' % (K_cond_bins[i], K_cond_bins[i+1]) for i in range(1,len(K_cond_bins)-2)] + [r'$K_{\rm cond} > %s$ m/s' % K_cond_bins[-2]]
for i,x in enumerate(N_obs_min_20p_per_K_cond_bin):
    plt.plot((np.arange(len(x))+1.)/np.float(len(x)), np.sort(x), drawstyle='steps-post', color=color_bins[i], ls='-', lw=1, label=label_bins[i])
for i,x in enumerate(N_obs_min_20p_per_K_cond_bin_single_planet):
    plt.plot((np.arange(len(x))+1.)/np.float(len(x)), np.sort(x), drawstyle='steps-post', color=color_bins[i], ls='--', lw=1)
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelleft=False, labelsize=20)
plt.xlim([0,1])
plt.ylim([4,400])
plt.xlabel('CDF', fontsize=20)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc='upper right', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=16)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_R%s_%s_%sNobs%sto%s_sigma%s_Nobs_vs_Kcond.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()
plt.show()

# Quantiles for N_obs_min vs. middle of K_cond bins:
K_cond_bins_mid = np.logspace(np.log10(0.05), np.log10(10.), 101)
K_cond_bins_halfwidth = np.sqrt(4.) # multiplicative factor; log10 of this value gives the bin half-width in log(K_cond)
N_obs_min_20p_qtls_per_K_cond_bin = np.zeros((len(K_cond_bins_mid),3))
N_obs_min_20p_qtls_per_K_cond_bin_single_planet = np.zeros((len(K_cond_bins_mid),3))
for i,K_mid in enumerate(K_cond_bins_mid):
    N_obs_min_20p_bin = outputs['N_obs_min_20p'][(K_mid/K_cond_bins_halfwidth <= outputs['K_cond']) & (outputs['K_cond'] <= K_mid*K_cond_bins_halfwidth)]
    N_obs_min_20p_bin_single_planet = outputs_single_planet_RVs['N_obs_min_20p'][(K_mid/K_cond_bins_halfwidth <= outputs_single_planet_RVs['K']) & (outputs_single_planet_RVs['K'] <= K_mid*K_cond_bins_halfwidth)]
    N_obs_min_20p_bin = N_obs_min_20p_bin[~np.isnan(N_obs_min_20p_bin)]
    if len(N_obs_min_20p_bin) > 0:
        N_obs_min_20p_qtls_per_K_cond_bin[i] = np.quantile(N_obs_min_20p_bin, [0.16,0.5,0.84])
    else:
        N_obs_min_20p_qtls_per_K_cond_bin[i] = [np.nan, np.nan, np.nan]
    N_obs_min_20p_bin_single_planet = N_obs_min_20p_bin_single_planet[~np.isnan(N_obs_min_20p_bin_single_planet)]
    if len(N_obs_min_20p_bin_single_planet) > 0:
        N_obs_min_20p_qtls_per_K_cond_bin_single_planet[i] = np.quantile(N_obs_min_20p_bin_single_planet, [0.16,0.5,0.84])
    else:
        N_obs_min_20p_qtls_per_K_cond_bin_single_planet[i] = [np.nan, np.nan, np.nan]
    
fig = plt.figure(figsize=(10,8))
plot = GridSpec(1,5,left=0.15,bottom=0.12,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
plt.title(r'$P_{\rm cond} = %s$d, $R_{p,\rm cond} = %s R_\oplus$' % (P_cond_bounds, Rp_cond_bounds), fontsize=20)
plt.loglog(K_cond_bins_mid, N_obs_min_20p_qtls_per_K_cond_bin[:,0], ls='--', color='k', label=r'16%-84% quantiles')
plt.loglog(K_cond_bins_mid, N_obs_min_20p_qtls_per_K_cond_bin[:,1], ls='-', color='k', label='Simulated systems')
plt.loglog(K_cond_bins_mid, N_obs_min_20p_qtls_per_K_cond_bin[:,2], ls='--', color='k')
plt.loglog(K_cond_bins_mid, N_obs_min_20p_qtls_per_K_cond_bin_single_planet[:,0], ls='--', color='r')
plt.loglog(K_cond_bins_mid, N_obs_min_20p_qtls_per_K_cond_bin_single_planet[:,1], ls='-', color='r', label='Single planet case')
plt.loglog(K_cond_bins_mid, N_obs_min_20p_qtls_per_K_cond_bin_single_planet[:,2], ls='--', color='r')
plt.axvline(x=σ_1obs, ls=':', color='r', label=r'Single measurement noise ($\sigma_{1,\rm obs}$)')
yticks_custom = [5,10,20,50,100,200,500]
ax.set_xticks([0.1,0.2,0.3,0.5,1.,2.,3.,5.,10.])
ax.set_yticks(yticks_custom)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlim([0.05,20.])
plt.ylim([4,400])
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$K_{\rm cond}$ (m/s)', fontsize=20)
plt.ylabel(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=16)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_R%s_%s_%sNobs%sto%s_sigma%s_Nobs_vs_Kcond_qtls.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()
plt.show()
#'''





##### No conditioning on radius: remake above plots (K_cond/sum(K) vs. K_cond, N_obs vs K_cond) replacing K_cond with Rp_cond:
'''
fname = 'RV_obs_P8_12d_N%s_repeat%s_20Nobs5to300_sigma1.txt' % (N_sample, repeat)
outputs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('i4','f8','f8','f8','f8','f8','f8','f8'))

# K_cond/sum(K) vs. R_p plot:
fig = plt.figure(figsize=(10,8))
plot = GridSpec(1,1,left=0.12,bottom=0.12,right=0.8,top=0.95) # main plot
ax = plt.subplot(plot[:,:])
plt.title(r'$P_{\rm cond} = %s$d' % P_cond_bounds, fontsize=20)
cmap = matplotlib.cm.viridis
cmap.set_bad('r')
sc = plt.scatter(outputs['Rp_cond'], outputs['K_cond']/outputs['K_sum'], c=outputs['N_obs_min_20p'], cmap=cmap, norm=matplotlib.colors.LogNorm(), plotnonfinite=True)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.set_xticks([0.5,1.,2.,3.,5.,10.])
ax.set_yticks([0.1,0.2,0.3,0.4,0.5,1.])
plt.minorticks_off()
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlim([0.5, 10.])
plt.ylim([0.05, 1.])
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$R_{p,\rm cond}$ ($R_\oplus$)', fontsize=20)
plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=20)
plot = GridSpec(1,1,left=0.81,bottom=0.12,right=0.85,top=0.95) # side panel colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [5,10,20,50,100,200,300,500]
cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter()) #ticks=N_obs_all,
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=20)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_%sNobs%sto%s_sigma%s_Kcondfrac_vs_Rpcond.pdf' % (P_cond_bounds[0], P_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()
plt.show()

# N_obs_min vs R_p plot:
fig = plt.figure(figsize=(10,8))
plot = GridSpec(1,5,left=0.15,bottom=0.12,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:-1]) # main plot
plt.title(r'$P_{\rm cond} = %s$d' % P_cond_bounds, fontsize=20)
plt.loglog(outputs['Rp_cond'], outputs['N_obs_min_20p'], 'o', color='k')
yticks_custom = [5,10,20,50,100,200,500]
ax.set_xticks([0.5,1.,2.,3.,5.,10.])
ax.set_yticks(yticks_custom)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlim([0.5,20.])
plt.ylim([4,400])
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$R_{p,\rm cond}$ ($R_\oplus$)', fontsize=20)
plt.ylabel(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=20)
ax = plt.subplot(plot[:,-1]) # side panel CDFs
Rp_cond_bins = [0.5,1.5,3.,4.5,np.inf]
N_obs_min_20p_per_Rp_cond_bin = [outputs['N_obs_min_20p'][(Rp_cond_bins[i] <= outputs['Rp_cond']) & (outputs['Rp_cond'] <= Rp_cond_bins[i+1])] for i in range(len(Rp_cond_bins)-1)]
color_bins = ['k','b','g','r','orange','m']
label_bins = [r'$R_{p,\rm cond} < %s$' % Rp_cond_bins[1]] + [r'$%s \leq R_{p,\rm cond} < %s$' % (Rp_cond_bins[i], Rp_cond_bins[i+1]) for i in range(1,len(Rp_cond_bins)-2)] + [r'$R_{p,\rm cond} > %s$' % Rp_cond_bins[-2]]
for i,x in enumerate(N_obs_min_20p_per_Rp_cond_bin):
    plt.plot((np.arange(len(x))+1.)/np.float(len(x)), np.sort(x), drawstyle='steps-post', color=color_bins[i], ls='-', lw=1, label=label_bins[i])
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelleft=False, labelsize=20)
plt.xlim([0,1])
plt.ylim([4,400])
plt.xlabel('CDF', fontsize=20)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc='upper right', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=16)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_%sNobs%sto%s_sigma%s_Nobs_vs_Rpcond.pdf' % (P_cond_bounds[0], P_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()
plt.show()
'''





# K_cond vs. R_p plot:
'''
fig = plt.figure(figsize=(10,8))
plot = GridSpec(1,1,left=0.12,bottom=0.12,right=0.8,top=0.95) # main plot
ax = plt.subplot(plot[:,:])
plt.title(r'$P_{\rm cond} = %s$d' % P_cond_bounds, fontsize=20)
cmap = matplotlib.cm.viridis
cmap.set_bad('r')
sc = plt.scatter(outputs['Rp_cond'], outputs['K_cond'], c=outputs['N_obs_min_20p'], cmap=cmap, norm=matplotlib.colors.LogNorm(), plotnonfinite=True)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.set_xticks([0.5,1.,2.,3.,5.,10.])
ax.set_xticks([0.5,1.,2.,3.,5.,10.])
#plt.minorticks_off()
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlim([0.5, 10.])
plt.ylim([0.05, 20.])
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$R_{p,\rm cond}$ ($R_\oplus$)', fontsize=20)
plt.ylabel(r'$K_{\rm cond}$ (m/s)', fontsize=20)
plot = GridSpec(1,1,left=0.81,bottom=0.12,right=0.85,top=0.95) # side panel colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [5,10,20,50,100,200,300,500]
cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter()) #ticks=N_obs_all,
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=20)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_%sNobs%sto%s_sigma%s_Kcond_vs_Rpcond.pdf' % (P_cond_bounds[0], P_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()
plt.show()
'''





##### To save the data for a sample of systems as a LaTeX-formatted table file:
table_array = generate_latex_table_RVobs_systems_conditional(outputs)
fname = 'table_RV_obs.txt'
#np.savetxt(loadfiles_directory + fname, table_array, fmt='%s')





##### Test plotting N_obs ratios vs some system properties:

i_out = outputs['id_sys']
N_obs_ratios = outputs['N_obs_min_20p']/outputs['N_obs_min_20p_ideal']
N_obs_ratios = N_obs_ratios[outputs['n_pl'] > 1]

P_ratios_largestK = []
K_ratios_largestK = []
P_ratios_closestP = []
K_ratios_closestP = []
sum_K_ratios = []
sum_K_in_ratios = []
sum_K_out_ratios = []
Kweighted_sum_P_ratios = []
Kweighted_absdiff_logP = []
# other ideas: sum of K's (interior vs exterior)? P-weighted sum of K's (interior vs exterior)?
for i,id_sys in enumerate(i_out):
    if outputs['n_pl'][i] > 1:
        Mstar_sys = sssp['Mstar_all'][id_sys]
        P_sys = sssp_per_sys['P_all'][id_sys]
        det_sys = sssp_per_sys['det_all'][id_sys]
        Mp_sys = sssp_per_sys['mass_all'][id_sys]
        Rp_sys = sssp_per_sys['radii_all'][id_sys]
        e_sys = sssp_per_sys['e_all'][id_sys]
        incl_sys = sssp_per_sys['incl_all'][id_sys]
        
        det_sys = det_sys[P_sys > 0]
        Mp_sys = Mp_sys[P_sys > 0]
        Rp_sys = Rp_sys[P_sys > 0]
        e_sys = e_sys[P_sys > 0]
        incl_sys = incl_sys[P_sys > 0]
        P_sys = P_sys[P_sys > 0]
        
        K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=Mstar_sys)
        id_pl_cond = np.arange(len(P_sys))[(P_sys > conds['P_lower']) & (P_sys < conds['P_upper']) & (Rp_sys > conds['Rp_lower']) & (Rp_sys < conds['Rp_upper'])][0] # index of conditioned planet
        K_cond = outputs['K_cond'][i]
        P_cond = outputs['P_cond'][i]
        
        K_diff_cond = K_sys - K_cond
        K_others = K_sys[np.arange(len(P_sys)) != id_pl_cond]
        P_others = P_sys[np.arange(len(P_sys)) != id_pl_cond]
        #print('K_diffs:', K_diff_cond, ' (K_all: ', K_sys, '; K_cond = ', K_cond)
        #print('K_others:', K_others)
        
        P_ratios_largestK.append(P_others[K_others == np.max(K_others)] / P_cond)
        K_ratios_largestK.append(K_others[K_others == np.max(K_others)] / K_cond)
        
        P_ratios_closestP.append(P_others[np.argmin(np.abs(np.log10(P_others) - np.log10(P_cond)))] / P_cond)
        #K_ratios_closestP.append(K_others[np.argmin(np.abs(P_others - P_cond))] / K_cond) # closest P in linear space
        K_ratios_closestP.append(K_others[np.argmin(np.abs(np.log10(P_others) - np.log10(P_cond)))] / K_cond) # closest P in log space
        
        sum_K_ratios.append(np.sum(K_others)/K_cond)
        sum_K_in_ratios.append(np.sum(K_others[P_others < P_cond])/K_cond)
        sum_K_out_ratios.append(np.sum(K_others[P_others > P_cond])/K_cond)
        Kweighted_sum_P_ratios.append(10.**(np.sum(K_others*np.log10(P_others/P_cond))/np.sum(K_others))) # Problem with this metric: planets interior and exterior average out (e.g. P-ratios of 0.5+2 is the same value as P-ratios of 0.1+10)
        Kweighted_absdiff_logP.append(np.sum(K_others*np.abs(np.log10(P_others) - np.log10(P_cond)))/np.sum(K_others))

P_ratios_largestK = np.array(P_ratios_largestK).flatten()
K_ratios_largestK = np.array(K_ratios_largestK).flatten()
P_ratios_closestP = np.array(P_ratios_closestP).flatten()
K_ratios_closestP = np.array(K_ratios_closestP).flatten()
sum_K_ratios = np.array(sum_K_ratios).flatten()
sum_K_in_ratios = np.array(sum_K_in_ratios).flatten()
sum_K_out_ratios = np.array(sum_K_out_ratios).flatten()
Kweighted_sum_P_ratios = np.array(Kweighted_sum_P_ratios).flatten()
Kweighted_absdiff_logP = np.array(Kweighted_absdiff_logP).flatten()



# K_cond/sum(K):
'''
x = (outputs['K_cond']/outputs['K_sum'])[outputs['n_pl'] > 1]
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95)
ax = plt.subplot(plot[:,:])
plt.scatter(x, N_obs_ratios, color='b')
plt.scatter(x[np.isnan(N_obs_ratios)], (1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], color='r')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$K_{\rm cond}/\sum{K}$', fontsize=20)
plt.ylabel(r'$N_{\rm obs}/N_{\rm obs,ideal}$', fontsize=20)

# K_cond/K_max:
x = (outputs['K_cond']/outputs['K_max'])[outputs['n_pl'] > 1]
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95)
ax = plt.subplot(plot[:,:])
plt.scatter(x, N_obs_ratios, color='b')
plt.scatter(x[np.isnan(N_obs_ratios)], (1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], color='r')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$K_{\rm cond}/K_{\rm max}$', fontsize=20)
plt.ylabel(r'$N_{\rm obs}/N_{\rm obs,ideal}$', fontsize=20)

# Period ratio of next largest K planet to K_cond planet:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95)
ax = plt.subplot(plot[:,:])
plt.scatter(P_ratios_largestK, N_obs_ratios, color='b')
plt.scatter(P_ratios_largestK[np.isnan(N_obs_ratios)], (1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], color='r')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$P_{K_{\rm max}}/P_{\rm cond}$', fontsize=20)
plt.ylabel(r'$N_{\rm obs}/N_{\rm obs,ideal}$', fontsize=20)

# K ratio of nearest period planet to K_cond planet
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95)
ax = plt.subplot(plot[:,:])
plt.scatter(K_ratios_closestP, N_obs_ratios, color='b')
plt.scatter(K_ratios_closestP[np.isnan(N_obs_ratios)], (1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], color='r')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=20)
plt.xlabel(r'$K_{\rm nearest}/K_{\rm cond}$', fontsize=20)
plt.ylabel(r'$N_{\rm obs}/N_{\rm obs,ideal}$', fontsize=20)

plt.show()
'''



# K_ratio vs P_ratio with N_obs_ratio colorscale:
fig = plt.figure(figsize=(6,9))
plot = GridSpec(2,1,left=0.2,bottom=0.1,right=0.8,top=0.95)
plt.figtext(0.5, 0.96, r'$P_{\rm cond} = %s$d, $R_{p,\rm cond} = %s R_\oplus$' % (P_cond_bounds, Rp_cond_bounds), va='bottom', ha='center', fontsize=16)
#plt.figtext(0.5, 0.96, r'Transiting Venus-like planets', va='bottom', ha='center', fontsize=16)

cmap = matplotlib.cm.coolwarm
cmap.set_bad('r')
vmin, vmax = 0.5, np.nanmax(N_obs_ratios) #np.nanmin(N_obs_ratios), np.nanmax(N_obs_ratios)
xmin, xmax = 0.2, 50. #0.2, 50. #0.01, 2.
ymin, ymax = 0.005, 100. #0.005, 100. #0.1, 500.
xticks_custom = [0.01,0.1,1.,10.,100.]
yticks_custom = [0.01,0.1,1.,10.,100.]

ax = plt.subplot(plot[0,:]) # for next largest K planet to K_cond planet
#sc = plt.scatter(P_ratios_largestK, K_ratios_largestK, c=N_obs_ratios, cmap=cmap, norm=matplotlib.colors.LogNorm(), plotnonfinite=True)
sc = plt.scatter(P_ratios_largestK, K_ratios_largestK, c=N_obs_ratios, cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.scatter(P_ratios_largestK[np.isnan(N_obs_ratios)], K_ratios_largestK[np.isnan(N_obs_ratios)], marker='^', c=(1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
#ax.set_xticks(xticks_custom)
#ax.set_yticks(yticks_custom)
#plt.minorticks_off()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$P_{K_{\rm max,others}}/P_{\rm cond}$', fontsize=16)
plt.ylabel(r'$K_{\rm max,others}/K_{\rm cond}$', fontsize=16)

ax = plt.subplot(plot[1,:]) # for nearest log-period planet to K_cond planet
#sc = plt.scatter(P_ratios_closestP, K_ratios_closestP, c=N_obs_ratios, cmap=cmap, norm=matplotlib.colors.LogNorm(), plotnonfinite=True)
sc = plt.scatter(P_ratios_closestP, K_ratios_closestP, c=N_obs_ratios, cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.scatter(P_ratios_closestP[np.isnan(N_obs_ratios)], K_ratios_closestP[np.isnan(N_obs_ratios)], marker='^', c=(1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
#ax.set_xticks(xticks_custom)
#ax.set_yticks(yticks_custom)
#plt.minorticks_off()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$P_{\rm nearest}/P_{\rm cond}$', fontsize=16)
plt.ylabel(r'$K_{\rm nearest}/K_{\rm cond}$', fontsize=16)

plot = GridSpec(1,1,left=0.82,bottom=0.1,right=0.85,top=0.95) # colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [0.5,1.,2.,5.,10.,20.,50.]
cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter())
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$N_{\rm obs}/N_{\rm obs,ideal}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=16)

if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_R%s_%s_%sNobs%sto%s_sigma%s_Kratios_Pratios_Nobsratios.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()



# K_ratio (next largest K planet) vs K_ratio (nearest planet) with N_obs_ratio colorscale, and likewise for P_ratio (next largest K planet) vs P_ratio (nearest planet):
fig = plt.figure(figsize=(6,9))
plot = GridSpec(2,1,left=0.25,bottom=0.1,right=0.8,top=0.95)
plt.figtext(0.5, 0.96, r'$P_{\rm cond} = %s$d, $R_{p,\rm cond} = %s R_\oplus$' % (P_cond_bounds, Rp_cond_bounds), va='bottom', ha='center', fontsize=16)
#plt.figtext(0.5, 0.96, r'Transiting Venus-like planets', va='bottom', ha='center', fontsize=16)

ax = plt.subplot(plot[0,:]) # for K_ratios
sc = plt.scatter(K_ratios_closestP, K_ratios_largestK, c=N_obs_ratios, cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.scatter(K_ratios_closestP[np.isnan(N_obs_ratios)], K_ratios_largestK[np.isnan(N_obs_ratios)], marker='^', c=(1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.xlim([ymin, ymax])
plt.ylim([ymin, ymax])
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$K_{\rm nearest}/K_{\rm cond}$', fontsize=16)
plt.ylabel(r'$K_{\rm max,others}/K_{\rm cond}$', fontsize=16)

ax = plt.subplot(plot[1,:]) # for sum(K_in) vs sum(K_out) ### for P_ratios
#x, y = P_ratios_closestP, P_ratios_largestK
x, y = sum_K_in_ratios, sum_K_out_ratios
sc = plt.scatter(x, y, c=N_obs_ratios, cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.scatter(x[np.isnan(N_obs_ratios)], y[np.isnan(N_obs_ratios)], marker='^', c=(1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
#plt.xlim([xmin, xmax])
#plt.ylim([xmin, xmax])
plt.xlim([ymin, ymax])
plt.ylim([ymin, ymax])
ax.tick_params(axis='both', labelsize=16)
#plt.xlabel(r'$P_{\rm nearest}/P_{\rm cond}$', fontsize=16)
#plt.ylabel(r'$P_{K_{\rm max,others}}/P_{\rm cond}$', fontsize=16)
plt.xlabel(r'$\sum_{\rm interior}{K_i}/K_{\rm cond}$', fontsize=16)
plt.ylabel(r'$\sum_{\rm exterior}{K_i}/K_{\rm cond}$', fontsize=16)

plot = GridSpec(1,1,left=0.82,bottom=0.1,right=0.85,top=0.95) # colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [0.5,1.,2.,5.,10.,20.,50.]
cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter())
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$N_{\rm obs}/N_{\rm obs,ideal}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=16)

if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_R%s_%s_%sNobs%sto%s_sigma%s_KKratios_PPratios_Nobsratios.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()



# sum(K_ratios) vs weighted sum of absolute differences in logP, with N_obs_ratio colorscale:
fig = plt.figure(figsize=(6,9))
plot = GridSpec(2,1,left=0.25,bottom=0.1,right=0.8,top=0.95)
plt.figtext(0.5, 0.96, r'$P_{\rm cond} = %s$d, $R_{p,\rm cond} = %s R_\oplus$' % (P_cond_bounds, Rp_cond_bounds), va='bottom', ha='center', fontsize=16)
#plt.figtext(0.5, 0.96, r'Transiting Venus-like planets', va='bottom', ha='center', fontsize=16)

ax = plt.subplot(plot[0,:])
x, y = Kweighted_sum_P_ratios, sum_K_ratios
sc = plt.scatter(x, y, c=N_obs_ratios, cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.scatter(x[np.isnan(N_obs_ratios)], y[np.isnan(N_obs_ratios)], marker='^', c=(1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$10^{[\sum_{\rm others}{K_i \log(P_i/P_{\rm cond})}/K_{\rm cond}]}$', fontsize=16)
plt.ylabel(r'$\sum_{\rm others}{K_i}/K_{\rm cond}$', fontsize=16)

ax = plt.subplot(plot[1,:])
x, y = Kweighted_absdiff_logP, sum_K_ratios
sc = plt.scatter(x, y, c=N_obs_ratios, cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.scatter(x[np.isnan(N_obs_ratios)], y[np.isnan(N_obs_ratios)], marker='^', c=(1000./outputs['N_obs_min_20p_ideal'])[outputs['n_pl'] > 1][np.isnan(N_obs_ratios)], cmap=cmap, norm=MidPointLogNorm(vmin=vmin,vmax=vmax,midpoint=1.))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.xlim([0.1, 2.])
plt.ylim([ymin, ymax])
ax.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\sum_{\rm others}{K_i |\log(P_i) - \log(P_{\rm cond})|}/K_{\rm cond}$', fontsize=16)
plt.ylabel(r'$\sum_{\rm others}{K_i}/K_{\rm cond}$', fontsize=16)

plot = GridSpec(1,1,left=0.82,bottom=0.1,right=0.85,top=0.95) # colorbar
cax = plt.subplot(plot[:,:])
cticks_custom = [0.5,1.,2.,5.,10.,20.,50.]
cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter())
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$N_{\rm obs}/N_{\rm obs,ideal}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=16)

if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_R%s_%s_%sNobs%sto%s_sigma%s_sumKratios_wsumPratios_Nobsratios.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1], σ_1obs)
    plt.savefig(fig_name)
    plt.close()
plt.show()
