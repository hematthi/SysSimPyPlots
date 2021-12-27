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
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Conditional_P8_12d_R1p8_2_transiting/' #'Conditional_P8_12d_R3_4_transiting/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Systems_conditional/Conditional_P8_12d_R1p8_2_transiting/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number

P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [8.,12.], [1.8,2.0], [0.,np.inf]
#P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [215.,235.], [0.9,1.0], [0.77,0.86] # Venus
det = True
conds = conditionals_dict(P_cond_bounds=P_cond_bounds, Rp_cond_bounds=Rp_cond_bounds, Mp_cond_bounds=Mp_cond_bounds, det=det)





##### To simulate and fit RV observations of systems conditioned on a given planet, to see how the measured K varies with number of observations:

N_obs_all = np.array([int(round(x)) for x in np.logspace(np.log10(5), np.log10(300), 20)])
N_sample, repeat = 1000, 100

σ_1obs_all = [0.1, 0.3, 1.]

#'''
fname_all = ['RV_obs_N%s_repeat%s_20Nobs5to300_sigma0p1.txt' % (N_sample, repeat), 'RV_obs_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (N_sample, repeat), 'RV_obs_N%s_repeat%s_20Nobs5to300_sigma1.txt' % (N_sample, repeat)]
outputs_all = []
for fname in fname_all:
    outputs = np.genfromtxt(loadfiles_directory + fname, names=True)
    #outputs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('i4','f8','f8','f8','f8','f8'))
    #outputs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('i4','f8','f8','f8','f8','f8','f8','f8'))
    outputs_all.append(outputs)

# To also simulate and fit RV observations of single planet systems, to see how the measured K varies with number of observations:
K_array = np.logspace(np.log10(0.05), np.log10(10.), 100) # m/s
alpha_P, sigma_ecc = 0., 0.25
fname_all = ['RV_obs_P8_12d_singles_N%s_repeat%s_20Nobs5to300_sigma0p1.txt' % (len(K_array), repeat), 'RV_obs_P8_12d_singles_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (len(K_array), repeat), 'RV_obs_P8_12d_singles_N%s_repeat%s_20Nobs5to300_sigma1.txt' % (len(K_array), repeat)]
outputs_ideal_all = []
Nobs_norm_all, slope_all = [], []
for i,fname in enumerate(fname_all):
    outputs_ideal = np.genfromtxt(loadfiles_directory + '../RV_obs_singles/' + fname, names=True, dtype=('f8','f8','f8'))
    outputs_ideal_all.append(outputs_ideal)
    
    log_Nobs_norm, slope = fit_line_loglog_Nobs_K_single_planets(outputs_ideal, σ_1obs_all[i], [2., -2.]) # fit a line to the ideal case simulations
    Nobs_norm = 10.**log_Nobs_norm
    Nobs_norm_all.append(Nobs_norm)
    slope_all.append(slope)
#'''
#'''
# K_cond/sum(K) vs. K_cond plot:
fig = plt.figure(figsize=(5,10))
plot = GridSpec(3,1,left=0.15,bottom=0.05,right=0.8,top=0.95,wspace=0,hspace=0) # main plot
plt.suptitle(r'$P_{\rm cond} = %s$d, $R_{p,\rm cond} = %s R_\oplus$' % (P_cond_bounds, Rp_cond_bounds), fontsize=12)
for j,σ_1obs in enumerate(σ_1obs_all):
    ax = plt.subplot(plot[j,:])
    cmap = matplotlib.cm.viridis
    cmap.set_bad('r')
    sc = plt.scatter(outputs_all[j]['K_cond'], outputs_all[j]['K_cond']/outputs_all[j]['K_sum'], s=10., c=outputs_all[j]['N_obs_min_20p'], cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=N_obs_all[0], vmax=N_obs_all[-1]), plotnonfinite=True)
    plt.axvline(x=σ_1obs, ls=':', color='r', label=r'$\sigma_{1,\rm obs}$')
    plt.text(x=8., y=0.06, s=r'$\sigma_{1,\rm obs} = %s$ m/s' % σ_1obs, ha='right', color='r', fontsize=12)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    ax.set_xticks([0.1,0.2,0.3,0.5,1.,2.,3.,5.,10.]) # [0.5, 1., 2., 4., 8., 16.]
    ax.set_yticks([0.1,0.2,0.3,0.4,0.5,1.])
    plt.minorticks_off()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if j != len(σ_1obs_all)-1:
        plt.xticks([])
    plt.xlim([0.05, 10.])
    plt.ylim([0.05, 1.])
    ax.tick_params(axis='both', labelsize=10)
    if j == len(σ_1obs_all)-1:
        plt.xlabel(r'$K_{\rm cond}$ (m/s)', fontsize=12)
    plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=12)
    if j == len(σ_1obs_all)-1:
        plot = GridSpec(1,1,left=0.81,bottom=0.05,right=0.85,top=0.95) # side panel colorbar
        cax = plt.subplot(plot[:,:])
        cticks_custom = [5,10,20,50,100,200,300,500]
        cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter())
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=12)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_R%s_%s_%sNobs%sto%s_sigma_panels_Kcondfrac_vs_Kcond.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1])
    plt.savefig(fig_name)
    plt.close()

# N_obs_min vs K_cond plot:
fig = plt.figure(figsize=(5,10))
plot = GridSpec(3,5,left=0.15,bottom=0.05,right=0.96,top=0.95,wspace=0,hspace=0)
plt.suptitle(r'$P_{\rm cond} = %s$d, $R_{p,\rm cond} = %s R_\oplus$' % (P_cond_bounds, Rp_cond_bounds), fontsize=12)
for j,σ_1obs in enumerate(σ_1obs_all):
    Nobs_ideal_Karray = linear_logNobs_logK(K_array, σ_1obs, Nobs_norm_all[j], slope_all[j], round_to_ints=False) # ideal N_obs at K_array points
    Nobs_ideal_Kcond = linear_logNobs_logK(outputs_all[j]['K_cond'], σ_1obs, Nobs_norm_all[j], slope_all[j], round_to_ints=True) # ideal N_obs at K_cond points
    
    ax = plt.subplot(plot[j,:-1]) # main plot
    plt.loglog(outputs_all[j]['K_cond'], outputs_all[j]['N_obs_min_20p'], 'o', ms=3., color='k', label='Conditioned\nplanets')
    #plt.loglog(outputs_ideal_all[j]['K'], outputs_ideal_all[j]['N_obs_min_20p'], 'o', color='r')
    plt.plot(K_array[Nobs_ideal_Karray > 5], Nobs_ideal_Karray[Nobs_ideal_Karray > 5], '-', lw=3, color='r', label='Ideal')
    plt.axvline(x=σ_1obs, ls=':', color='r') # label=r'$\sigma_{1,\rm obs}$'
    plt.text(x=1.2*σ_1obs, y=300., s=r'$\sigma_{1,\rm obs} = %s$ m/s' % σ_1obs, color='r', fontsize=12)
    yticks_custom = [5,10,20,50,100,200,500]
    ax.set_xticks([0.1,0.2,0.5,1.,2.,5.,10.])
    ax.set_yticks(yticks_custom)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if j != len(σ_1obs_all)-1:
        plt.xticks([])
    plt.xlim([0.05,20.])
    plt.ylim([4,400])
    ax.tick_params(axis='both', labelsize=10)
    if j == len(σ_1obs_all)-1:
        plt.xlabel(r'$K_{\rm cond}$ (m/s)', fontsize=12)
        plt.legend(loc='lower left', bbox_to_anchor=(0,0.5), ncol=1, frameon=False, fontsize=10)
    plt.ylabel(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=12)
    
    ax = plt.subplot(plot[j,-1]) # side panel CDFs
    K_cond_bins = [0.,0.5,1.,2.,4.,8.,np.inf]
    N_obs_min_20p_per_K_cond_bin = [outputs_all[j]['N_obs_min_20p'][(K_cond_bins[i] <= outputs_all[j]['K_cond']) & (outputs_all[j]['K_cond'] <= K_cond_bins[i+1])] for i in range(len(K_cond_bins)-1)]
    N_obs_min_20p_per_K_cond_bin_single_planet = [Nobs_ideal_Kcond[(K_cond_bins[i] <= outputs_all[j]['K_cond']) & (outputs_all[j]['K_cond'] <= K_cond_bins[i+1])] for i in range(len(K_cond_bins)-1)]
    #N_obs_min_20p_per_K_cond_bin_single_planet = [outputs_ideal_all[j]['N_obs_min_20p'][(K_cond_bins[i] <= outputs_ideal_all[j]['K']) & (outputs_ideal_all[j]['K'] <= K_cond_bins[i+1])] for i in range(len(K_cond_bins)-1)]
    color_bins = ['k','b','g','r','orange','m']
    label_bins = [r'$K_{\rm cond} < %s$' % K_cond_bins[1]] + [r'$%s \leq K_{\rm cond} < %s$' % (K_cond_bins[i], K_cond_bins[i+1]) for i in range(1,len(K_cond_bins)-2)] + [r'$K_{\rm cond} > %s$ m/s' % K_cond_bins[-2]]
    for i,x in enumerate(N_obs_min_20p_per_K_cond_bin):
        plt.plot((np.arange(len(x))+1.)/np.float(len(x)), np.sort(x), drawstyle='steps-post', color=color_bins[i], ls='-', lw=1, label=label_bins[i])
    for i,x in enumerate(N_obs_min_20p_per_K_cond_bin_single_planet):
        plt.plot((np.arange(len(x))+1.)/np.float(len(x)), np.sort(x), drawstyle='steps-post', color=color_bins[i], ls='--', lw=1)
    plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelleft=False, labelsize=10)
    if j != len(σ_1obs_all)-1:
        plt.xticks([])
    plt.xlim([0,1])
    plt.ylim([4,400])
    if j == len(σ_1obs_all)-1:
        plt.xlabel('CDF', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if j == len(σ_1obs_all)-1:
        plt.legend(handles[::-1], labels[::-1], loc='lower right', bbox_to_anchor=(-1.9,0), ncol=1, frameon=False, fontsize=10)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_R%s_%s_%sNobs%sto%s_sigma_panels_Nobs_vs_Kcond.pdf' % (P_cond_bounds[0], P_cond_bounds[1], Rp_cond_bounds[0], Rp_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1])
    plt.savefig(fig_name)
    plt.close()
plt.show()
#'''





##### To simulate and fit RV observations of systems conditioned on a given planet, to see how the measured K varies with number of observations:

N_obs_all = np.array([int(round(x)) for x in np.logspace(np.log10(5), np.log10(300), 20)])
N_sample, repeat = 1000, 100
σ_1obs = 0.3
Rp_cond_bounds_all = [[0.9,1.1], [1.8,2.0], [3.0,4.0]]

'''
fname_all = ['Conditional_P8_12d_R0p9_1p1_transiting/RV_obs_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (N_sample, repeat), 'Conditional_P8_12d_R1p8_2_transiting/RV_obs_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (N_sample, repeat), 'Conditional_P8_12d_R3_4_transiting/RV_obs_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (N_sample, repeat)]
outputs_all = []
for fname in fname_all:
    outputs = np.genfromtxt(loadfiles_directory + fname, names=True)
    #outputs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('i4','f8','f8','f8','f8','f8'))
    #outputs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('i4','f8','f8','f8','f8','f8','f8','f8'))
    outputs_all.append(outputs)

# To also simulate and fit RV observations of single planet systems, to see how the measured K varies with number of observations:
K_array = np.logspace(np.log10(0.05), np.log10(10.), 100) # m/s
fname = 'RV_obs_P8_12d_singles_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (len(K_array), repeat)
outputs_ideal = np.genfromtxt(loadfiles_directory + 'RV_obs_singles/' + fname, names=True, dtype=('f8','f8','f8'))

log_Nobs_norm, slope = fit_line_loglog_Nobs_K_single_planets(outputs_ideal, σ_1obs, [2., -2.]) # fit a line to the ideal case simulations
Nobs_norm = 10.**log_Nobs_norm
'''
'''
fig = plt.figure(figsize=(10,10))
plt.suptitle(r'$P_{\rm cond} = %s$d' % P_cond_bounds, fontsize=12)

# K_cond/sum(K) vs. K_cond plot:
plot = GridSpec(3,1,left=0.09,bottom=0.05,right=0.41,top=0.95,wspace=0,hspace=0) #left=0.08,bottom=0.05,right=0.4,top=0.95
for j,outputs in enumerate(outputs_all):
    ax = plt.subplot(plot[j,:])
    plt.text(x=0.02, y=0.98, s=r'$R_{p,\rm cond} =$' + '\n' + r'$%s R_\oplus$' % Rp_cond_bounds_all[j], ha='left', va='top', fontsize=12, transform=ax.transAxes)
    cmap = matplotlib.cm.viridis
    cmap.set_bad('r')
    sc = plt.scatter(outputs['K_cond'], outputs['K_cond']/outputs['K_sum'], s=10., c=outputs['N_obs_min_20p'], cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=N_obs_all[0], vmax=N_obs_all[-1]), plotnonfinite=True)
    plt.axvline(x=σ_1obs, ls=':', color='r', label=r'$\sigma_{1,\rm obs}$')
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    ax.set_xticks([0.1,0.2,0.3,0.5,1.,2.,3.,5.,10.]) # [0.5, 1., 2., 4., 8., 16.]
    ax.set_yticks([0.1,0.2,0.3,0.4,0.5,1.])
    plt.minorticks_off()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if j != len(σ_1obs_all)-1:
        plt.xticks([])
    plt.xlim([0.05, 10.])
    plt.ylim([0.05, 1.])
    ax.tick_params(axis='both', which='both', labelsize=10)
    if j == len(outputs_all)-1:
        plt.xlabel(r'$K_{\rm cond}$ (m/s)', fontsize=12)
    plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=12)
    if j == len(outputs_all)-1:
        plot = GridSpec(1,1,left=0.43,bottom=0.05,right=0.45,top=0.95) # side panel colorbar
        cax = plt.subplot(plot[:,:])
        cticks_custom = [5,10,20,50,100,200,300,500]
        cbar = plt.colorbar(sc, cax=cax, ticks=cticks_custom, orientation='vertical', format=ticker.ScalarFormatter())
        cbar.ax.tick_params(labelsize=10)
        #cbar.set_label(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=12)

# N_obs_min vs K_cond plot:
plot = GridSpec(3,5,left=0.56,bottom=0.05,right=0.96,top=0.95,wspace=0,hspace=0) #left=0.58,bottom=0.05,right=0.98,top=0.95
for j,outputs in enumerate(outputs_all):
    Nobs_ideal_Karray = linear_logNobs_logK(K_array, σ_1obs, Nobs_norm, slope, round_to_ints=False) # ideal N_obs at K_array points
    Nobs_ideal_Kcond = linear_logNobs_logK(outputs['K_cond'], σ_1obs, Nobs_norm, slope, round_to_ints=True) # ideal N_obs at K_cond points
    
    ax = plt.subplot(plot[j,:-1]) # main plot
    plt.loglog(outputs['K_cond'], outputs['N_obs_min_20p'], 'o', ms=3., color='k', label='Conditioned\nplanets')
    #plt.loglog(outputs_ideal_all[j]['K'], outputs_ideal_all[j]['N_obs_min_20p'], 'o', color='r')
    plt.plot(K_array[Nobs_ideal_Karray > 5], Nobs_ideal_Karray[Nobs_ideal_Karray > 5], '-', lw=3, color='r', label='Ideal')
    plt.axvline(x=σ_1obs, ls=':', color='r', label=r'$\sigma_{1,\rm obs}$')
    yticks_custom = [5,10,20,50,100,200,500]
    ax.set_xticks([0.1,0.2,0.5,1.,2.,5.,10.])
    ax.set_yticks(yticks_custom)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if j != len(σ_1obs_all)-1:
        plt.xticks([])
    plt.xlim([0.05,20.])
    plt.ylim([4,400])
    ax.tick_params(axis='both', which='both', labelsize=10)
    if j == len(σ_1obs_all)-1:
        plt.xlabel(r'$K_{\rm cond}$ (m/s)', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=10)
    plt.ylabel(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=12)
    
    ax = plt.subplot(plot[j,-1]) # side panel CDFs
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
    ax.tick_params(axis='both', labelleft=False, labelsize=10)
    if j != len(σ_1obs_all)-1:
        plt.xticks([])
    plt.xlim([0,1])
    plt.ylim([4,400])
    if j == len(σ_1obs_all)-1:
        plt.xlabel('CDF', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if j == 0:
        plt.legend(handles[::-1], labels[::-1], loc='upper right', bbox_to_anchor=(0,1), ncol=1, frameon=False, fontsize=10)

if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_%sNobs%sto%s_radii_conds_panels_Kcondfrac_Nobs_vs_Kcond.pdf' % (P_cond_bounds[0], P_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1])
    plt.savefig(fig_name)
    plt.close()
plt.show()
'''





##### To plot CDFs of N_obs for different radius bins (no conditioning on radius):
'''
σ_1obs_linestyles_all = [':','--','-']

fname_all = ['RV_obs_P8_12d_N%s_repeat%s_20Nobs5to300_sigma0p1.txt' % (N_sample, repeat), 'RV_obs_P8_12d_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (N_sample, repeat), 'RV_obs_P8_12d_N%s_repeat%s_20Nobs5to300_sigma1.txt' % (N_sample, repeat)]
outputs_all = []
for fname in fname_all:
    outputs = np.genfromtxt(loadfiles_directory + fname, names=True)
    outputs_all.append(outputs)

# CDFs of N_obs_min for different radius bins:
fig = plt.figure(figsize=(8,10))
plot = GridSpec(4,1,left=0.15,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
plt.suptitle(r'$P_{\rm cond} = %s$d' % P_cond_bounds, fontsize=20)
Rp_cond_bins = [0.5,1.5,3.,4.5,np.inf][::-1] # reverse so larger radii bins are on top
color_bins = ['k','b','g','r'][::-1]
label_bins = [r'$R_{p,\rm cond} > %s R_\oplus$' % Rp_cond_bins[1]] + [r'$%s R_\oplus \leq R_{p,\rm cond} < %s R_\oplus$' % (Rp_cond_bins[i+1], Rp_cond_bins[i]) for i in range(1,len(Rp_cond_bins)-2)] + [r'$R_{p,\rm cond} < %s R_\oplus$' % Rp_cond_bins[-2]]
xticks_custom = [5,10,20,50,100,200,500]
for i in range(len(Rp_cond_bins)-1):
    ax = plt.subplot(plot[i,:])
    plt.text(x=4.5, y=1., s=label_bins[i], fontsize=16)
    for j,σ_1obs in enumerate(σ_1obs_all):
        x = outputs_all[j]['N_obs_min_20p'][(Rp_cond_bins[i+1] <= outputs_all[j]['Rp_cond']) & (outputs_all[j]['Rp_cond'] <= Rp_cond_bins[i])]
        plt.plot(np.sort(x), (np.arange(len(x))+1.)/np.float(len(x)), drawstyle='steps-post', color=color_bins[i], ls=σ_1obs_linestyles_all[j], lw=1, label=r'$\sigma_{1,\rm obs} = %s$ m/s' % σ_1obs)
    plt.gca().set_xscale("log")
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks(xticks_custom)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if i != len(label_bins)-1:
        plt.xticks([])
    plt.xlim([4,300])
    plt.ylim([0,1.15])
    if i == len(label_bins)-1:
        plt.xlabel(r'$N_{\rm obs}$ for $RMSD(K_{\rm cond})/K_{\rm cond} < 0.2$', fontsize=20)
    plt.ylabel('CDF', fontsize=20)
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=16)
if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_P%s_%s_%sNobs%sto%s_Nobs_CDFs_Rpcond_panels.pdf' % (P_cond_bounds[0], P_cond_bounds[1], len(N_obs_all), N_obs_all[0], N_obs_all[-1])
    plt.savefig(fig_name)
    plt.close()
plt.show()
'''





##### To plot CDFs of N_obs for Venus-like planets (transiting and not):
'''
outputs_Venus_all = np.genfromtxt(loadfiles_directory + 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p1.txt' % (N_sample, repeat), names=True)
outputs_Venus_all_fit_all = np.genfromtxt(loadfiles_directory + 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p1_fit_all.txt' % (N_sample, repeat), names=True)
outputs_Venus_cond_only_all = np.genfromtxt(loadfiles_directory + 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p1_cond_only.txt' % (N_sample, repeat), names=True)
outputs_Venus_transiting = np.genfromtxt(loadfiles_directory + '../Conditional_Venus_transiting/' + 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p1.txt' % (N_sample, repeat), names=True)
outputs_Venus_transiting_fit_all = np.genfromtxt(loadfiles_directory + '../Conditional_Venus_transiting/' + 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p1_fit_all.txt' % (N_sample, repeat), names=True)
outputs_Venus_cond_only_transiting = np.genfromtxt(loadfiles_directory + '../Conditional_Venus_transiting/' + 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p1_cond_only.txt' % (N_sample, repeat), names=True)

def compute_quantiles_with_nans(x, qtls=[0.16,0.5,0.84]):
    xlen = len(x)
    x = np.sort(x)
    if xlen < 100:
        print('Warning: computing quantiles from fewer than 100 samples; results may not be accurate (len(x) = %s).' % xlen)
    x_qtls = np.zeros(len(qtls))
    for i,q in enumerate(qtls):
        x_qtls[i] = x[int(np.round(q*xlen))]
    return x_qtls

def report_quantiles_as_uncertainties_string(x_qtls, nan_above_val=300):
    assert len(x_qtls) == 3
    # This function assumes x_qtls[0], x_qtls[1], x_qtls[2] are 16th, 50th, and 84th percentiles
    if np.isnan(x_qtls[1]): # Case 1: median > nan_above_val
        if np.isnan(x_qtls[0]): # Case 1a: all quantiles > nan_above_val
            snum = r'$>%s$' % nan_above_val
        else: # Case 1b: lower quantile <= nan_above_val
            snum = r'$>{:d}_{{-{:d}}}$'.format(nan_above_val, int(nan_above_val-x_qtls[0]))
    else: # Case 2: median <= nan_above_val
        if np.isnan(x_qtls[2]): # Case 2a: upper quantile > nan_above_val
            snum = r'${:d}_{{-{:d}}}$'.format(int(x_qtls[1]), int(x_qtls[1]-x_qtls[0]))
        else: # Case 2b: all quantiles <= nan_above_val
            snum = r'${:d}_{{-{:d}}}^{{+{:d}}}$'.format(int(x_qtls[1]), int(x_qtls[1]-x_qtls[0]), int(x_qtls[2]-x_qtls[1]))
    return snum

fig = plt.figure(figsize=(8,8))
plot = GridSpec(2,1,left=0.15,bottom=0.125,right=0.95,top=0.925,wspace=0,hspace=0)
plt.suptitle(r'Venus-like planets', fontsize=20)
xticks_custom = [5,10,20,50,100,200,500,1000]
fields_thres = ['N_obs_min_20p', 'N_obs_min_10p']
labels_thres = [r'$RMSD(K_{\rm cond})/K_{\rm cond} < 20$%', r'$RMSD(K_{\rm cond})/K_{\rm cond} < 10$%']
colors_thres = ['r','b']
ytext_thres = [0.75, 0.25]

ax = plt.subplot(plot[0,:]) # for transiting Venuses only
plt.text(x=6., y=1.05, s='Transiting', fontsize=16)
plot_snums = []
plot_lines = []
for i,field in enumerate(fields_thres):
    x = outputs_Venus_cond_only_transiting[field] # Venus is the only planet (ideal case)
    x_qtls = compute_quantiles_with_nans(x)
    snum0 = report_quantiles_as_uncertainties_string(x_qtls, nan_above_val=int(np.nanmax(x)))
    l0, = plt.plot(np.sort(x), (np.arange(len(x))+1.)/np.float(len(x)), drawstyle='steps-post', color=colors_thres[i], ls=':', lw=1, label=labels_thres[i])
    
    x = outputs_Venus_transiting_fit_all[field] # simulated systems with a Venus, fit all planets
    x_qtls = compute_quantiles_with_nans(x)
    snum1 = report_quantiles_as_uncertainties_string(x_qtls, nan_above_val=int(np.nanmax(x)))
    l1, = plt.plot(np.sort(x), (np.arange(len(x))+1.)/np.float(len(x)), drawstyle='steps-post', color=colors_thres[i], ls='--', lw=1)
    
    x = outputs_Venus_transiting[field] # simulated systems with a Venus, fit only Venus
    x_qtls = compute_quantiles_with_nans(x)
    snum2 = report_quantiles_as_uncertainties_string(x_qtls, nan_above_val=int(np.nanmax(x)))
    l2, = plt.plot(np.sort(x), (np.arange(len(x))+1.)/np.float(len(x)), drawstyle='steps-post', color=colors_thres[i], ls='-', lw=1)
    
    plot_snums.append([snum0, snum1, snum2])
    plot_lines.append([l0, l1, l2])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=20)
ax.set_xticks(xticks_custom)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlim([5,1000])
plt.ylim([0,1.15])
plt.xticks([])
plt.ylabel('CDF', fontsize=20)
legend1 = plt.legend(plot_lines[-1], ['Ideal case', 'Fit all planets', 'Fit Venus only'], loc='upper left', bbox_to_anchor=(0,0.9), ncol=1, frameon=False, fontsize=16) # for different scenarios (line styles)
plt.legend([l for l_set in plot_lines for l in l_set], [snum for snum_set in plot_snums for snum in snum_set], loc='lower left', bbox_to_anchor=(0,0), ncol=2, frameon=False, fontsize=12) # for reporting median and quantiles (all colors and line styles)
plt.gca().add_artist(legend1)

ax = plt.subplot(plot[1,:]) # for all Venuses
plt.text(x=6., y=1.05, s='All', fontsize=16)
plot_snums = []
plot_lines = []
for i,field in enumerate(fields_thres):
    x = outputs_Venus_cond_only_all[field] # Venus is the only planet (ideal case)
    x_qtls = compute_quantiles_with_nans(x)
    snum0 = report_quantiles_as_uncertainties_string(x_qtls, nan_above_val=int(np.nanmax(x)))
    l0, = plt.plot(np.sort(x), (np.arange(len(x))+1.)/np.float(len(x)), drawstyle='steps-post', color=colors_thres[i], ls=':', lw=1, label=labels_thres[i])
    
    x = outputs_Venus_all_fit_all[field] # simulated systems with a Venus, fit all planets
    x_qtls = compute_quantiles_with_nans(x)
    snum1 = report_quantiles_as_uncertainties_string(x_qtls, nan_above_val=int(np.nanmax(x)))
    l1, = plt.plot(np.sort(x), (np.arange(len(x))+1.)/np.float(len(x)), drawstyle='steps-post', color=colors_thres[i], ls='--', lw=1)
    
    x = outputs_Venus_all[field] # simulated systems with a Venus, fit only Venus
    x_qtls = compute_quantiles_with_nans(x)
    snum2 = report_quantiles_as_uncertainties_string(x_qtls, nan_above_val=int(np.nanmax(x)))
    l2, = plt.plot(np.sort(x), (np.arange(len(x))+1.)/np.float(len(x)), drawstyle='steps-post', color=colors_thres[i], ls='-', lw=1)
    
    plot_snums.append([snum0, snum1, snum2])
    plot_lines.append([l0, l1, l2])
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=20)
ax.set_xticks(xticks_custom)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlim([5,1000])
plt.ylim([0,1.15])
plt.xlabel(r'$N_{\rm obs}$', fontsize=20)
plt.ylabel('CDF', fontsize=20)
legend1 = plt.legend([l[-1] for l in plot_lines], labels_thres, loc='upper left', bbox_to_anchor=(0,0.9), ncol=1, frameon=False, fontsize=16) # for different thresholds (colors)
plt.legend([l for l_set in plot_lines for l in l_set], [snum for snum_set in plot_snums for snum in snum_set], loc='lower left', bbox_to_anchor=(0,0), ncol=2, frameon=False, fontsize=12) # for reporting median and quantiles (all colors and line styles)
plt.gca().add_artist(legend1)

if savefigures:
    fig_name = savefigures_directory + model_name + '_RVs_%sNobs%sto%s_Nobs_CDFs_Venus_panels.pdf' % (len(N_obs_all), N_obs_all[0], N_obs_all[-1])
    plt.savefig(fig_name)
    plt.close()
plt.show()
'''
