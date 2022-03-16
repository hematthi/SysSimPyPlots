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





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/OneRayleigh/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Misc_Presentations/PhD_Thesis_Defense/Figures/'
run_number = ''
model_name = 'OneRayleigh' + run_number





##### To load the files with the systems with observed planets:

# To load and process the observed Kepler catalog:
stars_cleaned = load_Kepler_stars_cleaned()
Rstar_med = np.nanmedian(stars_cleaned['radius'])
Mstar_med = np.nanmedian(stars_cleaned['mass'])
teff_med = np.nanmedian(stars_cleaned['teff'])
#bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
bp_rp_corr_med = np.nanmedian(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'])

P_min, P_max, radii_min, radii_max = 3., 300., 0.5, 10.

ssk_per_sys0, ssk0 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
ssk_per_sys1, ssk1 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_max=bp_rp_corr_med) #_max=_med
ssk_per_sys2, ssk2 = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=bp_rp_corr_med) #_min=_med





##### To plot the simulated and Kepler catalogs:

subdirectory = ''

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 2 #linewidth
alpha = 0.2 #transparency of histograms

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





Mtot_obs, Nmult_obs = count_planets_from_loading_cat_obs_stars_only(file_name_path=loadfiles_directory, run_number=run_number)
print(np.sum(Mtot_obs), Nmult_obs)
Nmult_obs = bin_Nmult(Nmult_obs)



samples = 2
sample_names = ['all'] + [str(i) for i in range(samples)]
sample_labels = ['All', 'Bluer', 'Redder'] # for samples=2 only
sample_colors = ['k', 'b', 'r'] # for samples=2 only
sample_bprp_bounds = np.quantile(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'], np.linspace(0,1,samples+1))

Nmult_bins = ['1','2','3','4','5+']

Nmult_Kep = {key: [] for key in sample_names}
Nmult_runs = {key: [] for key in sample_names}

N_pl_Kep = {}
N_pl_runs = {key: [] for key in sample_names} # total number of planets

# Kepler counts first:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max) # combined sample
N_pl_Kep['all'] = np.sum(ssk['Nmult_obs'] * np.arange(1,len(ssk['Nmult_obs'])+1))
Nmult_Kep['all'] = bin_Nmult(ssk['Nmult_obs'])
print('Kepler (all): ', N_pl_Kep['all'], ssk['Nmult_obs'])
for j in range(samples):
    ssk_per_sys_j, ssk_j = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=sample_bprp_bounds[j], bp_rp_max=sample_bprp_bounds[j+1])
    print('Kepler (sample %s): ' % j, ssk_j['Nmult_obs'])
    Nmult_Kep[str(j)] = bin_Nmult(ssk_j['Nmult_obs'])
    N_pl_Kep[str(j)] = np.sum(ssk_j['Nmult_obs'] * np.arange(1,len(ssk_j['Nmult_obs'])+1))





def pad_zero_beg_and_end(x):
    return np.array([0] + list(x) + [0])

##### To plot observed multiplicity distributions for each sample for the Kepler data:
fig = plt.figure(figsize=(6,8))
plot = GridSpec(samples+1,1,left=0.2,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
for i,sample in enumerate(sample_names):
    ax = plt.subplot(plot[i,0])
    
    plt.plot(range(0,6+1), pad_zero_beg_and_end(Nmult_Kep[sample]), drawstyle='steps-mid', ls='-', color=sample_colors[i])
    plt.scatter(range(1,6), Nmult_Kep[sample], marker='x', color=sample_colors[i])
    plt.text(0.05, 0.1, r'${:0.0f}$ planets'.format(N_pl_Kep[sample]), ha='left', va='bottom', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    plt.text(0.98, 0.95, sample_labels[i], ha='right', va='top', color=sample_colors[i], fontsize=lfs, transform=ax.transAxes)
    
    plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=16)
    #plt.yticks([0.5,1,1.5])
    plt.xlim([0.5,5.5])
    plt.ylim([3,1500])
    a = np.array(ax.get_xticks(), dtype='int').tolist()
    a[5] = '5+'
    ax.set_xticklabels(a)
    if i==1:
        plt.ylabel(r'$N_{\rm Kep}$', fontsize=tfs)
    if i==2:
        plt.xlabel(r'Observed multiplicity $m$', fontsize=tfs)

if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_split_Kepler.pdf')
    plt.close()
plt.show()





##### To plot the multiplicity distribution illustrating the Kepler dichotomy for defense talk:

fig = plt.figure(figsize=(6,4))
plot = GridSpec(1,1,left=0.2,bottom=0.2,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[:,:])
plt.plot(range(0,6+1), pad_zero_beg_and_end(Nmult_Kep['all']), drawstyle='steps-mid', marker='x', ls='-', linewidth=1, color='k', label='Kepler data')
#plt.bar(range(0,6+1), pad_zero_beg_and_end(0.7*Nmult_obs/5), 1., color='b', label='Single population model')
plt.plot(range(0,6+1), pad_zero_beg_and_end(0.7*Nmult_obs/5), drawstyle='steps-mid', ls='--', linewidth=2, color='r', label='Single population model')
#plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=16)
#plt.yticks([0.5,1,1.5])
plt.xlim([0.5,5.5])
plt.ylim([0,1500]) #[3,1500]
a = np.array(ax.get_xticks(), dtype='int').tolist()
a[5] = '5+'
ax.set_xticklabels(a)
plt.ylabel(r'Number of systems', fontsize=tfs)
plt.xlabel(r'Observed multiplicity, $m$', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + 'Nmult_KeplerDichotomy.pdf')
    plt.close()
plt.show()

