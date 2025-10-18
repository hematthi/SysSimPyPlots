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





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

model_names = ['H20 model', 'HM-U', 'HM-C']

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8p1_KS/Params9_fix_highM/GP_best_models_100/'
loadfiles_directory3 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/'

model_loadfiles_dirs = [loadfiles_directory1, loadfiles_directory2, loadfiles_directory3]
models = len(model_loadfiles_dirs)

runs = 100

sssp_all = []
sssp_per_sys_all = []
params_all = []

Mp_Venus = 0.815 # mass of Venus (Earth masses)
P_Venus = 224.7 # orbital period of Venus (days)
planet_type_keys = ['all',
                    'R0.75-1.4_P3-100',
                    'R1.4-2.8_P3-100',
                    'R2-4_P3-100',
                    'M>50_P3-11',
                    'M3-10_P3-50',
                    'M1-3_P3-50',
                    'R0.75-2.5_P50-300',
                    'Venus_RP20p',
                    'Venus_MRP20p',
                    'M<4_R10p_silicate_P180-300',
                    ]

occurrence_dicts_all = []
fswp_dicts_all = []

for loadfiles_dir in model_loadfiles_dirs:
    sssp_dir = []
    sssp_per_sys_dir = []
    params_dir = []
    
    # Dictionaries to hold all the various occurrence rates and fractions of stars with planets for the current model, from each catalog:
    occurrence_dict = {key: [] for key in planet_type_keys}
    fswp_dict = {key: [] for key in planet_type_keys}
    
    for i in range(runs):
        run_number = i+1
        print(i)
        N_sim_i = read_targets_period_radius_bounds(loadfiles_dir + 'periods%s.out' % run_number)[0]
        params_i = read_sim_params(loadfiles_dir + 'periods%s.out' % run_number)
        sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_dir, run_number=run_number)
        
        # Catalogs and parameters:
        sssp_dir.append(sssp_i)
        sssp_per_sys_dir.append(sssp_per_sys_i)
        params_dir.append(params_i)
        
        ### Now calculate various planet occurrence rates and fractions of stars with planets:
        
        # Any planet (full simulation bounds):
        bools_per_sys = (sssp_per_sys_i['P_all'] > 0.)
        counts_pl = np.sum(bools_per_sys) # total number of planets
        counts_swp = np.sum(np.any(bools_per_sys, axis=1)) # total number of systems with at least one planet
        occurrence_dict['all'].append(counts_pl/N_sim_i)
        fswp_dict['all'].append(counts_swp/N_sim_i)
        
        # Rp=0.75-1.4, P=3-100:
        bools_per_sys = (sssp_per_sys_i['radii_all'] > 0.75) & (sssp_per_sys_i['radii_all'] < 1.4) & (sssp_per_sys_i['P_all'] > 3.) & (sssp_per_sys_i['P_all'] < 100.)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['R0.75-1.4_P3-100'].append(counts_pl/N_sim_i)
        fswp_dict['R0.75-1.4_P3-100'].append(counts_swp/N_sim_i)
        
        # Rp=1.4-2.8, P=3-100:
        bools_per_sys = (sssp_per_sys_i['radii_all'] > 1.4) & (sssp_per_sys_i['radii_all'] < 2.8) & (sssp_per_sys_i['P_all'] > 3.) & (sssp_per_sys_i['P_all'] < 100.)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['R1.4-2.8_P3-100'].append(counts_pl/N_sim_i)
        fswp_dict['R1.4-2.8_P3-100'].append(counts_swp/N_sim_i)

        # Rp=2.0-4.0, P=3-100:
        bools_per_sys = (sssp_per_sys_i['radii_all'] > 2.) & (sssp_per_sys_i['radii_all'] < 4.) & (sssp_per_sys_i['P_all'] > 3.) & (sssp_per_sys_i['P_all'] < 100.)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['R2-4_P3-100'].append(counts_pl/N_sim_i)
        fswp_dict['R2-4_P3-100'].append(counts_swp/N_sim_i)
        
        # Mp>50, P<11 (P=3-11):
        bools_per_sys = (sssp_per_sys_i['mass_all'] > 50.) & (sssp_per_sys_i['P_all'] > 3.) & (sssp_per_sys_i['P_all'] < 11.)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['M>50_P3-11'].append(counts_pl/N_sim_i)
        fswp_dict['M>50_P3-11'].append(counts_swp/N_sim_i)

        # Mp=3-10, P<50 (P=3-50):
        bools_per_sys = (sssp_per_sys_i['mass_all'] > 3.) & (sssp_per_sys_i['mass_all'] < 10.) & (sssp_per_sys_i['P_all'] > 3.) & (sssp_per_sys_i['P_all'] < 50.)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['M3-10_P3-50'].append(counts_pl/N_sim_i)
        fswp_dict['M3-10_P3-50'].append(counts_swp/N_sim_i)
        
        # Mp=1-3, P<50 (P=3-50):
        bools_per_sys = (sssp_per_sys_i['mass_all'] > 1.) & (sssp_per_sys_i['mass_all'] < 3.) & (sssp_per_sys_i['P_all'] > 3.) & (sssp_per_sys_i['P_all'] < 50.)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['M1-3_P3-50'].append(counts_pl/N_sim_i)
        fswp_dict['M1-3_P3-50'].append(counts_swp/N_sim_i)

        # Rp=0.75-2.5, P=50-300:
        bools_per_sys = (sssp_per_sys_i['radii_all'] > 0.75) & (sssp_per_sys_i['radii_all'] < 2.5) & (sssp_per_sys_i['P_all'] > 50.) & (sssp_per_sys_i['P_all'] < 300.)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['R0.75-2.5_P50-300'].append(counts_pl/N_sim_i)
        fswp_dict['R0.75-2.5_P50-300'].append(counts_swp/N_sim_i)

        # Within 20% of Venus by Burke+2015 definition:
        # (Rp=0.8-1.2, P=(0.8-1.2)P_Venus~180-270)
        bools_per_sys = (sssp_per_sys_i['radii_all'] > 0.8) & (sssp_per_sys_i['radii_all'] < 1.2) & (sssp_per_sys_i['P_all'] > 0.8*P_Venus) & (sssp_per_sys_i['P_all'] < 1.2*P_Venus)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['Venus_RP20p'].append(counts_pl/N_sim_i)
        fswp_dict['Venus_RP20p'].append(counts_swp/N_sim_i)

        # Same as above but also requiring within 20% of Venus' mass:
        # (Rp=0.8-1.2, Mp~0.65-0.98, P=(0.8-1.2)P_Venus~180-270)
        bools_per_sys = (sssp_per_sys_i['radii_all'] > 0.8) & (sssp_per_sys_i['radii_all'] < 1.2) & (sssp_per_sys_i['mass_all'] > 0.8*Mp_Venus) & (sssp_per_sys_i['mass_all'] < 1.2*Mp_Venus) & (sssp_per_sys_i['P_all'] > 0.8*P_Venus) & (sssp_per_sys_i['P_all'] < 1.2*P_Venus)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['Venus_MRP20p'].append(counts_pl/N_sim_i)
        fswp_dict['Venus_MRP20p'].append(counts_swp/N_sim_i)
        
        # Mp<4, Rp within 10% of pure silicate (Seager 2007) relation, P=180-300:
        R_S07_all = radius_given_mass_pure_silicate_fit_seager2007(sssp_per_sys_i['mass_all']) # 2d array of planet radii from S07 relation, for each planet mass
        bools_per_sys = (sssp_per_sys_i['radii_all'] > 0.9*R_S07_all) & (sssp_per_sys_i['radii_all'] < 1.1*R_S07_all) & (sssp_per_sys_i['mass_all'] < 4.) & (sssp_per_sys_i['P_all'] > 180.) & (sssp_per_sys_i['P_all'] < 300.)
        counts_pl = np.sum(bools_per_sys)
        counts_swp = np.sum(np.any(bools_per_sys, axis=1))
        occurrence_dict['M<4_R10p_silicate_P180-300'].append(counts_pl/N_sim_i)
        fswp_dict['M<4_R10p_silicate_P180-300'].append(counts_swp/N_sim_i)
    
    sssp_all.append(sssp_dir)
    sssp_per_sys_all.append(sssp_per_sys_dir)
    params_all.append(params_dir)
    
    for key in occurrence_dict.keys():
        occurrence_dict[key] = np.array(occurrence_dict[key])
    for key in fswp_dict.keys():
        fswp_dict[key] = np.array(fswp_dict[key])
    occurrence_dicts_all.append(occurrence_dict)
    fswp_dicts_all.append(fswp_dict)





##### To compute the credible intervals for each occurrence rate statistic:


print('{:<30}'.format('Occurrence rates') + (" "*18).join(model_names))
for key in planet_type_keys:
    occurrence_str_all = []
    for m,model in enumerate(model_names):
        q = np.quantile(occurrence_dicts_all[m][key], [0.16,0.5,0.84])
        q_pm = np.diff(q)
        occ_str = '%s_{-%s}^{+%s}' % ('{:0.3f}'.format(q[1]), '{:0.3f}'.format(q_pm[0]), '{:0.3f}'.format(q_pm[1]))
        occurrence_str_all.append(occ_str)
    
    print('{:<30}'.format(key) + "   ".join(occurrence_str_all))

print('{:<30}'.format('FSWP') + (" "*18).join(model_names))
for key in planet_type_keys:
    fswp_str_all = []
    for m,model in enumerate(model_names):
        q = np.quantile(fswp_dicts_all[m][key], [0.16,0.5,0.84])
        q_pm = np.diff(q)
        fswp_str = '%s_{-%s}^{+%s}' % ('{:0.3f}'.format(q[1]), '{:0.3f}'.format(q_pm[0]), '{:0.3f}'.format(q_pm[1]))
        fswp_str_all.append(fswp_str)
    
    print('{:<30}'.format(key) + "   ".join(fswp_str_all))
