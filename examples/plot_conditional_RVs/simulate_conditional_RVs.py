# -*- coding: utf-8 -*-
# To import required modules:
import numpy as np
import time
import os
import sys

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *
from syssimpyplots.compute_RVs import *





##### To load the underlying and observed populations:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/Conditional_Venus_transiting/'
run_number = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

P_cond_bounds, Rp_cond_bounds, Mp_cond_bounds = [215.,235.], [0.9,1.0], [0.77,0.86]
conds = conditionals_dict(P_cond_bounds=P_cond_bounds, Rp_cond_bounds=Rp_cond_bounds, Mp_cond_bounds=Mp_cond_bounds, det=False)





##### To simulate and fit RV observations of systems conditioned on a given planet, to see how the measured K varies with number of observations:

N_obs_all = np.array([int(round(x)) for x in np.logspace(np.log10(5), np.log10(1000), 20)])
sigma_1obs = 0.1
N_sample, repeat = 10, 100
fname = 'RV_obs_N%s_repeat%s_20Nobs5to1000_sigma0p1_cases.txt' % (N_sample, repeat)
#fname = 'RV_obs_P8_12d_N%s_repeat%s_20Nobs5to300_sigma0p3.txt' % (N_sample, repeat)

outputs = fit_RVobs_systems_conditional(sssp_per_sys, sssp, conds, N_obs_all, cond_only=True, fit_sys_cond=True, fit_all_planets=True, N_sample=N_sample, repeat=repeat, σ_1obs=sigma_1obs)
#np.savetxt(loadfiles_directory + fname, outputs, fmt='%i %i' + ' %f'*(len(outputs.dtype.names)-2), header=' '.join(outputs.dtype.names), footer='N_obs = '+', '.join([str(x) for x in N_obs_all]))
#outputs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('i4','i4')+('f8',)*(len(outputs.dtype.names)-2))



##### To also simulate and fit RV observations of single planet systems, to see how the measured K varies with number of observations:
#K_array = np.logspace(np.log10(0.05), np.log10(5.), 100) # m/s
#alpha_P, sigma_ecc = 0., 0.25
#fname = 'RV_obs_P8_12d_singles_N%s_repeat%s_20Nobs5to300_sigma0p1.txt' % (len(K_array), repeat)

#outputs_single_planet_RVs = fit_RVobs_single_planets_vs_K(K_array, N_obs_all, P_cond_bounds, alpha_P=alpha_P, sigma_ecc=sigma_ecc, repeat=repeat, sigma_1obs=sigma_1obs)
#np.savetxt(loadfiles_directory + fname, outputs_single_planet_RVs, fmt='%f %f %f', header=' '.join(outputs_single_planet_RVs.dtype.names), footer='N_obs = '+', '.join([str(x) for  x in N_obs_all]))
#outputs_single_planet_RVs = np.genfromtxt(loadfiles_directory + fname, names=True, dtype=('f8','f8','f8'))
