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

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *
from src.functions_plot_catalogs import *
from src.functions_plot_params import *





##### To load the underlying populations:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
#sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

cat_obs = load_cat_obs(loadfiles_directory + 'observed_catalog%s.csv' % run_number)
cat_phys = load_cat_phys(loadfiles_directory + 'physical_catalog%s.csv' % run_number)





##### To save the subset of the physical catalog with only the detected planets:

targetid_all = cat_phys['target_id']
P_all = cat_phys['period']
ferr_match = 0.05 # fractional error in period to match
print('Matching periods to within %s (fractional error)...' % ferr_match)

id_obs = []
for i,tid in enumerate(cat_obs['target_id']):
    P_obs_i = cat_obs['period'][i]
    fp_diff = np.abs(P_all - P_obs_i) / P_obs_i
    id_pl = np.where((targetid_all == tid) & (fp_diff < ferr_match))[0][0]
    id_obs.append(id_pl)
id_obs = np.array(id_obs)

#np.savetxt(loadfiles_directory + 'physical_catalog%s_obs_only.csv' % run_number, cat_phys[id_obs], delimiter=',', header=','.join(cat_phys.dtype.names), comments='', fmt='%s')


