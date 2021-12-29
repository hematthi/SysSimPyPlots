# To import required modules:
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.functions_general import *
from src.functions_compare_kepler import *





# Test functions in "src/functions_compare_kepler.py":

def test_load_Kepler_planets_cleaned():
    pc = load_Kepler_planets_cleaned()
    assert 3. <= np.min(pc['P']) <= np.max(pc['P']) <= 300. # check range of periods (days)
    assert 0 <= np.min(pc['t_D']) <= np.max(pc['t_D']) <= 100. # check range of transit durations (hrs)
    assert 0 <= np.min(pc['depth'])/1e6 <= np.max(pc['depth'])/1e6 <= 1. # check range of transit depths
    assert 0.5 <= np.min(pc['Rp']) <= np.max(pc['Rp']) <= 10. # check range of planet radii (Earth radii)
    assert 0 < np.min(pc['Rstar'])
    assert 0 < np.min(pc['Mstar'])

def test_load_Kepler_stars_cleaned():
    sc = load_Kepler_stars_cleaned()
    assert 0 < np.min(sc['mass']) <= np.max(sc['mass']) < 2. # check range of stellar mass (Solar masses)
    assert 0 < np.min(sc['radius']) <= np.max(sc['radius']) < 4. # check range of stellar radii (Solar radii)
    assert 3000. < np.min(sc['teff']) <= np.max(sc['teff']) < 10000. # check range of T_eff (K)

def test_compute_summary_stats_from_Kepler_catalog():
    ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(3., 300., 0.5, 10.)
    N_sys = len(ssk_per_sys['Mtot_obs']) # total number of systems
    N_pl = np.sum(ssk_per_sys['Mtot_obs']) # total number of planets
    
    # Check that all the fields in 'ssk_per_sys' have the same number of systems:
    assert N_sys == len(ssk_per_sys['Rstar_obs']) == len(ssk_per_sys['Mstar_obs'])
    assert N_sys == len(ssk_per_sys['teff_obs']) == len(ssk_per_sys['cdpp4p5_obs'])
    assert N_sys == len(ssk_per_sys['bp_rp_obs']) == len(ssk_per_sys['e_bp_rp_obs'])
    assert N_sys == len(ssk_per_sys['P_obs']) == len(ssk_per_sys['D_obs']) == len(ssk_per_sys['radii_obs'])
    assert N_sys == len(ssk_per_sys['tdur_obs']) == len(ssk_per_sys['tdur_tcirc_obs'])
    assert N_sys == len(ssk_per_sys['Rm_obs']) == len(ssk_per_sys['D_ratio_obs']) == len(ssk_per_sys['xi_obs'])
    
    # Check that all the fields in 'ssk_per_sys' have the same number of planets:
    assert N_pl == np.sum(ssk_per_sys['P_obs'] > 0) == np.sum(ssk_per_sys['D_obs'] > 0) == np.sum(ssk_per_sys['radii_obs'] > 0)
    assert N_pl == np.sum(ssk_per_sys['tdur_obs'] >= 0) == np.sum(ssk_per_sys['tdur_tcirc_obs'] >= 0)
    assert np.sum(ssk_per_sys['Rm_obs'] > 0) == np.sum(ssk_per_sys['D_ratio_obs'] > 0) == np.sum(ssk_per_sys['xi_obs'] >= 0)
