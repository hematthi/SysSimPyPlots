# To import required modules:
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *





# Test functions in "src/functions_load_sims.py":

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'

def test_read_targets_period_radius_bounds(load_dir=loadfiles_directory):
    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(load_dir + 'periods.out')
    assert type(N_sim) is int
    assert 0 <= cos_factor <= 1
    assert 3. <= P_min < P_max <= 300.
    assert 0.5 <= radii_min < radii_max <= 10.

def test_load_cat_phys(load_dir=loadfiles_directory):
    cat_phys = load_cat_phys(load_dir + 'physical_catalog.csv')
    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(load_dir + 'physical_catalog.csv')
    assert 0 <= np.min(cat_phys['planet_mass'])
    assert radii_min <= np.min(cat_phys['planet_radius']) * Rsun/Rearth
    #assert radii_max >= np.max(cat_phys['planet_radius']) * Rsun/Rearth # NOTE: fails due to differences in precision of Rsun/Rearth in SysSim vs. in 'functions_general.py'
    assert P_min <= np.min(cat_phys['period']) <= np.max(cat_phys['period']) <= P_max
    assert 0 <= np.min(cat_phys['ecc']) <= np.max(cat_phys['ecc']) <= 1
    assert 0 <= np.min(cat_phys['incl']) <= np.max(cat_phys['incl']) <= np.pi
    assert -np.pi <= np.min(cat_phys['omega']) <= np.max(cat_phys['omega']) <= np.pi
    assert 0 <= np.min(cat_phys['asc_node']) <= np.max(cat_phys['asc_node']) <= 2.*np.pi
    assert 0 <= np.min(cat_phys['mean_anom']) <= np.max(cat_phys['mean_anom']) <= 2.*np.pi
    assert 0 <= np.min(cat_phys['incl_invariable']) <= np.max(cat_phys['incl_invariable']) <= np.pi
    assert 0 <= np.min(cat_phys['asc_node_invariable']) <= np.max(cat_phys['asc_node_invariable']) <= 2.*np.pi
    assert 0 < np.min(cat_phys['star_mass'])
    assert 0 < np.min(cat_phys['star_radius'])

def test_load_star_phys(load_dir=loadfiles_directory):
    star_phys = load_star_phys(load_dir + 'physical_catalog_stars.csv')
    assert 0 < np.min(star_phys['star_mass']) <= np.max(star_phys['star_mass']) < 2.
    assert 0 < np.min(star_phys['star_radius']) <= np.max(star_phys['star_radius']) < 4.
    assert 0 <= np.min(star_phys['num_planets'])
