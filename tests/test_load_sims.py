# To import required modules:
import numpy as np
import os
import sys
from itertools import chain # for rapid flattening of lists of lists

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *





# Test functions in "src/functions_load_sims.py":

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'

def test_read_targets_period_radius_bounds(load_dir=loadfiles_directory, run_number=''):
    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(load_dir + 'periods%s.out' % run_number)
    assert type(N_sim) is int
    assert 0 <= cos_factor <= 1
    assert 3. <= P_min < P_max <= 300.
    assert 0.5 <= radii_min < radii_max <= 10.

def test_load_cat_phys(load_dir=loadfiles_directory, run_number=''):
    cat_phys = load_cat_phys(load_dir + 'physical_catalog.csv')
    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(load_dir + 'physical_catalog%s.csv' % run_number)
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

def test_load_star_phys(load_dir=loadfiles_directory, run_number=''):
    star_phys = load_star_phys(load_dir + 'physical_catalog_stars%s.csv' % run_number)
    assert 0 < np.min(star_phys['star_mass']) <= np.max(star_phys['star_mass']) < 2.
    assert 0 < np.min(star_phys['star_radius']) <= np.max(star_phys['star_radius']) < 4.
    assert 0 <= np.min(star_phys['num_planets'])

def test_load_planets_stars_phys_separate(load_dir=loadfiles_directory, run_number=''):
    clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all = load_planets_stars_phys_separate(load_dir, run_number=run_number)
    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(load_dir + 'periods_all%s.out' % run_number)
    
    clusterids = list(chain(*clusterids_per_sys))
    periods = list(chain(*P_per_sys))
    radii = list(chain(*radii_per_sys))
    masses = list(chain(*mass_per_sys))
    eccs = list(chain(*e_per_sys))
    inclmuts = list(chain(*inclmut_per_sys))
    incls = list(chain(*incl_per_sys))
    N_sys_pl, N_pl = len(clusterids_per_sys), len(clusterids)
    
    assert N_pl == len(periods) == len(radii) == len(masses) == len(eccs) == len(inclmuts) == len(incls)
    assert N_sys_pl == len(Mstar_all) == len(Rstar_all) <= N_sim
    assert P_min <= np.min(periods) <= np.max(periods) <= P_max
    assert radii_min <= np.min(radii) * Rsun/Rearth
    #assert radii_max >= np.max(radii) * Rsun/Rearth # NOTE: fails due to differences in precision of Rsun/Rearth in SysSim vs. in 'functions_general.py'
    assert 0 <= np.min(masses)
    assert 0 <= np.min(eccs) <= np.max(eccs) <= 1
    assert 0 <= np.min(inclmuts) <= np.max(inclmuts) <= np.pi
    assert 0 <= np.min(incls) <= np.max(incls) <= np.pi
    assert 0 < np.min(Mstar_all)
    assert 0 < np.min(Rstar_all)

def test_compute_basic_summary_stats_per_sys_cat_phys(load_dir=loadfiles_directory, run_number=''):
    clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all = load_planets_stars_phys_separate(load_dir, run_number=run_number)
    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(load_dir + 'periods_all%s.out' % run_number)
    
    N_sys_pl, N_pl = len(clusterids_per_sys), len(list(chain(*clusterids_per_sys)))
    
    sssp_per_sys_basic = compute_basic_summary_stats_per_sys_cat_phys(clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all)
    assert type(sssp_per_sys_basic['Mmax']) is np.int64
    assert sssp_per_sys_basic['Mmax'] == np.max(sssp_per_sys_basic['Mtot_all'])
    assert N_sys_pl == len(sssp_per_sys_basic['Mtot_all'])
    assert N_pl == np.sum(sssp_per_sys_basic['Mtot_all']) == np.sum(sssp_per_sys_basic['pl_per_cluster_all'])
    
    # Check that all fields have the right number of systems:
    keys = ['P_all', 'clusterids_all', 'e_all', 'inclmut_all', 'incl_all', 'radii_all', 'mass_all', 'Mstar_all', 'Rstar_all', 'mu_all', 'a_all', 'AMD_all', 'AMD_tot_all']
    for key in keys:
        assert N_sys_pl == len(sssp_per_sys_basic[key])

def test_compute_summary_stats_from_cat_phys(load_dir=loadfiles_directory, run_number=''):
    clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all = load_planets_stars_phys_separate(load_dir, run_number=run_number)
    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(load_dir + 'periods_all%s.out' % run_number)
    
    N_sys_pl, N_pl = len(clusterids_per_sys), len(list(chain(*clusterids_per_sys)))
    
    sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=load_dir, run_number=run_number, load_full_tables=True, match_observed=True)
    N_pl_pairs = np.sum(sssp_per_sys['Mtot_all'][sssp_per_sys['Mtot_all'] >= 2] - 1) # number of adjacent pairs of planets
    assert N_pl == np.sum(sssp_per_sys['Mtot_all'])
    
    # Check that all fields in 'sssp_per_sys' have the right number of systems:
    keys = ['clusterids_all', 'P_all', 'a_all', 'radii_all', 'mass_all', 'mu_all', 'e_all', 'inclmut_all', 'incl_all', 'AMD_all', 'Rm_all', 'radii_ratio_all', 'N_mH_all', 'dynamical_mass']
    for key in keys:
        assert N_sim == len(sssp_per_sys[key])
    assert len(sssp_per_sys['radii_partitioning']) == np.sum(sssp_per_sys['Mtot_all'] >= 2)
    assert len(sssp_per_sys['radii_monotonicity']) == np.sum(sssp_per_sys['Mtot_all'] >= 2)
    assert len(sssp_per_sys['gap_complexity']) == np.sum(sssp_per_sys['Mtot_all'] >= 3)
    
    # Check that all fields in 'sssp' have the right number of systems and planets:
    assert np.sum(sssp['clustertot_all']) == len(sssp['pl_per_cluster_all'])
    assert N_pl == np.sum(sssp['pl_per_cluster_all'])
    
    keys = ['Mstar_all', 'Rstar_all', 'clustertot_all', 'AMD_tot_all']
    for key in keys:
        assert N_sim == len(sssp[key])
    
    keys = ['P_all', 'radii_all', 'mass_all', 'e_all', 'incl_all']
    for key in keys:
        assert N_pl == len(sssp[key])
    assert np.sum(sssp_per_sys['Mtot_all'][sssp_per_sys['Mtot_all'] >= 2]) == len(sssp['inclmut_all'])
    assert N_pl == len(sssp['radii_above_all']) + len(sssp['radii_below_all'])
    assert N_pl_pairs == len(sssp['Rm_all'])
    assert N_pl_pairs == len(sssp['radii_ratio_all'])
    assert N_pl_pairs == len(sssp['N_mH_all'])
    assert N_pl_pairs == len(sssp['radii_ratio_above_all']) + len(sssp['radii_ratio_below_all']) + len(sssp['radii_ratio_across_all'])
    
    # Check that each field in 'sssp' has reasonable values:
    assert 0 < np.min(sssp['Mstar_all'])
    assert 0 < np.min(sssp['Rstar_all'])
    assert 0 <= np.min(sssp['clustertot_all'])
    assert 0 <= np.min(sssp['AMD_tot_all'])
    assert 1 <= np.min(sssp['pl_per_cluster_all'])
    assert P_min <= np.min(sssp['P_all']) <= np.max(sssp['P_all']) <= P_max
    assert radii_min <= np.min(sssp['radii_all']) <= np.max(sssp['radii_all']) #<= radii_max # NOTE: 'radii_max' check fails due to differences in precision of Rsun/Rearth in SysSim vs. in 'functions_general.py'
    assert 0 <= np.min(sssp['mass_all'])
    assert 0 <= np.min(sssp['e_all']) <= np.max(sssp['e_all']) <= 1
    assert 0 <= np.min(sssp['inclmut_all']) <= np.max(sssp['inclmut_all']) <= np.pi
    assert 0 <= np.min(sssp['incl_all']) <= np.max(sssp['incl_all']) <= np.pi
    assert radii_min <= np.min(sssp['radii_above_all']) <= np.max(sssp['radii_above_all']) #<= radii_max
    assert radii_min <= np.min(sssp['radii_below_all']) <= np.max(sssp['radii_below_all']) #<= radii_max
    assert 1 < np.min(sssp['Rm_all']) <= np.max(sssp['Rm_all']) <= P_max/P_min
    assert radii_min/radii_max <= np.min(sssp['radii_ratio_all']) <= np.max(sssp['radii_ratio_all']) <= radii_max/radii_min
    assert 0 < np.min(sssp['N_mH_all'])
    assert radii_min/radii_max <= np.min(sssp['radii_ratio_above_all']) <= np.max(sssp['radii_ratio_above_all']) <= radii_max/radii_min
    assert radii_min/radii_max <= np.min(sssp['radii_ratio_below_all']) <= np.max(sssp['radii_ratio_below_all']) <= radii_max/radii_min
    assert radii_min/radii_max <= np.min(sssp['radii_ratio_across_all']) <= np.max(sssp['radii_ratio_across_all']) <= radii_max/radii_min
