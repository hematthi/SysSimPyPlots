# To import required modules:
import numpy as np
import os
import sys

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *





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
    N_multis = np.sum(ssk['Nmult_obs'][1:]) # total number of multi-planet systems
    assert N_sys == np.sum(ssk['Nmult_obs'])
    N_pl = np.sum(ssk_per_sys['Mtot_obs']) # total number of planets
    assert N_pl == np.sum(ssk['Nmult_obs'] * np.arange(1,len(ssk['Nmult_obs'])+1))
    N_pl_pairs = np.sum(ssk['Nmult_obs'] * np.arange(len(ssk['Nmult_obs']))) # total number of adjacent planet pairs

    # Check that all the fields in 'ssk_per_sys' have the same number of systems:
    keys = ['Rstar_obs', 'Mstar_obs', 'teff_obs', 'bp_rp_obs', 'e_bp_rp_obs', 'cdpp4p5_obs', 'P_obs', 'D_obs', 'tdur_obs', 'tdur_tcirc_obs', 'radii_obs', 'Rm_obs', 'D_ratio_obs', 'xi_obs', 'radii_star_ratio']
    for key in keys:
        assert N_sys == len(ssk_per_sys[key])

    assert N_multis == len(ssk_per_sys['radii_partitioning'])
    assert N_multis == len(ssk_per_sys['radii_monotonicity'])
    assert np.sum(ssk['Nmult_obs'][2:]) == len(ssk_per_sys['gap_complexity'])

    # Check that all the fields in 'ssk_per_sys' have the same number of planets:
    assert N_pl == np.sum(ssk_per_sys['P_obs'] > 0) == np.sum(ssk_per_sys['D_obs'] > 0) == np.sum(ssk_per_sys['radii_obs'] > 0)
    assert N_pl == np.sum(ssk_per_sys['tdur_obs'] >= 0) == np.sum(ssk_per_sys['tdur_tcirc_obs'] >= 0)
    assert N_pl_pairs == np.sum(ssk_per_sys['Rm_obs'] > 0) == np.sum(ssk_per_sys['D_ratio_obs'] > 0) == np.sum(ssk_per_sys['xi_obs'] >= 0)

    # Check that all the fields in 'ssk' have the right number of planets:
    keys = ['Rstar_obs', 'Mstar_obs', 'teff_obs', 'bp_rp_obs', 'e_bp_rp_obs', 'cdpp4p5_obs', 'P_obs', 'D_obs', 'tdur_obs', 'tdur_tcirc_obs', 'radii_obs']
    for key in keys:
        assert N_pl == len(ssk[key])
    assert N_pl == len(ssk['tdur_tcirc_obs']) == len(ssk['tdur_tcirc_1_obs']) + len(ssk['tdur_tcirc_2p_obs'])
    assert N_pl == len(ssk['D_obs']) == len(ssk['D_above_obs']) + len(ssk['D_below_obs'])

    assert N_pl_pairs == len(ssk['Rm_obs'])
    assert N_pl_pairs == len(ssk['D_ratio_obs']) == len(ssk['D_ratio_above_obs']) + len(ssk['D_ratio_below_obs']) + len(ssk['D_ratio_across_obs'])
    assert N_pl_pairs == len(ssk['xi_obs']) == len(ssk['xi_res_obs']) + len(ssk['xi_nonres_obs'])
    assert len(ssk['xi_res_obs']) >= len(ssk['xi_res32_obs']) + len(ssk['xi_res21_obs'])

def test_CRPD_dist(seed=42):
    np.random.seed(seed)
    x = np.array([1205, 252, 97, 29, 7, 3])
    assert CRPD_dist(x, x) == CRPD_dist(x, x*np.random.randint(1,10)) == 0
    assert CRPD_dist(np.random.randint(1, 1000, 5), np.random.randint(1, 1000, 5)) > 0 # WARNING: this can potentially fail in edge cases!

def test_KS_dist_mult(seed=42):
    np.random.seed(seed)
    x1, x2 = np.random.randint(1, 10, 1000), np.random.randint(1, 10, 1000)
    x3 = np.random.randint(50, 100, 1000)
    assert KS_dist_mult(x1, x1)[0] == KS_dist_mult(x2, x2)[0] == KS_dist_mult(x3, x3)[0] == 0
    assert 0 <= KS_dist_mult(x1, x2)[0] <= 1
    assert np.isclose(KS_dist_mult(x1, x2)[0], KS_dist_mult(x2, x1)[0])
    assert np.isclose(KS_dist_mult(x1, x3)[0], 1)

def test_KS_dist(seed=42):
    np.random.seed(seed)
    x1, x2 = np.random.rand(1000), np.random.randn(500)
    x1b = np.random.rand(1000)
    x3 = np.random.uniform(10., 100., 300)
    assert 0 <= KS_dist(x1, x2)[0] <= 1
    assert np.isclose(KS_dist(x1, x2)[0], KS_dist(x2, x1)[0])
    assert KS_dist(x1, x1b)[0] < KS_dist(x1, x2)[0]
    assert np.isclose(KS_dist(x1, x3)[0], 1)

def test_AD_dist(seed=42):
    np.random.seed(seed)
    x1, x2 = np.random.rand(1000), np.random.randn(500)
    x1b = np.random.rand(1000)
    x3 = np.random.uniform(10., 100., 300)

    # Test 'AD_dist()':
    assert 0 <= AD_dist(x1, x2)
    assert np.isclose(AD_dist(x1, x2), AD_dist(x2, x1))
    assert AD_dist(x1, x1b) < AD_dist(x1, x2) < AD_dist(x1, x3)

    # Test that 'AD_dist2()' returns the same results as 'AD_dist()':
    assert np.isclose(AD_dist(x1, x2), AD_dist2(x1, x2))
    assert np.isclose(AD_dist(x1, x1b), AD_dist2(x1, x1b))
    assert np.isclose(AD_dist(x1, x3), AD_dist2(x1, x3))

    # Test 'AD_mod_dist()':
    assert 0 <= AD_mod_dist(x1, x2)
    assert np.isclose(AD_mod_dist(x1, x2), AD_mod_dist(x2, x1))
    assert AD_mod_dist(x1, x1b) < AD_mod_dist(x1, x2) < AD_mod_dist(x1, x3)

def test_compute_distances_sim_Kepler():
    loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
    compute_ratios = compute_ratios_adjacent
    AD_mod = True
    weights_all = load_split_stars_weights_only()
    dists_include = ['delta_f',
                     'mult_CRPD_r',
                     'periods_KS',
                     'period_ratios_KS',
                     'durations_KS',
                     'durations_norm_circ_KS',
                     'durations_norm_circ_singles_KS',
                     'durations_norm_circ_multis_KS',
                     'duration_ratios_nonmmr_KS',
                     'duration_ratios_mmr_KS',
                     'depths_KS',
                     'radius_ratios_KS',
                     'radii_partitioning_KS',
                     'radii_monotonicity_KS',
                     'gap_complexity_KS',
                     ]

    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods.out')

    sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, compute_ratios=compute_ratios)
    ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

    dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)

    for key in dists.keys():
        assert dists[key] >= 0 # check that all distance terms are non-negative
        if key[-2:] == 'KS':
            assert dists[key] <= 1
    for key in dists_w.keys():
        assert dists_w[key] >= 0 # check that all weighted distance terms are non-negative

    # Also implicitly tests 'compute_total_weighted_dist()':
    tot_dist_w = 0.
    for key in dists_include:
        tot_dist_w += dists_w[key]
    assert np.isclose(dists_w['tot_dist_w_include'], tot_dist_w)
