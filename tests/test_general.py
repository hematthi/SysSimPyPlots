# To import required modules:
import numpy as np
import os
import sys
from scipy.special import comb

from syssimpyplots.general import *





# Test functions in "src/functions_general.py":

def test_a_from_P():
    assert np.isclose(a_from_P(365.25, 1.), 1.) # Earth's orbit

def test_P_from_a():
    assert np.isclose(P_from_a(1., 1.), 365.25) # Earth's orbit

def test_a_from_P_from_a(seed=42):
    np.random.seed(seed)
    Mstar, a = np.random.rand(2)
    assert np.isclose(a_from_P(P_from_a(a, Mstar), Mstar), a)

def test_M_from_R_rho():
    assert np.isclose(M_from_R_rho(1.), 1., atol=1e-3) # Earth mass/radius
    assert M_from_R_rho(0.) == 0

def test_rho_from_M_R():
    assert np.isclose(rho_from_M_R(1., 1.), 5.513259, atol=1e-5) # Earth mean density

def test_tdur_circ():
    assert np.isclose(tdur_circ(365.25, 1., 1.), 12.976035, atol=1e-5) # transit duration of Earth

def test_AMD(seed=42):
    np.random.seed(seed)
    assert AMD(1e-6*np.random.rand(), np.random.rand(), 0., 0.) == 0
    assert np.isclose(AMD(1e-6, 1., 0.5, 0.), 1.339746e-7, atol=1e-5)
    assert np.isclose(AMD(1e-5, 0.5, 0.1, 0.1), 7.059299e-8, atol=1e-5)

def test_photoevap_boundary_Carrera2018():
    assert photoevap_boundary_Carrera2018(2.61, 1.)
    assert photoevap_boundary_Carrera2018(1.85, 10.) == 0
    assert photoevap_boundary_Carrera2018(1.5, 50.)

def test_incl_mult_power_law_Zhu2018():
    assert np.isclose(incl_mult_power_law_Zhu2018(5), 0.8)
    assert np.isclose(incl_mult_power_law_Zhu2018(2), 19.764235, atol=1e-5)
    assert np.isclose(incl_mult_power_law_Zhu2018(4, sigma_5=1., alpha=-2.), 1.5625, atol=1e-5)

def test_cdf_normal():
    assert np.isclose(cdf_normal(0), 0.5)
    assert np.isclose(cdf_normal(1.), 0.841345, atol=1e-5)
    assert np.isclose(cdf_normal(0.)-cdf_normal(-1.), cdf_normal(1.)-cdf_normal(0.))
    assert np.isclose(cdf_normal(0.)-cdf_normal(-3.), cdf_normal(3.)-cdf_normal(0.))

def test_cdf_empirical(seed=42):
    np.random.seed(seed)
    x_unif = np.random.rand(10_000_000) # Unif(0,1)
    x_ptls = np.random.rand(10) # randomly generated percentiles
    assert np.allclose(cdf_empirical(x_unif, x_ptls), x_ptls, atol=1e-3)

    x_norm = np.random.randn(10_000_000) # Normal(0,1)
    x_evals = 2.*np.random.rand(10) - 1. # Unif(-1,1)
    assert np.allclose(cdf_empirical(x_norm, x_evals), cdf_normal(x_evals), atol=1e-3)

def test_compute_ratios():
    x1 = np.array([0.5, 1., 2., 4.])
    x2 = np.array([88.0, 224.7, 365.2, 687.0, 4331., 10747., 30589., 59800.]) # periods (days) of Solar System planets
    x3 = np.array([-2., 1.4, 5.3, -0.33, np.pi])

    # Testing 'compute_ratios_adjacent()':
    assert np.allclose(compute_ratios_adjacent(x1), 2*np.ones(len(x1)-1))
    assert np.allclose(compute_ratios_adjacent(x1)*compute_ratios_adjacent(x1[::-1]), np.ones(len(x1)-1))
    assert np.allclose(compute_ratios_adjacent(x2), [2.553409, 1.625278, 1.881161, 6.304221, 2.481413, 2.846283, 1.954951], atol=1e-5)
    assert np.allclose(compute_ratios_adjacent(x3), [-0.7, 3.785714, -0.062264, -9.519978], atol=1e-5)
    assert len(compute_ratios_adjacent([5.])) == len(compute_ratios_adjacent([])) == 0

    # Testing 'compute_ratios_all()':
    assert np.allclose(compute_ratios_all(x1), [2, 4, 8, 2, 4, 2])
    assert len(compute_ratios_all(x2)) == comb(len(x2),2)
    assert len(compute_ratios_all([5.])) == len(compute_ratios_all([])) == 0

def test_mmrs_bounds_zeta(i_max=20):
    # Testing 'bounds_3rd_order_mmr_neighborhood_of_1st_order_mmr()':
    bounds_3rd_order_mmr = [bounds_3rd_order_mmr_neighborhood_of_1st_order_mmr(i) for i in range(1,i_max+1)]
    for idx,bounds in enumerate(bounds_3rd_order_mmr[:-1]):
        i = idx+1
        # Check that the bounds are valid and bracket the correct 1st order MMR:
        assert bounds[0] < (i+1)/i < bounds[1]
        # Check that the neighboring bounds do not overlap, and bracket the correct 2nd order MMR:
        bounds_next = bounds_3rd_order_mmr[idx+1]
        n_2nd = 2*i+1 # the relevant index for the 2nd order MMR just interior, i.e. n in (n+2)/n (NOTE: skips n=1 though since that is exterior to the 2:1 MMR)
        assert bounds_next[1] < (n_2nd+2)/n_2nd < bounds[0]
    assert bounds_next[0] > 1 # check that the smallest inner bound is still greater than 1
    
    # Testing 'bounds_3rd_order_mmr_neighborhood_of_2nd_order_mmr()':
    bounds_3rd_order_mmr = [bounds_3rd_order_mmr_neighborhood_of_2nd_order_mmr(i) for i in range(1,i_max+1)]
    for idx,bounds in enumerate(bounds_3rd_order_mmr[:-1]):
        i = idx+1
        n = 2*i-1
        # Check that the bounds are valid and bracket the correct 2nd order MMR:
        assert bounds[0] < (n+2)/n < bounds[1]
        # Check that the neighboring bounds do not overlap, and bracket the correct 1st order MMR:
        bounds_next = bounds_3rd_order_mmr[idx+1]
        assert bounds_next[1] < (i+1)/i < bounds[0]
    assert bounds_next[0] > 1 # check that the smallest inner bound is still greater than 1
    
    pratios = np.linspace(1.00001, 5., 10000)
    
    # Testing 'pratio_is_in_any_1st_order_mmr_neighborhood()':
    bools_1st_all, i_all = pratio_is_in_any_1st_order_mmr_neighborhood(pratios, i_max=i_max)
    assert np.sum(bools_1st_all) == np.sum(i_all != 0)
    assert np.min(i_all[bools_1st_all]) == 1
    assert np.max(i_all[bools_1st_all]) == i_max
    assert np.all(i_all[pratios > 2.5] == 0)
    bools_1st_all_double_imax, i_all_double_imax = pratio_is_in_any_1st_order_mmr_neighborhood(pratios, i_max=2*i_max)
    assert np.sum(bools_1st_all) <= np.sum(bools_1st_all_double_imax)
    
    # Testing 'pratio_is_in_any_2nd_order_mmr_neighborhood()':
    bools_2nd_all, i_all = pratio_is_in_any_2nd_order_mmr_neighborhood(pratios, i_max=i_max)
    assert np.sum(bools_2nd_all) == np.sum(i_all != 0)
    assert np.min(i_all[bools_2nd_all]) == 1
    assert np.max(i_all[bools_2nd_all]) == i_max
    assert np.all(i_all[pratios > 4.] == 0)
    bools_2nd_all_double_imax, i_all_double_imax = pratio_is_in_any_2nd_order_mmr_neighborhood(pratios, i_max=2*i_max)
    assert np.sum(bools_2nd_all) <= np.sum(bools_2nd_all_double_imax)
    
    assert ~np.any(bools_1st_all & bools_2nd_all) # check that no period ratios are counted in both a 1st and 2nd order MMR
    
    # Testing 'zeta()':
    zetas_1_1 = zeta(pratios, n=1, order=1)
    zetas_2_1 = zeta(pratios[bools_1st_all], n=2, order=1)
    zetas_1_2 = zeta(pratios, n=1, order=2)
    zetas_2_2 = zeta(pratios[bools_2nd_all], n=2, order=2)
    assert np.all(np.abs(zetas_1_1) <= 1.)
    assert np.all(np.abs(zetas_2_1) <= 1.)
    assert np.all(np.abs(zetas_1_2) <= 1.)
    assert np.all(np.abs(zetas_2_2) <= 1.)
    
    # Testing 'zeta_2_order()':
    zetas_2_1or2, bools_in_1st, bools_in_2nd = zeta_2_order(pratios[bools_1st_all | bools_2nd_all])
    assert ~np.any(bools_in_1st & bools_in_2nd) # check that no period ratios are counted in both a 1st and 2nd order MMR
    assert np.all(np.abs(zetas_2_1or2) <= 1.)
    assert np.all(np.isclose(zetas_2_1, zetas_2_1or2[bools_in_1st]))
    assert np.all(np.isclose(zetas_2_2, zetas_2_1or2[bools_in_2nd]))

def test_linear_fswp_bprp(seed=42):
    np.random.seed(seed)
    fswp_med, slope = np.random.rand(), 2.*np.random.rand() - 1.
    bprp_med = 0.81
    bprp_rand = np.random.rand(1000)
    fswp_rand = linear_fswp_bprp(bprp_rand, bprp_med, fswp_med=fswp_med, slope=slope)
    assert np.isclose(linear_fswp_bprp([bprp_med], bprp_med, fswp_med=fswp_med, slope=slope), fswp_med)
    assert np.all((0. <= fswp_rand) & (fswp_rand <= 1.))

def test_bin_Nmult(seed=42):
    np.random.seed(seed)
    Nmult_rand = np.random.randint(0, 1000, 8)
    Nmult_bin5p = bin_Nmult(Nmult_rand)
    assert np.sum(Nmult_bin5p) == np.sum(Nmult_rand)
    assert np.all(Nmult_rand[:4] == Nmult_bin5p[:4])
    assert Nmult_bin5p[-1] == np.sum(Nmult_rand[4:])

def test_correlation_coefficients(seed=42):
    np.random.seed(seed)
    x, y = np.random.rand(10000), np.random.rand(10000)
    assert np.abs(Pearson_correlation_coefficient(x, y)) < 0.05
    assert np.abs(Spearman_correlation_coefficient(x, y)) < 0.05
    y = x*np.random.rand()
    assert np.isclose(Pearson_correlation_coefficient(x, y), 1.)
    assert np.isclose(Spearman_correlation_coefficient(x, y), 1.)

def test_radii_star_ratio():
    r, Rstar = np.array([0.0134, 0.0085, 0.0211]), 0.94
    mu = radii_star_ratio(r, Rstar)
    assert np.isclose(mu, 0.045745, atol=1e-5)

def test_partitioning(seed=42):
    np.random.seed(seed)
    x = np.random.rand(10)
    x_lowpart = np.array([2., 2., 2., 2.])
    x_highpart = np.array([1., 1., 1., 5.])
    assert 0 <= partitioning(x) <= 1
    assert partitioning(x_lowpart) < partitioning(x_highpart)
    assert partitioning(np.random.rand()*x_lowpart) == 0

def test_monotonicity_GF2020():
    x_pos1, x_pos2 = np.array([1.1, 1.2, 1.3, 1.4]), np.array([1., 2., 3., 4.])
    x_neg1, x_neg2 = np.array([1.2, 2., 1.5, 0.9]), np.array([2., 1.5, 1.2, 0.9])
    assert 0 < monotonicity_GF2020(x_pos1) < monotonicity_GF2020(x_pos2)
    assert monotonicity_GF2020(x_neg2) < monotonicity_GF2020(x_neg1) < 0

def test_gap_complexity_GF2020():
    x_lowgc = np.array([1., 2., 4., 8.])
    x_highgc = np.array([1., 1.2, 8., 9.5])
    assert 0 <= gap_complexity_GF2020(x_lowgc) < gap_complexity_GF2020(x_highgc) <= 1
    assert np.isclose(gap_complexity_GF2020(x_highgc), gap_complexity_GF2020(x_highgc[::-1]))
