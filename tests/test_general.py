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
