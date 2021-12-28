# To import required modules:
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.functions_general import *





# Test functions in "src/functions_general.py":

def test_a_from_P():
    assert np.isclose(a_from_P(365.25, 1.), 1.) # Earth's orbit

def test_P_from_a():
    assert np.isclose(P_from_a(1., 1.), 365.25) # Earth's orbit

def test_M_from_R_rho():
    assert np.isclose(M_from_R_rho(1.), 1., atol=1e-3) # Earth mass/radius
    assert M_from_R_rho(0.) == 0.

def test_rho_from_M_R():
    assert np.isclose(rho_from_M_R(1., 1.), 5.513259, atol=1e-5) # Earth mean density

def test_tdur_circ():
    assert np.isclose(tdur_circ(365.25, 1., 1.), 12.976035, atol=1e-5) # transit duration of Earth
