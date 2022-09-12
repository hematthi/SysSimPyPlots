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





stars_cleaned = load_Kepler_stars_cleaned()
stars_cleaned = np.sort(stars_cleaned, order='teff') # sort stellar catalog by T_eff

def compute_HZ_Kopparapu_2014(teff, lum):
    # Compute the conservative and optimistic habitable zone boundaries given a stellar effective temperature 'teff' (K) and luminosity 'lum' (L_sun) from Kopparapu et al. (2014): http://depts.washington.edu/naivpl/sites/default/files/hz.shtml
    # Assumes an Earth-mass planet, and returns an array of [OHZ_in, CHZ_in, CHZ_out, OHZ_out] (AU)
    # OHZ_in and OHZ_out correspond to recent Venus and early Mars limits
    # CHZ_in and CHZ_out correspond to runaway greenhouse and maximum greenhouse limits

    seffsun  = np.array([1.776, 1.107, 0.356, 0.320])
    a = np.array([2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5])
    b = np.array([2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9])
    c = np.array([-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12])
    d = np.array([-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16])

    tstar = teff - 5780.
    seff = seffsun + a*tstar + b*tstar**2. + c*tstar**3. + d*tstar**4 # HZ fluxes (F_sun)
    d = np.sqrt(lum/seff) # HZ distance boundaries (AU)
    return d





##### To compute the HZ for each star in our catalog:

teff_all = stars_cleaned['teff']
HZ_all = [] # optimistic and conservative HZ boundaries (AU) for each star
for i,teff in enumerate(teff_all):
    HZ_boundaries = compute_HZ_Kopparapu_2014(teff, stars_cleaned['lum_val'][i])
    HZ_all.append(HZ_boundaries)
HZ_all = np.array(HZ_all)

##### To plot the HZ for each star:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(5,1,left=0.075,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)

ax = plt.subplot(plot[1:,:]) # HZ regions vs. T_eff
plt.fill_between(teff_all, HZ_all[:,0], HZ_all[:,3], alpha=0.2, color='b', label='Optimistic HZ')
plt.fill_between(teff_all, HZ_all[:,1], HZ_all[:,2], alpha=0.2, color='g', label='Conservative HZ')
ax.tick_params(axis='both', labelsize=16)
plt.xlim([4000.,7000.])
plt.xlabel(r'Stellar effective temperature $T_{\rm eff}$ (K)', fontsize=20)
plt.ylabel(r'Distance $a$ (AU)', fontsize=20)

ax = plt.subplot(plot[0,:]) # histogram of T_eff
plt.hist(teff_all, bins=100, histtype='step', color='k', ls='-')
ax.tick_params(axis='both', labelsize=16)
plt.xlim([4000.,7000.])
plt.xticks([])
plt.yticks([])
#plt.minorticks_off()

plt.show()
