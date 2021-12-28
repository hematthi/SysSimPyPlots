# To import required modules:
import numpy as np
from matplotlib.colors import LogNorm #for log color scales
from scipy.special import erf #error function, used in computing CDF of normal distribution





# Useful fundamental constants:

AU = 1.496*10.**13. # AU in cm
Msun = 1.989*10.**30. # Solar mass in kg
Rsun = 6.957*10.**10. # Solar radius in cm
Mearth = 5.972*10.**24. # Earth mass in kg
Rearth = 6.371*10.**8. # Earth radius in cm
Mjup = 1.898*10.**27. # Jupiter mass in kg
Rjup = 6.991*10.**9. # Jupiter radius in cm





# Miscellaneous definitions:

res_ratios, res_width = [2.0, 1.5, 4/3., 5/4.], 0.05 #NOTE: in the model, the near-resonant planets have period ratios between X and (1+w)*X where X = [2/1, 3/2, 4/3, 5/4] and w = 0.05!





# Miscellaneous functions:

def a_from_P(P, Mstar):
    # Convert period (days) to semi-major axis (AU) assuming mass of planet m << Mstar (Msun)
    y = (P/365.25)**(2./3.)*(Mstar/1.0)**(1./3.)
    return y

def P_from_a(a, Mstar):
    # Convert semi-major axis (AU) to period (days) assuming mass of planet m << Mstar (Msun)
    y = 365.25*(a**(3./2.))*(Mstar/1.0)**(-1./2.)
    return y

def M_from_R_rho(R, rho=5.51):
    # Compute planet mass (M_earth) from radius (R_earth) given a constant mean density rho (g/cm^3)
    M_in_g = rho * (4.*np.pi/3.)*(R*Rearth)**3.
    return M_in_g/(Mearth*1000.)

def rho_from_M_R(M, R):
    # Compute the mean density rho (g/cm^3) given a planet mass (M_earth) and radius (R_earth)
    rho = (M*Mearth*1000.) / ((4.*np.pi/3.)*(R*Rearth)**3.)
    return rho

def tdur_circ(P, Mstar, Rstar):
    # Calculate the transit duration (hrs) assuming a circular orbit with b=0, given a period (days), Mstar (Msun), and Rstar (Rsun)
    y = 24.*(Rstar*Rsun*P)/(np.pi*a_from_P(P,Mstar)*AU)
    return y

def AMD(mu, a, e, im):
    # Compute the AMD (angular momentum deficit) of a planet given the planet/star mass ratio mu, semimajor axis a (AU), eccentricity e, and mutual inclination im (rad); assumes GMstar = 1
    y = mu*np.sqrt(a) * (1. - np.sqrt(1. - e**2.)*np.cos(im))
    return y

def photoevap_boundary_Carrera2018(R, P):
    # R is the planet radius in Earth radii, P is the period in days
    # This function returns 1 if the planet is above the boundary, and 0 if the planet is below the boundary as defined by Eq. 5 in Carrera et al 2018
    Rtrans = 2.6*P**(-0.1467)
    if R >= Rtrans:
        above_boundary = 1
    else:
        above_boundary = 0
    return above_boundary

def incl_mult_power_law_Zhu2018(k, sigma_5=0.8, alpha=-3.5):
    # Compute the mutual inclination Rayleigh scale as a function of planet multiplicity using the power-law relation from Zhu et al 2018
    # Default parameters set to the best-fit values they found, sigma_5 = 0.8 (deg) and alpha = -3.5
    return sigma_5*(k/5.)**alpha

def cdf_normal(x, mu=0., std=1.):
    # This function computes the CDF (i.e. the integral of the normal distribution between -inf and x) at x given mean 'mu' and standard deviation 'std'
    # Note: this function can deal with array inputs for x, mu and std, as long as the array inputs are the same shape
    return 0.5*(1. + erf((x - mu)/(std*np.sqrt(2))))

def cdf_empirical(xdata, xeval):
    # Compute the CDF at 'xeval' given a sample 'xdata'
    # Note: this function is designed to deal with either scalar or array inputs of 'xeval'
    N = len(xdata)
    xeval = np.asarray(xeval)
    is_xeval_scalar = False
    if xeval.ndim == 0:
        xeval = xeval[None] # turn x_eval into array with 1 element
        is_xeval_scalar = True
    
    cdf_at_xeval = np.zeros(len(xeval))
    for i,x in enumerate(xeval):
        cdf_at_xeval[i] = np.sum(xdata <= x)/N
    
    if is_xeval_scalar:
        return np.squeeze(cdf_at_xeval)
    return cdf_at_xeval

def calc_f_near_pratios(sssp_per_sys, pratios=res_ratios, pratio_width=res_width):
    # This function computes the intrinsic fraction of planets near a period ratio with another planet for any period ratio in the given list of period ratios, with 'near' defined as being between 'pratio' and 'pratio*(1+pratio_width)' for 'pratio' in 'pratios'; defaults to calculating the fraction of all planets near an MMR
    count_mmr = 0
    for p_sys in sssp_per_sys['P_all']:
        p_sys = p_sys[p_sys > 0]
        mmr_sys = np.zeros(len(p_sys))
        pr_sys = p_sys[1:]/p_sys[:-1]
        pr_mmr_sys = np.zeros(len(pr_sys))
        
        for mmr in res_ratios:
            pr_mmr_sys[(pr_sys >= mmr) & (pr_sys <= mmr*(1.+res_width))] = 1
        for j,res in enumerate(pr_mmr_sys):
            if res == 1:
                mmr_sys[j] = 1
                mmr_sys[j+1] = 1
        count_mmr += np.sum(mmr_sys == 1)
    f_mmr = float(count_mmr)/np.sum(sssp_per_sys['P_all'] > 0)
    return f_mmr

def compute_ratios_adjacent(x):
    # This function computes an array of the adjacent ratios (x[j+1]/x[j]) of the terms given in an input array
    if len(x) <= 1:
        return np.array([])
    
    return x[1:]/x[0:-1]

def compute_ratios_all(x):
    # This function computes an array of all the unique ratios (x[j]/x[i] for j > i) of the terms given in an input array
    if len(x) <= 1:
        return np.array([])
    
    ratios = list(x[1:]/x[0])
    for i in range(len(x)-2):
        ratios += list(x[i+2:]/x[i+1])
    return np.array(ratios)

def zeta1(pratios):
    # This function computes the zeta statistic for first-order resonances as defined in Fabrycky et al 2014
    return 3.*((1./(pratios - 1.)) - np.round(1./(pratios - 1.)))

def split_colors_per_cdpp_bin(stars_cleaned, nbins=10):
    # This function computes a histogram of CDPP values (log bins), then splits each bin by bp-rp color into a bluer half (small bp-rp) and redder half (large bp-rp)
    cdpp_min, cdpp_max = np.min(stars_cleaned['rrmscdpp04p5']), np.max(stars_cleaned['rrmscdpp04p5'])
    bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
    counts, bins = np.histogram(stars_cleaned['rrmscdpp04p5'], bins=np.logspace(np.log10(cdpp_min), np.log10(cdpp_max), nbins+1))
    i_blue_per_bin, i_red_per_bin = [], []
    for i in range(nbins):
        i_bin = np.where((stars_cleaned['rrmscdpp04p5'] >= bins[i]) & (stars_cleaned['rrmscdpp04p5'] < bins[i+1]))[0]
        bp_rp_bin = stars_cleaned['bp_rp'][i_bin]
        bp_rp_bin_med = np.nanmedian(bp_rp_bin)
        i_bin_blue = i_bin[bp_rp_bin <= bp_rp_bin_med]
        i_bin_red = i_bin[bp_rp_bin > bp_rp_bin_med]
        i_blue_per_bin += list(i_bin_blue)
        i_red_per_bin += list(i_bin_red)
    i_blue_per_bin = np.array(i_blue_per_bin)
    i_red_per_bin = np.array(i_red_per_bin)
    return bins, i_blue_per_bin, i_red_per_bin

def linear_fswp_bprp(bprp, bprp_med, fswp_med=0.5, slope=0.):
    bprp = np.array(bprp)
    fswp_bprp = slope*(bprp - bprp_med) + fswp_med
    fswp_bprp[fswp_bprp < 0] = 0.
    fswp_bprp[fswp_bprp > 1] = 1.
    return fswp_bprp

def linear_alphaP_bprp(bprp, bprp_med, alphaP_med=0.5, slope=0.):
    bprp = np.array(bprp)
    alphaP_bprp = slope*(bprp - bprp_med) + alphaP_med
    return alphaP_bprp

def bin_Nmult(Nmult_obs, m_geq=5):
    # This function bins an observed multiplicity distribution at multiplicity orders greater than or equal to 'm_geq' (default to m_geq=5, so returns counts for m=1,2,3,4,5+)
    Nmult_obs = list(Nmult_obs) + [0]*(m_geq-len(Nmult_obs)) # zero-pad first
    Nmult_obs[m_geq-1] = np.sum(Nmult_obs[m_geq-1:]) # bin everything greater than or equal to m_geq
    return np.array(Nmult_obs[:m_geq])





# Information theory quantities and metrics from Gilbert & Fabrycky 2020 (GF2020):

def Shannon_entropy(p):
    # Assuming natural log, although any choice is valid
    assert all(p >= 0), 'Negative probabilities!'
    assert all(p <= 1), 'Probabilities greater than 1!'
    
    H = -np.sum(p*np.log(p))
    return H

def disequilibrium(p):
    assert all(p >= 0), 'Negative probabilities!'
    assert all(p <= 1), 'Probabilities greater than 1!'
    
    D = np.sum((p - (1./len(p)))**2.)
    return D

def LMC_complexity(K, p):
    # Lopez-Ruiz, Mancini, & Calbet (1995) complexity; product of Shannon entropy and disequilibrium
    assert K > 0
    
    H = Shannon_entropy(p)
    D = disequilibrium(p)
    C = K*H*D
    return C

def Pearson_correlation_coefficient(x, y):
    xmean, ymean = np.mean(x), np.mean(y)
    r_xy = np.sum((x - xmean)*(y - ymean)) / np.sqrt(np.sum((x - xmean)**2.)*np.sum((y - ymean)**2.))
    return r_xy

def Spearman_correlation_coefficient(x, y):
    # Spearman correlation coefficient is the Pearson correlation coefficient applied to the rank variables of x and y
    xsort, ysort = np.argsort(x), np.argsort(y)
    xranks, yranks = np.zeros(len(x)), np.zeros(len(y))
    xranks[xsort], yranks[ysort] = np.arange(len(x)), np.arange(len(y))
    rho_S = Pearson_correlation_coefficient(xranks, yranks)
    return rho_S



def radii_star_ratio(r, Rstar):
    # Sum of planet to stellar radii ratios for a system
    # Similar to "dynamical mass" mu as in GF2020
    assert all(r >= 0), 'Negative planet radii!'
    assert Rstar > 0, 'Negative stellar radii!'

    mu = np.sum(r/Rstar)
    return mu

def partitioning(x):
    # Partitioning of quantity 'x' for a system
    # E.g. "mass partitioning", Q if x=mass, as in GF2020
    # Factor of (N/(N-1)) should normalize Q to (0,1)
    assert all(x >= 0), 'Negative x values!'
    
    xnorm = x/np.sum(x)
    Q = (len(x)/(len(x)-1.))*disequilibrium(xnorm)
    return Q

def monotonicity_GF2020(x):
    # Monotonicity of quantity 'x' for a system
    # E.g. ordering in mass, "monotonicity" M if x=mass, as in GF2020
    rho_S = Spearman_correlation_coefficient(np.arange(len(x)), x)
    Q = partitioning(x)
    M = rho_S*(Q**(1./len(x)))
    return M

def gap_complexity_GF2020(P):
    assert len(P) >= 3, 'Need at least 3 planets in system to compute.'
    n = len(P)-1

    P = np.sort(P)
    Pmin, Pmax = np.min(P), np.max(P)
    Pratios = P[1:]/P[:-1]

    pnorm = np.log(Pratios)/np.log(Pmax/Pmin) # assuming natural log is what GF2020 used?
    Cmax = Cmax_table_GF2020(n) if n < 10 else Cmax_approx_GF2020(n)
    K = 1./Cmax

    C = LMC_complexity(K, pnorm)
    return C

def Cmax_table_GF2020(n):
    # n is the number of gaps, i.e. the number of planets minus 1
    Cmax_dict = {2: 0.106, 3: 0.212, 4: 0.291, 5: 0.350, 6: 0.398, 7: 0.437, 8: 0.469, 9: 0.497}
    return Cmax_dict[n]

def Cmax_approx_GF2020(n):
    Cmax = 0.262*np.log(0.766*n)
    return Cmax





# Useful functions for general purposes:

# Class to set midpoint of colormap on a log scale:
# Taken from: https://stackoverflow.com/questions/48625475/python-shifted-logarithmic-colorbar-white-color-offset-to-center
class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self,vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint=midpoint
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(np.log(value), x, y))
