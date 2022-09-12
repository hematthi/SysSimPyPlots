# To import required modules:
import numpy as np
from matplotlib.colors import LogNorm # for log color scales
from scipy.special import erf # error function, used in computing CDF of normal distribution





# Useful fundamental constants:

AU = 1.496*10.**13. # AU in cm
Msun = 1.989*10.**30. # Solar mass in kg
Rsun = 6.957*10.**10. # Solar radius in cm
Mearth = 5.972*10.**24. # Earth mass in kg
Rearth = 6.371*10.**8. # Earth radius in cm
Mjup = 1.898*10.**27. # Jupiter mass in kg
Rjup = 6.991*10.**9. # Jupiter radius in cm





# Miscellaneous definitions:

res_ratios, res_width = [2.0, 1.5, 4/3., 5/4.], 0.05 # NOTE: in the model, the near-resonant planets have period ratios between X and (1+w)*X where X = [2/1, 3/2, 4/3, 5/4] and w = 0.05





# Miscellaneous functions:

def a_from_P(P, Mstar):
    # Convert period (days) to semi-major axis (AU) assuming mass of planet m << Mstar (Msun)
    """
    Compute the semi-major axis using Kepler's third law.

    Note
    ----
    Assumes that the planet mass is negligible compared to the stellar mass.

    Parameters
    ----------
    P : float or array[floats]
        The orbital period (days).
    Mstar : float or array[floats]
        The stellar mass (solar masses).

    Returns
    -------
    a : float or array[floats]
        The semi-major axis (AU).
    """
    a = (P/365.25)**(2./3.)*(Mstar/1.0)**(1./3.)
    return a

def P_from_a(a, Mstar):
    """
    Compute the orbital period using Kepler's third law.

    Note
    ----
    Assumes that the planet mass is negligible compared to the stellar mass.

    Parameters
    ----------
    a : float or array[floats]
        The semi-major axis (AU).
    Mstar : float or array[floats]
        The stellar mass (solar masses).

    Returns
    -------
    P : float or array[floats]
        The orbital period (days).
    """
    P = 365.25*(a**(3./2.))*(Mstar/1.0)**(-1./2.)
    return P

def M_from_R_rho(R, rho=5.51):
    """
    Compute the planet mass from the radius and constant mean density.

    Parameters
    ----------
    R : float or array[floats]
        The planet radius (Earth radii).
    rho : float or array[floats], default=5.51
        The planet density (g/cm^3). Default value is that of Earth.

    Returns
    -------
    M : float or array[floats]
        The planet mass (Earth masses).
    """
    M_in_g = rho * (4.*np.pi/3.)*(R*Rearth)**3.
    return M_in_g/(Mearth*1000.)

def rho_from_M_R(M, R):
    """
    Compute the planet mean density (total mass divided by volume).

    Parameters
    ----------
    M : float or array[floats]
        The planet mass (Earth masses).
    R : float or array[floats]
        The planet radius (Earth radii).

    Returns
    -------
    rho : float or array[floats]
        The planet mean density (g/cm^3).
    """
    rho = (M*Mearth*1000.) / ((4.*np.pi/3.)*(R*Rearth)**3.)
    return rho

def tdur_circ(P, Mstar, Rstar):
    """
    Compute the transit duration assuming a circular orbit with zero impact parameter.

    Parameters
    ----------
    P : float or array[floats]
        The orbital period (days).
    Mstar : float or array[floats]
        The stellar mass (solar masses).
    Rstar : float or array[floats]
        The stellar radius (solar radii).

    Returns
    -------
    tdur : float or array[floats]
        The transit duration (hrs).
    """
    tdur = 24.*(Rstar*Rsun*P)/(np.pi*a_from_P(P,Mstar)*AU)
    return tdur

def AMD(mu, a, e, im):
    """
    Compute the AMD (angular momentum deficit) of a planet(s).

    Note
    ----
    Uses units of ``G*M_star = 1``.

    Parameters
    ----------
    mu : float or array[floats]
        The planet/star mass ratio.
    a : float or array[floats]
        The semi-major axis (AU).
    e : float or array[floats]
        The orbital eccentricity.
    im : float or array[floats]
        The inclination relative to the system invariable plane (radians).

    Returns
    -------
    amd_pl : float or array[floats]
        The AMD of the planet(s).
    """
    amd_pl = mu*np.sqrt(a) * (1. - np.sqrt(1. - e**2.)*np.cos(im))
    return amd_pl

def NAMD(m, a, e, im):
    """
    Compute the NAMD (normalized angular momentum deficit) of a planetary system.

    First defined in Chambers 2001, the NAMD of a system is the AMD divided by the angular momentum of the same system with circular and coplanar orbits.

    Parameters
    ----------
    m : float or array[floats]
        The planet masses (Earth masses).
    a : float or array[floats]
        The semi-major axes (AU).
    e : float or array[floats]
        The orbital eccentricities.
    im : float or array[floats]
        The inclinations relative to the system invariable plane (radians).

    Returns
    -------
    normed_AMD : float
        The NAMD of the system (unitless).
    """
    num = np.sum(m*np.sqrt(a) * (1. - np.sqrt(1. - e**2.)*np.cos(im)))
    den = np.sum(m*np.sqrt(a))
    normed_AMD = num/den
    return normed_AMD

def photoevap_boundary_Carrera2018(R, P):
    """
    Evaluate whether a planet is above or below the 'photo-evaporation' valley defined by Eq. 5 in Carrera et al. (2018).

    Parameters
    ----------
    R : float
        The planet radius (Earth radii).
    P : float
        The orbital period (days).

    Returns
    -------
    above_boundary : 1 or 0
        Whether the planet is above (1) or below (0) the boundary.
    """
    Rtrans = 2.6*P**(-0.1467)
    if R >= Rtrans:
        above_boundary = 1
    else:
        above_boundary = 0
    return above_boundary

def incl_mult_power_law_Zhu2018(k, sigma_5=0.8, alpha=-3.5):
    """
    Compute the Rayleigh scale of the mutual inclination distribution for a given planet multiplicity using the power-law relation.

    Note
    ----
    Default values for the power-law parameters ``sigma_5`` and ``alpha`` are set to best-fit values from Zhu et al. (2018).

    Parameters
    ----------
    k : float or array[floats]
        The planet multiplicity.
    sigma_5 : float or array[floats], default=0.8
        The normalization (Rayleigh scale of the mutual inclination distribution at ``k=5``) (degrees).
    alpha : float or array[floats], default=-3.5
        The power-law index.

    Returns
    -------
    sigma_k : float or array[floats]
        The Rayleigh scale (degrees) of the mutual inclination distribution for multiplicity ``k``.
    """
    sigma_k = sigma_5*(k/5.)**alpha
    return sigma_k

def cdf_normal(x, mu=0., std=1.):
    """
    Compute the cumulative distribution function (CDF; i.e. the integral between ``-inf`` and ``x`` of) the normal distribution at ``x`` given a mean and standard deviation.

    Note
    ----
    Can deal with array inputs for ``x``, ``mu``, and ``std`` as long as they are the same shape.

    Parameters
    ----------
    x : float or array[floats]
        The position to evaluate the CDF (between ``-inf`` and ``inf``).
    mu : float or array[floats], default=0.
        The mean of the normal distribution.
    std : float or array[floats], default=1.
        The standard deviation of the normal distribution.

    Returns
    -------
    cdf_x : float or array[floats]
        The CDF at ``x``.
    """
    cdf_x = 0.5*(1. + erf((x - mu)/(std*np.sqrt(2))))
    return cdf_x

def cdf_empirical(xdata, xeval):
    """
    Compute the empirical cumulative distribution function (CDF) at ``xeval`` given a sample.

    Note
    ----
    Is designed to deal with either scalar or array inputs of ``xeval``.

    Parameters
    ----------
    xdata : array
        The sample of data points.
    xeval : float or array[floats]
        The position to evaluate the CDF.

    Returns
    -------
    cdf_at_xeval : float or array[floats]
        The CDF at ``xeval``.
    """
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
    """
    Compute the intrinsic fraction of planets 'near' a period ratio with another planet for any period ratio in the given list of period ratios.

    'Near' is defined as a period ratio between ``pr`` and ``pratio*(1+pratio_width)`` for ``pratio`` in ``pratios``.

    Note
    ----
    Defaults to calculating the fraction of all planets near a mean-motion resonance (MMR), defined by ``res_ratios``.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary of summary statistics for a physical catalog of planets.
    pratios : list, default=res_ratios
        The list of period ratios to consider.
    pratio_width : float, default=res_width
        The fractional width to be considered 'near' a period ratio.

    Returns
    -------
    f_mmr : float
        The fraction of planets being 'near' at least one of the period ratios.
    """
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

def compute_ratios_adjacent(x, inverse=False, avoid_div_zeros=False):
    """
    Compute an array of the adjacent ratios (e.g. ``x[j+1]/x[j]``) of the terms in the input array ``x``.

    Parameters
    ----------
    x : array[float]
        The input values.
    inverse : bool, default=False
        Whether to take the inverse ratios (i.e. ``x[j]/x[j+1]`` instead of ``x[j+1]/x[j]``).
    avoid_div_zeros : bool, default=False
        Whether to avoid dividing by zeros. If True, will avoid computing ratios where the denominator is zero, and insert either 'inf' (when the numerator is non-zero) or 'nan' (when the numerator is also zero). If False, the computed ratios will still have the same results, but will produce ``RuntimeWarning`` messages.

    Returns
    -------
    ratios_adj : array[float]
        The ratios of adjacent pairs in ``x``.


    Note
    ----
    Will return an empty array if there are fewer than two elements in the input array.
    """
    if len(x) <= 1:
        return np.array([])

    if avoid_div_zeros and any(x==0):
        ratios_adj = np.zeros(len(x)-1)
        numers = x[0:-1] if inverse else x[1:] # the numerators
        denoms = x[1:] if inverse else x[0:-1] # the denominators
        bools_denoms_nonzero = denoms != 0 # booleans indicating the denominators that are non-zero
        bools_denoms_zero = ~bools_denoms_nonzero
        bools_numers_zero = numers == 0 # booleans indicating the numerators that are zero (for finding where we have 0/0)
        ratios_adj[bools_denoms_nonzero] = numers[bools_denoms_nonzero] / denoms[bools_denoms_nonzero]
        ratios_adj[bools_denoms_zero & ~bools_numers_zero] = np.sign(numers[bools_denoms_zero & ~bools_numers_zero])*np.inf # put +/-inf (depending on the sign of the numerator) when only denominator is zero
        ratios_adj[bools_denoms_zero & bools_numers_zero] = np.nan # put nans when both denominator and numerator are zero
        return ratios_adj

    # If get to this point, 'x' has at least two elements and none are zero
    ratios_adj = x[0:-1]/x[1:] if inverse else x[1:]/x[0:-1]
    return ratios_adj

def compute_ratios_all(x, inverse=False, avoid_div_zeros=False):
    """
    Compute an array of all the unique ratios (``x[j]/x[i]`` for all ``j > i``) of the terms in the input array ``x``.

    Parameters
    ----------
    x : array[float]
        The input values.
    inverse : bool, default=False
        Whether to take the inverse ratios (i.e. ``x[j]/x[i]`` for all ``j < i`` instead of for all ``j > i``).
    avoid_div_zeros : bool, default=False
        Whether to avoid dividing by zeros (WARNING: not implemented yet!).

    Returns
    -------
    ratios_all : array[float]
        The ratios of all pairs in ``x``.


    Note
    ----
    Will return an empty array if there are fewer than two elements in the input array.
    """
    if len(x) <= 1:
        return np.array([])

    ratios_all = list(x[0]/x[1:]) if inverse else list(x[1:]/x[0])
    for i in range(len(x)-2):
        ratios_all += list(x[i+1]/x[i+2:]) if inverse else list(x[i+2:]/x[i+1])
    return np.array(ratios_all)

def zeta1(pratios):
    """Compute the zeta statistic for each period ratio in ``pratios`` as defined in Fabrycky et al. (2014)."""
    return 3.*((1./(pratios - 1.)) - np.round(1./(pratios - 1.)))

def split_colors_per_cdpp_bin(stars_cleaned, nbins=10):
    # Compute a histogram of combined differential photometric precision (CDPP) values, and then split each bin by `bp-rp` color into a bluer half (smaller `bp-rp`) and a redder half (larger `bp-rp`).
    """
    Return the indices of the bluer and redder stars that split the stellar sample into two samples based on the histogram of combined differential photometric precision (CDPP) values.

    Note
    ----
    Uses log-uniform bins for the histograms.

    Parameters
    ----------
    stars_cleaned : structured array
        A table of stars, including columns for 4.5hr CDPP (``rrmscdpp04p5``) and Gaia DR2 bp-rp color (``bp_rp``).
    nbins : int, default=10
        The number of bins to use.

    Returns
    -------
    bins : array[floats]
        The bin edges.
    i_blue_per_bin : array[floats]
        The indices corresponding to the rows of bluer stars in each bin.
    i_red_per_bin : array[floats]
        The indices corresponding to the rows of redder stars in each bin.
    """
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
    """
    Evaluate the fraction of stars with planets (fswp) at a number of bp-rp colors using a linear relation.

    Note
    ----
    The fswp cannot be negative or greater than one.

    Parameters
    ----------
    bprp : array[floats]
        The bp-rp colors at which to evaluate the fswp.
    bprp_med : float
        The median bp-rp color (or some normalization point).
    fswp_med : float, default=0.5
        The fswp at ``bprp_med`` (normalization).
    slope : float, default=0.
        The slope of the linear relation.

    Returns
    -------
    fswp_bprp : array[floats]
        The fswp at each bp-rp color.
    """
    bprp = np.array(bprp)
    fswp_bprp = slope*(bprp - bprp_med) + fswp_med
    fswp_bprp[fswp_bprp < 0] = 0.
    fswp_bprp[fswp_bprp > 1] = 1.
    return fswp_bprp

def linear_alphaP_bprp(bprp, bprp_med, alphaP_med=0.5, slope=0.):
    """
    Evaluate the period power-law index at a number of bp-rp colors using a linear relation.

    Parameters
    ----------
    bprp : array[floats]
        The bp-rp colors at which to evaluate the fswp.
    bprp_med : float
        The median bp-rp color (or some normalization point).
    alphaP_med : float, default=0.5
        The period power-law index at ``bprp_med`` (normalization).
    slope : float, default=0.
        The slope of the linear relation.

    Returns
    -------
    alphaP_bprp : array[floats]
        The period power-law index at each bp-rp color.
    """
    bprp = np.array(bprp)
    alphaP_bprp = slope*(bprp - bprp_med) + alphaP_med
    return alphaP_bprp

def bin_Nmult(Nmult_obs, m_geq=5):
    """
    Bins the higher orders of the input multiplicity distribution.

    Parameters
    ----------
    Nmult_obs : array[ints]
        The multiplicity distribution for consecutive multiplicity orders (i.e. number of systems with 1,2,3,... planets).
    m_geq : int, default=5
        The multiplicity order at and above which to bin together.

    Returns
    -------
    Nmult_obs : array[ints]
        The multiplicity distribution with multiplicity orders greater than or equal to ``m_geq`` binned together. For the default ``m_geq=5``, this is the number of systems with 1,2,3,4,5+ planets.
    """
    Nmult_obs = list(Nmult_obs) + [0]*(m_geq-len(Nmult_obs)) # zero-pad first
    Nmult_obs[m_geq-1] = np.sum(Nmult_obs[m_geq-1:]) # bin everything greater than or equal to m_geq
    return np.array(Nmult_obs[:m_geq])





# Information theory quantities and metrics from Gilbert & Fabrycky 2020 (GF2020):

def Shannon_entropy(p):
    """
    Compute the Shannon entropy.

    Note
    ----
    Assumes base-natural log, although any choice is valid.

    Parameters
    ----------
    p : array[floats]
        An array of probabilities (values between 0 and 1).

    Returns
    -------
    H : float
        The Shannon entropy.
    """
    assert all(p >= 0), 'Negative probabilities!'
    assert all(p <= 1), 'Probabilities greater than 1!'

    H = -np.sum(p*np.log(p))
    return H

def disequilibrium(p):
    """
    Compute the disequilibrium of a system.

    Parameters
    ----------
    p : array[floats]
        An array of probabilities (values between 0 and 1).

    Returns
    -------
    D : float
        The disequilibrium.
    """
    assert all(p >= 0), 'Negative probabilities!'
    assert all(p <= 1), 'Probabilities greater than 1!'

    D = np.sum((p - (1./len(p)))**2.)
    return D

def LMC_complexity(K, p):
    """
    Compute the Lopez-Ruiz, Mancini, and Calbet (1995) complexity, which is the product of Shannon entropy and disequilibrium.

    Parameters
    ----------
    K : float
        The normalization constant.
    p : array[floats]
        An array of probabilities (values between 0 and 1).

    Returns
    -------
    C : float
        The LMC complexity.
    """
    assert K > 0

    H = Shannon_entropy(p)
    D = disequilibrium(p)
    C = K*H*D
    return C

def Pearson_correlation_coefficient(x, y):
    """Compute the Pearson correlation coefficient of two variables."""
    assert len(x) == len(y)

    xmean, ymean = np.mean(x), np.mean(y)
    r_xy = np.sum((x - xmean)*(y - ymean)) / np.sqrt(np.sum((x - xmean)**2.)*np.sum((y - ymean)**2.))
    return r_xy

def Spearman_correlation_coefficient(x, y):
    """
    Compute the Spearman correlation coefficient of two variables.

    This is the Pearson correlation coefficient applied to the ranks of the variables.
    """
    xsort, ysort = np.argsort(x), np.argsort(y)
    xranks, yranks = np.zeros(len(x)), np.zeros(len(y))
    xranks[xsort], yranks[ysort] = np.arange(len(x)), np.arange(len(y))
    rho_S = Pearson_correlation_coefficient(xranks, yranks)
    return rho_S



def radii_star_ratio(r, Rstar):
    """
    Compute the sum of planet/star radius ratios for a system.

    Parameters
    ----------
    r : array[floats]
        The planet radii.
    Rstar : float
        The stellar radius.

    Returns
    -------
    mu : float
        The sum of the planet/star radius ratios.
    """
    assert all(r >= 0), 'Negative planet radii!'
    assert Rstar > 0, 'Negative stellar radii!'

    mu = np.sum(r/Rstar)
    return mu

def partitioning(x):
    """
    Compute the 'partitioning' of a quantity ``x`` for the system.

    For example, if ``x`` is an array of planet masses, this is the 'mass partitioning' quantity.

    Parameters
    ----------
    x : array[floats]
        The partitions of the quantity.

    Returns
    -------
    Q : float
        The partitioning metric for the quantity (between 0 and 1).
    """
    assert all(x >= 0), 'Negative x values!'

    xnorm = x/np.sum(x)
    Q = (len(x)/(len(x)-1.))*disequilibrium(xnorm) # factor of (N/(N-1)) should normalize Q to (0,1)
    return Q

def monotonicity_GF2020(x):
    """
    Compute the 'monotonicity' (a measure of the degree of ordering) of a quantity ``x`` for the system, defined in Gilbert and Fabrycky (2020).

    For example, if ``x`` is an array of planet radii, this is the 'radius monotonicity' quantity.

    Parameters
    ----------
    x : array[floats]
        The array of quantities.

    Returns
    -------
    M : float
        The monotonicity of the quantity.
    """
    rho_S = Spearman_correlation_coefficient(np.arange(len(x)), x)
    Q = partitioning(x)
    M = rho_S*(Q**(1./len(x)))
    return M

def gap_complexity_GF2020(P):
    """
    Compute the 'gap complexity' metric of a planetary system, as defined in Gilbert and Fabrycky (2020).

    Parameters
    ----------
    P : array[floats]
        The array of orbital periods (days). Does not have to be sorted.

    Returns
    -------
    C : float
        The gap complexity of the system.
    """
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
    """
    Return the value of 'C_max', based on the number of gaps ``n`` (i.e. the number of planets minus 1) in the system.

    This is a normalization used in the gap complexity metric, with the values computed by Gilbert and Fabrycky (2020).

    Note
    ----
    The table of 'C_max' only goes up to ``n=9``; for greater numbers of gaps, use the approximation function :py:func:`syssimpyplots.general.Cmax_approx_GF2020`.
    """
    Cmax_dict = {2: 0.106, 3: 0.212, 4: 0.291, 5: 0.350, 6: 0.398, 7: 0.437, 8: 0.469, 9: 0.497}
    return Cmax_dict[n]

def Cmax_approx_GF2020(n):
    """Compute an approximation for the value of 'C_max' based on the number of gaps, defined in Gilbert and Fabrycky (2020)."""
    Cmax = 0.262*np.log(0.766*n)
    return Cmax





# Useful functions for general purposes:

class MidPointLogNorm(LogNorm):
    """
    Set the midpoint of a colormap on a log scale, taken from: https://stackoverflow.com/questions/48625475/python-shifted-logarithmic-colorbar-white-color-offset-to-center (in a post by ImportanceOfBeingErnest)
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self,vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint=midpoint
    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.array(np.interp(np.log(value), x, y), mask=result.mask, copy=False)
