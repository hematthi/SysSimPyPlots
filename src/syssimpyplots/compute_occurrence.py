# To import required modules:
import numpy as np





# Functions for computing occurrence rates from the models:

def compute_occurrence_rate_and_fswp_in_period_radius_mass_bounds(sssp_per_sys, period_bounds=None, radius_bounds=None, mass_bounds=None, verbose=True):
    """
    Compute the occurrence rate (mean number of planets per star) and the fraction of stars with planets from a simulated catalog in the given period, radius, and/or mass bounds.
    
    Parameters
    ----------
    sssp_per_sys : dict
        The summary statistics dictionary for the simulated catalog, containing the planet properties per system.
    period_bounds : tuple, optional
        The period bounds (min,max) in days. If not provided, will assume the full simulation range.
    radius_bounds : tuple, optional
        The radius bounds (min,max) in Earth radii. If not provided, will assume the full simulation range.
    mass_bounds : tuple, optional
        The mass bounds (min,max) in Earth masses. If not provided, will assume the full simulation range.

    Returns
    -------
    occurrence : float
        The planet occurrence rate (mean number of planets per star).
    fswp : float
        The fraction of stars with planets.
    """
    assert 'N_sim' in sssp_per_sys, 'No number of systems found in simulated catalog!'
    assert 'P_all' in sssp_per_sys, 'No periods found in simulated catalog!'
    assert 'radii_all' in sssp_per_sys, 'No radii found in simulated catalog!'
    assert 'mass_all' in sssp_per_sys, 'No masses found in simulated catalog!'
    
    # Apply the period bounds, if given (or initialize 'bools_per_sys' as a 2d array of 'True' for each planet, if not):
    P_per_sys = sssp_per_sys['P_all']
    if period_bounds is not None:
        P_min, P_max = period_bounds
        assert P_min < P_max, 'Upper period bound must be larger than lower period bound.'
        bools_per_sys = (P_per_sys > P_min) & (P_per_sys < P_max)
    else:
        if verbose:
            P_min, P_max = sssp_per_sys.get('P_min', None), sssp_per_sys.get('P_max', None)
            print(f'No period bounds given; assuming the full simulation range of ({P_min},{P_max}) days.')
        bools_per_sys = (P_per_sys > 0)
    
    # Apply the radius bounds, if given:
    if radius_bounds is not None:
        radii_per_sys = sssp_per_sys['radii_all']
        radii_min, radii_max = radius_bounds
        assert radii_min < radii_max, 'Upper radius bound must be larger than lower radius bound.'
        bools_per_sys &= (radii_per_sys > radii_min) & (radii_per_sys < radii_max)
    else:
        if verbose:
            radii_min, radii_max = sssp_per_sys.get('radii_min', None), sssp_per_sys.get('radii_max', None)
            print(f'No radius bounds given; assuming the full simulation range of ({radii_min},{radii_max}) Earth radii.')
    
    # Apply the mass bounds, if given:
    if mass_bounds is not None:
        mass_per_sys = sssp_per_sys['mass_all']
        mass_min, mass_max = mass_bounds
        assert mass_min < mass_max, 'Upper mass bound must be larger than lower mass bound.'
        bools_per_sys &= (mass_per_sys > mass_min) & (mass_per_sys < mass_max)
    else:
        if verbose:
            mass_min, mass_max = sssp_per_sys.get('mass_min', None), sssp_per_sys.get('mass_max', None)
            print(f'No mass bounds given; assuming the full simulation range of ({mass_min},{mass_max}) Earth masses.')
    
    # Compute the occurrence rate and fraction of stars with planets:
    N_sim = sssp_per_sys['N_sim']
    counts_pl = np.sum(bools_per_sys)
    counts_swp = np.sum(np.any(bools_per_sys, axis=1))
    occurrence = counts_pl/N_sim
    fswp = counts_swp/N_sim
    
    assert 0 <= occurrence
    assert 0 <= fswp <= 1
    return occurrence, fswp

def compute_occurrence_rates_and_fswps_in_period_radius_mass_bounds_many_catalogs(sssp_per_sys_all, period_bounds=None, radius_bounds=None, mass_bounds=None):
    """
    Compute the occurrence rate (mean number of planets per star) and the fraction of stars with planets from many simulated catalogs in the given period, radius, and/or mass bounds.
    
    Calls the function :py:func:`syssimpyplots.compute_occurrence.compute_occurrence_rate_and_fswp_in_period_radius_mass_bounds` for each simulated catalog.
    
    Parameters
    ----------
    sssp_per_sys_all : list[dict]
        A list of dictionaries of summary statistics for each simulated catalog, containing the planet properties per system.
    period_bounds : tuple, optional
        The period bounds (min,max) in days. If not provided, will assume the full simulation range.
    radius_bounds : tuple, optional
        The radius bounds (min,max) in Earth radii. If not provided, will assume the full simulation range.
    mass_bounds : tuple, optional
        The mass bounds (min,max) in Earth masses. If not provided, will assume the full simulation range.

    Returns
    -------
    occurrence_all : array[float]
        The planet occurrence rates (mean number of planets per star) for each simulated catalog.
    fswp_all : array[float]
        The fractions of stars with planets for each simulated catalog.
    """
    N_catalogs = len(sssp_per_sys_all)
    occurrence_all = np.zeros(N_catalogs)
    fswp_all = np.zeros(N_catalogs)
    for i,sssp_per_sys in enumerate(sssp_per_sys_all):
        occurrence_all[i], fswp_all[i] = compute_occurrence_rate_and_fswp_in_period_radius_mass_bounds(sssp_per_sys, period_bounds=period_bounds, radius_bounds=radius_bounds, mass_bounds=mass_bounds, verbose=True if i==0 else False)
    
    # Also compute the quantiles for the occurrence rates and fswp over the catalogs:
    if N_catalogs < 10:
        print(f'Warning: too few catalogs ({N_catalogs}) provided; the computed uncertainties will be unreliable.')
    q = np.quantile(occurrence_all, [0.16,0.5,0.84])
    q_pm = np.diff(q)
    occ_str = '%s_{-%s}^{+%s}' % ('{:0.3f}'.format(q[1]), '{:0.3f}'.format(q_pm[0]), '{:0.3f}'.format(q_pm[1]))
    
    q = np.quantile(fswp_all, [0.16,0.5,0.84])
    q_pm = np.diff(q)
    fswp_str = '%s_{-%s}^{+%s}' % ('{:0.3f}'.format(q[1]), '{:0.3f}'.format(q_pm[0]), '{:0.3f}'.format(q_pm[1]))
    
    print('Occurrence rate =', occ_str)
    print('f_swp =', fswp_str)
    
    return occurrence_all, fswp_all
