# To import required modules:
import numpy as np
import os

import syssimpyplots.general as gen

path_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')





# Functions to load and analyze the Kepler observed catalog:

N_Kep = 86760 #86760 (Paper III) #88912 (Paper II) #79935 (Paper I) # number of Kepler targets satisfying our cuts to give our observed catalog # TODO: automate or read this from a file (stellar catalog?) instead of hardcoding

def load_Kepler_planets_cleaned():
    """
    Load a table of the Kepler objects of interest (KOIs) from a CSV file.

    Returns
    -------
    planets_cleaned : structured array
        A table with the properties of the KOIs.


    The table has the following columns:

    - `kepid`: The Kepler ID.
    - `KOI`: The KOI number.
    - `koi_disposition`: The disposition of the KOI.
    - `koi_pdisposition`: (TODO: TBD).
    - `koi_score`: The disposition score (between 0 and 1).
    - `P`: The orbital period (days).
    - `t_D`: The transit duration (hrs).
    - `depth`: The transit depth (ppm).
    - `Rp`: The planet radius (Earth radii).
    - `teff`: The stellar effective temperature (K).
    - `logg`: The log surface gravity of the star.
    - `Rstar`: The stellar radius (solar radii).
    - `Mstar`: The stellar mass (solar masses).

    """
    # q1_q17_dr25_gaia_fgk_HFR2021a_koi_cleaned.csv for Paper II
    # q1_q17_dr25_gaia_berger_fgk_H2020_koi_cleaned.csv for Paper III
    planets_cleaned = np.genfromtxt(os.path.join(path_data, 'q1_q17_dr25_gaia_berger_fgk_H2020_koi_cleaned.csv'), dtype={'names': ('kepid', 'KOI', 'koi_disposition', 'koi_pdisposition', 'koi_score', 'P', 't_D', 'depth', 'Rp', 'teff', 'logg', 'Rstar', 'Mstar'), 'formats': ('i8', 'S9', 'S15', 'S15', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',)}, delimiter=',') #orbit periods 'P' are in days; transit durations 't_D' are in hrs; transit depths 'depth' are in ppm; planetary radii 'Rp' are in Rearth; stellar radii 'Rstar' are in Rsolar
    planets_cleaned = planets_cleaned[1:]
    return planets_cleaned

def load_Kepler_stars_cleaned():
    """
    Load a table of Kepler target stars from a CSV file.

    Returns
    -------
    stars_cleaned : structured array
        A table with the properties of the Kepler target stars.


    The table has the following columns:

    - `kepid`: The Kepler ID.
    - `mass`: The stellar mass (solar masses).
    - `radius`: The stellar radius (solar radii).
    - `teff`: The stellar effective temperature (K).
    - `bp_rp`: The Gaia DR2 bp-rp color (mag).
    - `lum_val`: The luminosity (solar luminosities).
    - `e_bp_rp_interp`: The extinction in bp-rp color (mag) interpolated from a model/binning.
    - `e_bp_rp_true`: The extinction in bp-rp color (mag) as given in the Gaia DR2 catalog.
    - `rrmscdpp04p5`: The root-mean-square combined differential photometric precision (CDPP) for 4.5 hr durations (ppm).

    """
    # q1_q17_dr25_gaia_fgk_HFR2021a_cleaned.csv for Paper II
    # q1_q17_dr25_gaia_berger_fgk_H2020_cleaned.csv for Paper III
    #stars_cleaned = np.genfromtxt(os.path.join(path_data, 'q1_q17_dr25_gaia_fgk_HFR2021a_cleaned.csv'), dtype={'names': ('kepid', 'mass', 'radius', 'teff', 'bp_rp', 'e_bp_rp_interp', 'e_bp_rp_true', 'rrmscdpp04p5'), 'formats': ('i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')}, delimiter=',')
    stars_cleaned = np.genfromtxt(os.path.join(path_data, 'q1_q17_dr25_gaia_berger_fgk_H2020_cleaned.csv'), dtype={'names': ('kepid', 'mass', 'radius', 'teff', 'bp_rp', 'lum_val', 'e_bp_rp_interp', 'e_bp_rp_true', 'rrmscdpp04p5'), 'formats': ('i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')}, delimiter=',')
    stars_cleaned = stars_cleaned[1:]
    return stars_cleaned

def compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, Rstar_min=0., Rstar_max=10., Mstar_min=0., Mstar_max=10., teff_min=0., teff_max=1e4, bp_rp_min=-5., bp_rp_max=5., i_stars_custom=None, compute_ratios=gen.compute_ratios_adjacent):
    """
    Compute detailed summary statistics per system in the Kepler catalog.

    Parameters
    ----------
    P_min : float
        The minimum orbital period (days) to be included.
    P_max : float
        The maximum orbital period (days) to be included.
    radii_min : float
        The minimum planet radius (Earth radii) to be included.
    radii_max : float
        The maximum planet radius (Earth radii) to be included.
    Rstar_min : float, default=0.
        The minimum stellar radius (solar radii) to be included.
    Rstar_max : float, default=10.
        The maximum stellar radius (solar radii) to be included.
    Mstar_min : float, default=0.
        The minimum stellar mass (solar masses) to be included.
    Mstar_max : float, default=10.
        The maximum stellar mass (solar masses) to be included.
    teff_min : float, default=0.
        The minimum stellar effective temperature (K) to be included.
    teff_max : float, default=0.
        The maximum stellar effective temperature (K) to be included.
    bp_rp_min : float, default=-5.
        The minimum Gaia DR2 bp-rp color (mag) to be included.
    bp_rp_max : float, default=5.
        The maximum Gaia DR2 bp-rp color (mag) to be included.
    i_stars_custom : array[int], default=None
        An array of indices for the stars in the Kepler stellar catalog to be included.
    compute_ratios : func, default=compute_ratios_adjacent
        The function to use for computing ratios; can be either :py:func:`syssimpyplots.general.compute_ratios_adjacent` or :py:func:`syssimpyplots.general.compute_ratios_all`.

    Returns
    -------
    ssk_per_sys : dict
        A dictionary containing the planetary and stellar properties for each observed system (2-d and 1-d arrays).
    ssk : dict
        A dictionary containing the planetary and stellar properties of all observed planets (1-d arrays).


    The fields of ``ssk_per_sys`` and ``ssk`` are the same as those returned by :py:func:`syssimpyplots.load_sims.compute_summary_stats_from_cat_obs`.
    """
    planets_cleaned = load_Kepler_planets_cleaned()
    stars_cleaned = load_Kepler_stars_cleaned()

    if i_stars_custom is not None:
        stars_cleaned = stars_cleaned[i_stars_custom]

        i_pl_from_stars = []
        for i,kepid in enumerate(planets_cleaned['kepid']):
            i_stars_cleaned = np.where(stars_cleaned['kepid'] == kepid)[0]
            if len(i_stars_cleaned) == 1:
                i_pl_from_stars.append(i)
        i_pl_from_stars = np.array(i_pl_from_stars)
        planets_cleaned = planets_cleaned[i_pl_from_stars]

    # NOTE: these include duplicates for stars with multiple planets:
    n_pl = len(planets_cleaned)
    Rstar_per_pl = np.zeros(n_pl)
    Mstar_per_pl = np.zeros(n_pl)
    teff_per_pl = np.zeros(n_pl)
    bp_rp_per_pl = np.zeros(n_pl)
    e_bp_rp_per_pl = np.zeros(n_pl)
    cdpp4p5_per_pl = np.zeros(n_pl)
    for i,kepid in enumerate(planets_cleaned['kepid']):
        i_stars_cleaned = np.where(stars_cleaned['kepid'] == kepid)[0][0]
        Rstar_per_pl[i] = stars_cleaned['radius'][i_stars_cleaned]
        Mstar_per_pl[i] = stars_cleaned['mass'][i_stars_cleaned]
        teff_per_pl[i] = stars_cleaned['teff'][i_stars_cleaned]
        bp_rp_per_pl[i] = stars_cleaned['bp_rp'][i_stars_cleaned]
        e_bp_rp_per_pl[i] = stars_cleaned['e_bp_rp_interp'][i_stars_cleaned]
        cdpp4p5_per_pl[i] = stars_cleaned['rrmscdpp04p5'][i_stars_cleaned]

    planets_cleaned['Rstar'] = Rstar_per_pl
    planets_cleaned['Mstar'] = Mstar_per_pl
    planets_cleaned['teff'] = teff_per_pl

    # To make cuts in period, planet radii, and stellar properties:
    indices_keep = np.arange(len(planets_cleaned))[(planets_cleaned['P'] >= P_min) &
                                                   (planets_cleaned['P'] < P_max) &
                                                   (planets_cleaned['Rp'] >= radii_min) &
                                                   (planets_cleaned['Rp'] < radii_max) &
                                                   (Rstar_per_pl >= Rstar_min) &
                                                   (Rstar_per_pl < Rstar_max) &
                                                   (Mstar_per_pl >= Mstar_min) &
                                                   (Mstar_per_pl < Mstar_max) &
                                                   (teff_per_pl >= teff_min) &
                                                   (teff_per_pl < teff_max) &
                                                   (bp_rp_per_pl - e_bp_rp_per_pl >= bp_rp_min) &
                                                   (bp_rp_per_pl - e_bp_rp_per_pl < bp_rp_max)]

    planets_cleaned = planets_cleaned[indices_keep]



    Rstar_cand = Rstar_per_pl[indices_keep] #planets_cleaned['Rstar']
    Mstar_cand = Mstar_per_pl[indices_keep] #planets_cleaned['Mstar']
    teff_cand = teff_per_pl[indices_keep] #planets_cleaned['teff']
    bp_rp_cand = bp_rp_per_pl[indices_keep]
    e_bp_rp_cand = e_bp_rp_per_pl[indices_keep]
    cdpp4p5_cand = cdpp4p5_per_pl[indices_keep]

    # To keep arrays for the stellar properties only counting each star once:
    Rstar_sys_cand = []
    Mstar_sys_cand = []
    teff_sys_cand = []
    bp_rp_sys_cand = []
    e_bp_rp_sys_cand = []
    cdpp4p5_sys_cand = []



    #To compute the arrays of observables:
    KOI_systems = np.array([x[:6] for x in planets_cleaned['KOI']])
    checked_bools = np.zeros(len(planets_cleaned)) #0's denote KOI that were not checked yet; 1's denote already checked KOI

    P_sys_cand = [] # periods per system
    D_sys_cand = [] # depths per system
    tdur_sys_cand = [] # durations per system
    tdur_tcirc_sys_cand = [] # circular normalized durations per system
    radii_sys_cand = [] # planet radii per system
    Rm_sys_cand = [] # period ratios per system
    D_ratio_sys_cand = [] # depth ratios per system
    xi_sys_cand = [] # period-normalized duration ratios per system

    M_cand = [] # planet multiplicities of systems with planets
    Rm_cand = [] # period ratios
    D_ratio_cand = [] # depth ratios
    xi_cand = [] # period-normalized duration ratios
    xi_res_cand = [] # period-normalized duration ratios of planet pairs near resonance
    xi_res32_cand = [] # period-normalized duration ratios of planet pairs near 3:2 resonance
    xi_res21_cand = [] # period-normalized duration ratios of planet pairs near 2:1 resonance
    xi_nonres_cand = [] # period-normalized duration ratios of planet pairs not near resonance
    tdur_cand = planets_cleaned['t_D'] # durations (hrs)

    D_cand = planets_cleaned['depth']/(1e6) # depths (fraction)
    radii_cand = planets_cleaned['Rp'] # planet radii (Rearth)
    D_above_cand = [] # depths of planets above the photoevaporation boundary
    D_below_cand = [] # depths of planets below the photoevaporation boundary
    D_ratio_above_cand = [] # depth ratios of planet pairs above the photoevaporation boundary
    D_ratio_below_cand = [] # depth ratios of planet pairs below the photoevaporation boundary
    D_ratio_across_cand = [] # depth ratios of planet pairs across the photoevaporation boundary

    pad_extra = 100 #####

    for i in range(len(KOI_systems)):
        if checked_bools[i] == 0: #if the KOI has not been checked (included while looking at another planet in the same system)
            system_i = np.where(KOI_systems == KOI_systems[i])[0]
            checked_bools[system_i] = 1
            system_Rstar = planets_cleaned['Rstar'][system_i][0]
            system_Mstar = planets_cleaned['Mstar'][system_i][0]
            system_teff = planets_cleaned['teff'][system_i][0]
            system_bp_rp = bp_rp_cand[system_i][0]
            system_e_bp_rp = e_bp_rp_cand[system_i][0]
            system_cdpp4p5 = cdpp4p5_cand[system_i][0]

            #To get the planet properties in this system:
            system_P = planets_cleaned['P'][system_i] #periods of all the planets in this system
            system_tdur = planets_cleaned['t_D'][system_i] #transit durations of all the planets in this system
            system_D = planets_cleaned['depth'][system_i]/(1e6) #transit depths of all the planets in this system
            system_radii = planets_cleaned['Rp'][system_i] #radii of all the planets in this system
            system_sort_i = np.argsort(system_P) #indices that would sort the periods of the planets in this system
            system_P = system_P[system_sort_i] #periods of all the planets in this system, sorted
            system_tdur = system_tdur[system_sort_i] #transit durations of all the planets in this system, sorted by period
            system_D = system_D[system_sort_i] #transit depths of all the planets in this system, sorted by period
            system_radii = system_radii[system_sort_i] #radii of all the planets in this system, sorted by period

            system_tdur_tcirc = system_tdur/gen.tdur_circ(system_P, system_Mstar, system_Rstar)

            system_Rm = compute_ratios(system_P) #period ratios of all the adjacent planet pairs in this system
            system_D_ratio = compute_ratios(system_D) #transit depth ratios of all the adjacent planet pairs in this system
            system_xi = compute_ratios(system_tdur, inverse=True, avoid_div_zeros=True)*compute_ratios(system_P)**(1./3.) #period-normalized transit duration ratios of all the adjacent planet pairs in this system

            # To append the stellar properties counting the star only once:
            Rstar_sys_cand.append(system_Rstar)
            Mstar_sys_cand.append(system_Mstar)
            teff_sys_cand.append(system_teff)
            bp_rp_sys_cand.append(system_bp_rp)
            e_bp_rp_sys_cand.append(system_e_bp_rp)
            cdpp4p5_sys_cand.append(system_cdpp4p5)

            # To count the total number of planets in this system:
            M_cand.append(len(system_P))

            # To save arrays of planet properties per system:
            P_sys_cand.append(list(system_P) + [-1]*(pad_extra - len(system_P)))
            D_sys_cand.append(list(system_D) + [-1]*(pad_extra - len(system_D)))
            tdur_sys_cand.append(list(system_tdur) + [-1]*(pad_extra - len(system_tdur)))
            tdur_tcirc_sys_cand.append(list(system_tdur_tcirc) + [np.nan]*(pad_extra - len(system_tdur_tcirc)))
            radii_sys_cand.append(list(system_radii) + [-1]*(pad_extra - len(system_radii)))
            Rm_sys_cand.append(list(system_Rm) + [-1]*(pad_extra - len(system_Rm)))
            D_ratio_sys_cand.append(list(system_D_ratio) + [-1]*(pad_extra - len(system_D_ratio)))
            xi_sys_cand.append(list(system_xi) + [-1]*(pad_extra - len(system_xi)))

            # To separate into planet pairs near vs. not in resonance:
            mask_res_system = np.zeros(len(system_Rm), dtype=bool)
            mask_res32_system = np.zeros(len(system_Rm), dtype=bool)
            mask_res21_system = np.zeros(len(system_Rm), dtype=bool)

            for ratio in gen.res_ratios:
                mask_res_system[(system_Rm >= ratio) & (system_Rm <= ratio*(1.+gen.res_width))] = 1

            mask_res32_system[(system_Rm >= 1.5) & (system_Rm <= 1.5*(1.+gen.res_width))] = 1
            mask_res21_system[(system_Rm >= 2.) & (system_Rm <= 2.*(1.+gen.res_width))] = 1
            system_xi_res = system_xi[mask_res_system]
            system_xi_res32 = system_xi[mask_res32_system]
            system_xi_res21 = system_xi[mask_res21_system]
            system_xi_nonres = system_xi[~mask_res_system]
            #if sum(mask_res_system) > 0:
            #print(system_R[mask_res_system], system_xi_res)

            for Rm in system_Rm:
                Rm_cand.append(Rm)
            for D_ratio in system_D_ratio:
                D_ratio_cand.append(D_ratio)
            for xi in system_xi:
                xi_cand.append(xi)
            for xi in system_xi_res:
                xi_res_cand.append(xi)
            for xi in system_xi_res32:
                xi_res32_cand.append(xi)
            for xi in system_xi_res21:
                xi_res21_cand.append(xi)
            for xi in system_xi_nonres:
                xi_nonres_cand.append(xi)

            #To separate the planets in the system as above and below the boundary:
            system_above_bools = np.array([gen.photoevap_boundary_Carrera2018(system_radii[x], system_P[x]) for x in range(len(system_P))])
            #if len(system_above_bools) > 1:
            #print(system_above_bools)

            #To record the transit depths of the planets above and below the boundary:
            for j in range(len(system_D)):
                D_above_cand.append(system_D[j]) if system_above_bools[j] == 1 else D_below_cand.append(system_D[j])

            #To record the transit depth ratios of the planets above, below, and across the boundary:
            if compute_ratios == gen.compute_ratios_adjacent:
                for j in range(len(system_D_ratio)):
                    if system_above_bools[j] + system_above_bools[j+1] == 2: #both planets are above the boundary
                        D_ratio_above_cand.append(system_D_ratio[j])
                    elif system_above_bools[j] + system_above_bools[j+1] == 1: #one planet is above, the other is below the boundary
                        D_ratio_across_cand.append(system_D_ratio[j])
                    elif system_above_bools[j] + system_above_bools[j+1] == 0: #both planets are below the boundary
                        D_ratio_below_cand.append(system_D_ratio[j])
            elif compute_ratios == gen.compute_ratios_all:
                ncum_ratio_j = np.concatenate((np.array([0]),np.cumsum(np.arange(len(system_D))[::-1][:-2])))
                for j in range(len(system_D_ratio)):
                    for k in range(j+1,len(system_D)):
                        i_ratio = ncum_ratio_j[j] + k - 1 - j
                        if system_above_bools[j] + system_above_bools[k] == 2: #both planets are above the boundary
                            D_ratio_above_cand.append(system_D_ratio[i_ratio])
                        elif system_above_bools[j] + system_above_bools[k] == 1: #one planet is above, the other is below the boundary
                            D_ratio_across_cand.append(system_D_ratio[i_ratio])
                        elif system_above_bools[j] + system_above_bools[k] == 0: #both planets are below the boundary
                            D_ratio_below_cand.append(system_D_ratio[i_ratio])

    Rstar_sys_cand = np.array(Rstar_sys_cand)
    Mstar_sys_cand = np.array(Mstar_sys_cand)
    teff_sys_cand = np.array(teff_sys_cand)
    bp_rp_sys_cand = np.array(bp_rp_sys_cand)
    e_bp_rp_sys_cand = np.array(e_bp_rp_sys_cand)
    cdpp4p5_sys_cand = np.array(cdpp4p5_sys_cand)

    P_sys_cand = np.array(P_sys_cand)
    D_sys_cand = np.array(D_sys_cand)
    tdur_sys_cand = np.array(tdur_sys_cand)
    tdur_tcirc_sys_cand = np.array(tdur_tcirc_sys_cand)
    radii_sys_cand = np.array(radii_sys_cand)
    Rm_sys_cand = np.array(Rm_sys_cand)
    D_ratio_sys_cand = np.array(D_ratio_sys_cand)
    xi_sys_cand = np.array(xi_sys_cand)

    P_cand = planets_cleaned['P']
    M_cand = np.array(M_cand)
    Nmult_cand = np.array([np.sum(M_cand == x) for x in range(1,np.max(M_cand)+1)])
    Rm_cand = np.array(Rm_cand)
    D_ratio_cand = np.array(D_ratio_cand)
    xi_cand = np.array(xi_cand)
    xi_res_cand = np.array(xi_res_cand)
    xi_res32_cand = np.array(xi_res32_cand)
    xi_res21_cand = np.array(xi_res21_cand)
    xi_nonres_cand = np.array(xi_nonres_cand)

    D_above_cand = np.array(D_above_cand)
    D_below_cand = np.array(D_below_cand)
    D_ratio_above_cand = np.array(D_ratio_above_cand)
    D_ratio_below_cand = np.array(D_ratio_below_cand)
    D_ratio_across_cand = np.array(D_ratio_across_cand)

    tdur_tcirc_1_cand = tdur_tcirc_sys_cand[M_cand == 1, 0] # observed singles, 1d
    tdur_tcirc_2p_cand = tdur_tcirc_sys_cand[M_cand > 1] # observed multis, but 2d
    tdur_tcirc_2p_cand = tdur_tcirc_2p_cand[~np.isnan(tdur_tcirc_2p_cand)] # observed multis, 1d

    # Create dictionaries to hold summary stats ('ssk' stands for 'summary stats Kepler'):

    ssk_per_sys = {}
    # Stellar properties:
    ssk_per_sys['Rstar_obs'] = Rstar_sys_cand
    ssk_per_sys['Mstar_obs'] = Mstar_sys_cand
    ssk_per_sys['teff_obs'] = teff_sys_cand
    ssk_per_sys['bp_rp_obs'] = bp_rp_sys_cand
    ssk_per_sys['e_bp_rp_obs'] = e_bp_rp_sys_cand
    ssk_per_sys['cdpp4p5_obs'] = cdpp4p5_sys_cand
    # Planet properties:
    ssk_per_sys['Mtot_obs'] = M_cand
    ssk_per_sys['P_obs'] = P_sys_cand
    ssk_per_sys['D_obs'] = D_sys_cand
    ssk_per_sys['tdur_obs'] = tdur_sys_cand
    ssk_per_sys['tdur_tcirc_obs'] = tdur_tcirc_sys_cand
    ssk_per_sys['radii_obs'] = radii_sys_cand
    # Planet property ratios:
    ssk_per_sys['Rm_obs'] = Rm_sys_cand
    ssk_per_sys['D_ratio_obs'] = D_ratio_sys_cand
    ssk_per_sys['xi_obs'] = xi_sys_cand

    ssk = {}
    # Stellar properties (repeated to match number of planets):
    ssk['Rstar_obs'] = Rstar_cand
    ssk['Mstar_obs'] = Mstar_cand
    ssk['teff_obs'] = teff_cand
    ssk['bp_rp_obs'] = bp_rp_cand
    ssk['e_bp_rp_obs'] = e_bp_rp_cand
    ssk['cdpp4p5_obs'] = cdpp4p5_cand
    # Planet multiplicities:
    ssk['Nmult_obs'] = Nmult_cand
    # Planet properties:
    ssk['P_obs'] = P_cand
    ssk['D_obs'] = D_cand
    ssk['tdur_obs'] = tdur_cand
    ssk['tdur_tcirc_obs'] = tdur_tcirc_sys_cand[~np.isnan(tdur_tcirc_sys_cand)]
    ssk['tdur_tcirc_1_obs'] = tdur_tcirc_1_cand
    ssk['tdur_tcirc_2p_obs'] = tdur_tcirc_2p_cand
    ssk['radii_obs'] = radii_cand
    ssk['D_above_obs'] = D_above_cand
    ssk['D_below_obs'] = D_below_cand
    # Planet property ratios:
    ssk['Rm_obs'] = Rm_cand
    ssk['D_ratio_obs'] = D_ratio_cand
    ssk['xi_obs'] = xi_cand
    ssk['xi_res_obs'] = xi_res_cand
    ssk['xi_res32_obs'] = xi_res32_cand
    ssk['xi_res21_obs'] = xi_res21_cand
    ssk['xi_nonres_obs'] = xi_nonres_cand
    ssk['D_ratio_above_obs'] = D_ratio_above_cand
    ssk['D_ratio_below_obs'] = D_ratio_below_cand
    ssk['D_ratio_across_obs'] = D_ratio_across_cand

    # To compute some summary stats (system-level metrics) from GF2020:
    Nsys_obs = len(M_cand)
    assert Nsys_obs == len(radii_sys_cand) == len(P_sys_cand)

    radii_star_ratio = []
    radii_partitioning = []
    radii_monotonicity = []
    gap_complexity = []
    for i in range(Nsys_obs):
        P_sys = P_sys_cand[i][P_sys_cand[i] > 0]
        radii_sys = radii_sys_cand[i][P_sys_cand[i] > 0]

        radii_star_ratio.append(gen.radii_star_ratio(radii_sys, Rstar_sys_cand[i]))

        if len(radii_sys) >= 2:
            radii_partitioning.append(gen.partitioning(radii_sys))
            radii_monotonicity.append(gen.monotonicity_GF2020(radii_sys))
        if len(P_sys) >= 3:
            gap_complexity.append(gen.gap_complexity_GF2020(P_sys))
    ssk_per_sys['radii_star_ratio'] = np.array(radii_star_ratio)
    ssk_per_sys['radii_partitioning'] = np.array(radii_partitioning)
    ssk_per_sys['radii_monotonicity'] = np.array(radii_monotonicity)
    ssk_per_sys['gap_complexity'] = np.array(gap_complexity)

    return [ssk_per_sys, ssk]





# Functions for computing distances between the simulated and Kepler observed catalogs:

def CRPD_dist(En, On):
    """
    Compute the Cressie-Read Power Divergence (CRPD) statistic for observed planet multiplicity distributions.

    Warning
    -------
    Can potentially return negative values for extreme/edge cases!

    Parameters
    ----------
    En : array[int]
        The 'expected' (i.e. simulated) numbers of total systems with 1,2,3,... observed planets.
    On : array[int]
        The 'observed' (i.e. actual Kepler) numbers of total systems with 1,2,3,... observed planets.

    Returns
    -------
    rho : float
        The CRPD statistic.
    """
    n_max = max(len(En), len(On))
    En = np.array(list(En) + [0]*(n_max - len(En)))
    On = np.array(list(On) + [0]*(n_max - len(On)))

    E_array = En/float(np.sum(En)) # normalized numbers (fractions) of simulated systems with 1,2,3,... observed planets
    O_array = On/float(np.sum(On)) # normalized numbers (fractions) of actual Kepler systems with 1,2,3,... observed planets
    rho = 0.
    for i,E_i in enumerate(E_array):
        if En[i] != 0:
            rho += O_array[i]*((O_array[i]/E_array[i])**(2./3.) - 1.)
    rho = (9./5.)*rho

    return rho

def KS_dist_mult(x1, x2):
    """
    Compute the two-sample Kolmogorov-Smirnov (KS) distance between two discrete distributions taking on integer values.

    Parameters
    ----------
    x1, x2 : array[int]
        A sample of integers.

    Returns
    -------
    KS_dist : float
        The KS distance between the two distributions (i.e. the greatest distance between the cumulative distributions).
    KS_x : float
        The x-value that corresponds to the greatest distance between the two cumulative distributions.
    """
    x12_max = np.max((np.max(x1), np.max(x2))) # maximum value of x1 and x2
    x1_counts, x1_bins = np.histogram(x1, bins=x12_max, range=(0.5, x12_max+0.5))
    x2_counts, x2_bins = np.histogram(x2, bins=x12_max, range=(0.5, x12_max+0.5))
    pdf_diffs = x1_counts/np.float(len(x1)) - x2_counts/np.float(len(x2))
    cdf_diffs = np.cumsum(pdf_diffs)
    KS_dist = np.max(np.abs(cdf_diffs)) # K-S distance
    KS_x = np.arange(1, x12_max+1)[np.where(np.abs(cdf_diffs) == KS_dist)[0][0]] # x value where the CDF difference is the largest

    return KS_dist, KS_x

def KS_dist(x1, x2):
    """
    Compute the two-sample Kolmogorov-Smirnov (KS) distance between two continuous distributions (i.e. no repeat values).

    Parameters
    ----------
    x1, x2 : array[float]
        A sample of real values.

    Returns
    -------
    KS_dist : float
        The KS distance between the two distributions (i.e. the greatest distance between the cumulative distributions).
    KS_x : float
        The x-value that corresponds to the greatest distance between the two cumulative distributions.
    """
    x_all = np.concatenate((x1, x2)) # combined array
    i_all_sorted = np.argsort(x_all) # array of indices that would sort the combined array
    pdf_diffs = np.concatenate((np.ones(len(x1))/np.float(len(x1)), -np.ones(len(x2))/np.float(len(x2))))[i_all_sorted]
    cdf_diffs = np.cumsum(pdf_diffs)
    KS_dist = np.max(np.abs(cdf_diffs)) # K-S distance
    KS_x = x_all[i_all_sorted][np.where(np.abs(cdf_diffs) == KS_dist)[0][0]] # x value (a value in either x1 or x2) where the CDF difference is the largest

    return KS_dist, KS_x

def AD_dist(x1, x2):
    """
    Compute the two-sample Anderson-Darling (AD) distance between two continuous distributions.

    Implements Equation 1.2 of A. N. Pettitt (1976).

    Note
    ----
    Returns ``np.inf`` if there are not enough points (less than two) in either ``x1`` or ``x2`` for computing the AD distance.

    Parameters
    ----------
    x1, x2 : array[float]
        A sample of real values.

    Returns
    -------
    AD_dist : float
        The AD distance between the two distributions.
    """
    n, m = len(x1), len(x2)
    if n > 1 and m > 1:
        N = n + m
        x_all = np.concatenate((x1, x2)) # combined array
        i_all_sorted = np.argsort(x_all) # array of indices that would sort the combined array
        M_i_diffs = np.concatenate((np.ones(n), np.zeros(m)))[i_all_sorted]
        M_i_array = np.cumsum(M_i_diffs)[:-1] # array of M_i except for last element, i.e. from i=1 to i=N-1
        i_array = 1. + np.arange(N-1) # array of i from i=1 to i=N-1
        AD_dist = (1./(n*m))*np.sum(((M_i_array*N - n*i_array)**2.)/(i_array*(N - i_array))) # AD distance
    else:
        print('Not enough points to compute AD distance; returning inf.')
        AD_dist = np.inf

    return AD_dist

def AD_dist2(x1, x2):
    """
    Compute the two-sample Anderson-Darling (AD) distance between two continuous distributions.

    Implements Equation 3 of Scholz \& Stephens (1987). Tested to be equivalent to :py:func:`syssimpyplots.compare_kepler.AD_dist`.

    Note
    ----
    Returns ``np.inf`` if there are not enough points (less than two) in either ``x1`` or ``x2`` for computing the AD distance.

    Parameters
    ----------
    x1, x2 : array[float]
        A sample of real values.

    Returns
    -------
    AD_dist : float
        The AD distance between the two distributions.
    """
    n1, n2 = len(x1), len(x2)
    if n1 > 1 and n2 > 1:
        N = n1 + n2
        x_all = np.concatenate((x1, x2)) # combined array
        i_all_sorted = np.argsort(x_all) # array of indices that would sort the combined array

        M_1j_diffs = np.concatenate((np.ones(n1), np.zeros(n2)))[i_all_sorted]
        M_1j_array = np.cumsum(M_1j_diffs)[:-1] # array of M_1j except for last element, i.e. from j=1 to j=N-1
        M_2j_diffs = np.concatenate((np.zeros(n1), np.ones(n2)))[i_all_sorted]
        M_2j_array = np.cumsum(M_2j_diffs)[:-1] # array of M_2j except for last element, i.e. from j=1 to j=N-1
        j_array = 1. + np.arange(N-1) # array of j from j=1 to j=N-1

        AD_dist = (1./N)*((1./n1)*np.sum(((N*M_1j_array - n1*j_array)**2.)/(j_array*(N - j_array))) + (1./n2)*np.sum(((N*M_2j_array - n2*j_array)**2.)/(j_array*(N - j_array)))) # AD distance
    else:
        print('Not enough points to compute AD distance; returning inf.')
        AD_dist = np.inf

    return AD_dist

def AD_mod_dist(x1, x2):
    """
    Compute a modified version of the two-sample Anderson-Darling (AD) distance between two continuous distributions.

    Equivalent to the AD distance (implemented by :py:func:`syssimpyplots.compare_kepler.AD_dist` and :py:func:`syssimpyplots.compare_kepler.AD_dist2`) without the factor of 'n*m/N' in front of the integral, where 'n' and 'm' are the sample sizes (and 'N=n+m' is the combined sample size).

    Note
    ----
    Returns ``np.inf`` if there are not enough points (less than two) in either ``x1`` or ``x2`` for computing the AD distance.

    Parameters
    ----------
    x1, x2 : array[float]
        A sample of real values.

    Returns
    -------
    AD_dist : float
        The (modified) AD distance between the two distributions.
    """
    n, m = len(x1), len(x2)
    if n > 1 and m > 1:
        N = n + m
        x_all = np.concatenate((x1, x2)) # combined array
        i_all_sorted = np.argsort(x_all) # array of indices that would sort the combined array
        M_i_diffs = np.concatenate((np.ones(n), np.zeros(m)))[i_all_sorted]
        M_i_array = np.cumsum(M_i_diffs)[:-1] # array of M_i except for last element, i.e. from i=1 to i=N-1
        i_array = 1. + np.arange(N-1) # array of i from i=1 to i=N-1
        AD_dist = (N/((n*m)**2.))*np.sum(((M_i_array*N - n*i_array)**2.)/(i_array*(N - i_array))) # AD distance
    else:
        print('Not enough points to compute AD distance; returning inf.')
        AD_dist = np.inf

    return AD_dist

def load_split_stars_model_evaluations_and_weights(file_name):
    """
    Load a file containing the distances from many evaluations of the same model, and compute the weights for each distance term.

    Parameters
    ----------
    file_name : str
        The path/name of the file containing the distances of many model evaluations.

    Returns
    -------
    Nmult_evals : dict
        A dictionary containing an array of observed planet multiplicity distributions for each model evaluation, for each stellar sample (`all`, `bluer`, and `redder` fields).
    d_all_keys_evals : dict
        A dictionary containing an array of distance term names (strings) for each model evaluation, for each stellar sample.
    d_all_vals_evals : dict
        A dictionary containing an array of distances (corresponding to the distance term names) for each model evaluation, for each stellar sample.
    weights_all : dict
        A dictionary containing a dictionary for the weights corresponding to each distance term, for each stellar sample.


    Note
    ----
    The `bluer` and `redder` samples split the stellar sample into two equal sized samples of stars below and above the median Gaia DR2 bp-rp color, respectively.

    Warning
    -------
    Currently returns empty arrays in the `Nmult_evals` dictionary.
    """
    sample_names = ['all', 'bluer', 'redder']

    Nmult_max = 8
    Nmult_evals = {key: [] for key in sample_names}
    d_all_keys_evals = {key: [] for key in sample_names}
    d_all_vals_evals = {key: [] for key in sample_names}

    with open(file_name, 'r') as file:
        for line in file:
            for key in sample_names:
                n = len(key)
                if line[0:n+2] == '[%s]' % key:
                    #if line[n+3:n+3+6] == 'Counts':
                    #Nmult_str, counts_str = line[n+3+9:-2].split('][')
                    #Nmult = tuple([int(x) for x in Nmult_str.split(', ')])
                    #Nmult_evals[key].append(Nmult)

                    if line[n+3:n+3+11] == 'd_all_keys:':
                        d_all_keys = line[n+3+14:-3].split('", "')
                        d_all_keys_evals[key].append(d_all_keys)

                    elif line[n+3:n+3+11] == 'd_all_vals:':
                        d_all_vals_str = line[n+3+13:-2].split(', ')
                        d_all_vals = tuple([float(x) for x in d_all_vals_str])
                        d_all_vals_evals[key].append(d_all_vals)

    for key in sample_names:
        Nmult_evals[key] = np.array(Nmult_evals[key], dtype=[(str(n), 'i8') for n in range(1,Nmult_max+1)])
        d_all_keys_evals[key] = np.array(d_all_keys_evals[key])
        d_all_vals_evals[key] = np.array(d_all_vals_evals[key], dtype=[(dist_key, 'f8') for dist_key in d_all_keys_evals[key][0]])

    weights_all = {}
    for key in sample_names:
        dict_weights = {}
        for dist_key in d_all_vals_evals[key].dtype.names:
            dict_weights[dist_key] = 1./np.sqrt(np.mean(d_all_vals_evals[key][dist_key]**2.))
        weights_all[key] = dict_weights

    return Nmult_evals, d_all_keys_evals, d_all_vals_evals, weights_all

def load_split_stars_weights_only():
    """
    Compute the weights for each distance term.

    Wrapper to return just the weights from the function :py:func:`syssimpyplots.compare_kepler.load_split_stars_model_evaluations_and_weights`.
    """
    Nmult_evals, d_all_keys_evals, d_all_vals_evals, weights_all = load_split_stars_model_evaluations_and_weights(os.path.join(path_data, 'Clustered_P_R_split_stars_weights_ADmod_true_targs88912_evals100_all_pairs.txt'))
    return weights_all

def compute_total_weighted_dist(weights, dists, dists_w, dists_include=[]):
    """
    Compute the total weighted distance including a number of distance terms.

    Also prints out the individual distance terms, their weights, and their unweighted and weighted distances.

    Parameters
    ----------
    weights : dict
        The dictionary containing the weights for to each distance term.
    dists : dict
        The dictionary containing the individual distance terms.
    dists_w : dict
        The dictionary containing the individual weighted distance terms.
    dists_include : list[str], default=[]
        The list of distance terms (strings) to include in the sum.

    Returns
    -------
    tot_dist_w : float
        The total weighted distance of the included distance terms.
    """
    if len(dists_include) == 0:
        print('No distance terms to include.')

    print('#####')
    tot_dist_w = 0.
    for key in dists_include:
        tot_dist_w += dists_w[key]
        print('{:<30}: weight = {:<8}, dist = {:<8}, dist_w = {:<8}'.format(key, np.round(weights[key],2), np.round(dists[key],4), np.round(dists_w[key],2)))
    print('Total weighted distance = %s' % tot_dist_w)
    print('#####')

    return tot_dist_w

def compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights, dists_include, N_sim, cos_factor=1., AD_mod=True, print_dists=True):
    """
    Compute weighted and unweighted distances for a large collection of distance terms.

    Parameters
    ----------
    sss_per_sys : dict
        A dictionary of summary statistics per observed system in a simulated observed catalog.
    sss : dict
        A dictionary of summary statistics for all observed planets in a simulated observed catalog.
    ssk_per_sys : dict
        A dictionary of summary statistics per observed system in the Kepler catalog.
    ssk : dict
        A dictionary of summary statistics for observed all planets in the Kepler catalog.
    weights : dict
        A dictionary of the weights corresponding to each distance term.
    dists_include : list[str]
        A list of distance terms (strings) to be printed.
    N_sim : int
        The number of target stars (i.e. simulated systems) in the simulated catalog.
    cos_factor : float, default=1.
        The cosine of the maximum inclination angle (relative to the sky plane) drawn for the reference planes of the simulated systems (between 0 and 1).
    AD_mod : bool, default=True
        Whether to compute the modified AD distance (:py:func:`syssimpyplots.compare_kepler.AD_mod_dist`, if True) or the standard AD distance (:py:func:`syssimpyplots.compare_kepler.AD_dist`, if False).
    print_dists : bool, default=True
        Whether to print the distances corresponding to the terms in `dists_include`. If True, also prints the total numbers of observed planets and planet pairs in the simulated and Kepler catalogs.

    Returns
    -------
    dists : dict
        A dictionary containing all the various (unweighted) distance terms.
    dists_w : dict
        A dictionary containing all the various (weighted) distance terms.


    Note
    ----
    The distance terms computed and included in ``dists`` and ``dists_w`` are not limited to the terms in ``dists_include``.
    """
    # Create a dict to hold all distance terms:
    dists = {}

    dists['delta_f'] = np.abs(len(sss['P_obs'])/(float(N_sim)/cos_factor) - len(ssk['P_obs'])/float(N_Kep)) # absolute difference in the rates of observed planets per star

    Nmult_obs_sim_5plus = np.array(list(sss['Nmult_obs'][:4]) + [sum(sss['Nmult_obs'][4:])])
    Nmult_obs_Kep_5plus = np.array(list(ssk['Nmult_obs'][:4]) + [sum(ssk['Nmult_obs'][4:])])
    dists['mult_CRPD'] = CRPD_dist(Nmult_obs_sim_5plus, Nmult_obs_Kep_5plus)
    dists['mult_CRPD_r'] = CRPD_dist(Nmult_obs_Kep_5plus, Nmult_obs_sim_5plus)

    R_res32_sim, R_res32_Kep = np.float(sum((sss['Rm_obs'] >= 1.5) & (sss['Rm_obs'] <= 1.5*(1.+gen.res_width))))/np.float(len(sss['Rm_obs'])), np.float(sum((ssk['Rm_obs'] >= 1.5) & (ssk['Rm_obs'] <= 1.5*(1.+gen.res_width))))/np.float(len(ssk['Rm_obs'])) # fractions of planet pairs within 5% of 3:2 MMR, for simulated and Kepler data
    R_res21_sim, R_res21_Kep = np.float(sum((sss['Rm_obs'] >= 2.) & (sss['Rm_obs'] <= 2.*(1.+gen.res_width))))/np.float(len(sss['Rm_obs'])), np.float(sum((ssk['Rm_obs'] >= 2.) & (ssk['Rm_obs'] <= 2.*(1.+gen.res_width))))/np.float(len(ssk['Rm_obs'])) # fractions of planet pairs within 5% of 2:1 MMR, for simulated and Kepler data
    R_res32_diff = np.abs(R_res32_sim - R_res32_Kep) # difference in fractions of planet pairs close to 3:2 MMR between simulated and Kepler data
    R_res21_diff = np.abs(R_res21_sim - R_res21_Kep) # difference in fractions of planet pairs close to 2:1 MMR between simulated and Kepler data

    tdur_tcirc_1 = sss_per_sys['tdur_tcirc_obs'][sss_per_sys['Mtot_obs'] == 1, 0]
    tdur_tcirc_Kep_1 = ssk_per_sys['tdur_tcirc_obs'][ssk_per_sys['Mtot_obs'] == 1, 0]
    tdur_tcirc_2p = sss_per_sys['tdur_tcirc_obs'][sss_per_sys['Mtot_obs'] > 1]
    tdur_tcirc_2p = tdur_tcirc_2p[tdur_tcirc_2p > 0]
    tdur_tcirc_Kep_2p = ssk_per_sys['tdur_tcirc_obs'][ssk_per_sys['Mtot_obs'] > 1]
    tdur_tcirc_Kep_2p = tdur_tcirc_Kep_2p[tdur_tcirc_Kep_2p > 0]

    # KS distances:
    dists['periods_KS'] = KS_dist(sss['P_obs'], ssk['P_obs'])[0]
    dists['period_ratios_KS'] = KS_dist(sss['Rm_obs'], ssk['Rm_obs'])[0]
    dists['durations_KS'] = KS_dist(sss['tdur_obs'], ssk['tdur_obs'])[0]
    dists['durations_norm_circ_KS'] = KS_dist(sss['tdur_tcirc_obs'], ssk['tdur_tcirc_obs'])[0]
    dists['durations_norm_circ_singles_KS'] = KS_dist(tdur_tcirc_1, tdur_tcirc_Kep_1)[0]
    dists['durations_norm_circ_multis_KS'] = KS_dist(tdur_tcirc_2p, tdur_tcirc_Kep_2p)[0]
    dists['duration_ratios_KS'] = KS_dist(sss['xi_obs'], ssk['xi_obs'])[0]
    dists['duration_ratios_nonmmr_KS'] = KS_dist(sss['xi_nonres_obs'], ssk['xi_nonres_obs'])[0]
    dists['duration_ratios_mmr_KS'] = KS_dist(sss['xi_res_obs'], ssk['xi_res_obs'])[0]
    dists['depths_KS'] = KS_dist(sss['D_obs'], ssk['D_obs'])[0]
    dists['depths_above_KS'] = KS_dist(sss['D_above_obs'], ssk['D_above_obs'])[0]
    dists['depths_below_KS'] = KS_dist(sss['D_below_obs'], ssk['D_below_obs'])[0]
    dists['radius_ratios_KS'] = KS_dist(sss['D_ratio_obs'], ssk['D_ratio_obs'])[0]
    dists['radius_ratios_above_KS'] = KS_dist(sss['D_ratio_above_obs'], ssk['D_ratio_above_obs'])[0]
    dists['radius_ratios_below_KS'] = KS_dist(sss['D_ratio_below_obs'], ssk['D_ratio_below_obs'])[0]
    dists['radius_ratios_across_KS'] = KS_dist(sss['D_ratio_across_obs'], ssk['D_ratio_across_obs'])[0]
    dists['radii_partitioning_KS'] = KS_dist(sss_per_sys['radii_partitioning'], ssk_per_sys['radii_partitioning'])[0]
    dists['radii_monotonicity_KS'] = KS_dist(sss_per_sys['radii_monotonicity'], ssk_per_sys['radii_monotonicity'])[0]
    dists['gap_complexity_KS'] = KS_dist(sss_per_sys['gap_complexity'], ssk_per_sys['gap_complexity'])[0]

    # AD distances:
    AD_stat = AD_mod_dist if AD_mod else AD_dist

    dists['periods_AD'] = AD_stat(sss['P_obs'], ssk['P_obs'])
    dists['period_ratios_AD'] = AD_stat(sss['Rm_obs'], ssk['Rm_obs'])
    dists['durations_AD'] = AD_stat(sss['tdur_obs'], ssk['tdur_obs'])
    dists['durations_norm_circ_AD'] = AD_stat(sss['tdur_tcirc_obs'], ssk['tdur_tcirc_obs'])
    dists['durations_norm_circ_singles_AD'] = AD_stat(tdur_tcirc_1, tdur_tcirc_Kep_1)
    dists['durations_norm_circ_multis_AD'] = AD_stat(tdur_tcirc_2p, tdur_tcirc_Kep_2p)
    dists['duration_ratios_AD'] = AD_stat(sss['xi_obs'], ssk['xi_obs'])
    dists['duration_ratios_nonmmr_AD'] = AD_stat(sss['xi_nonres_obs'], ssk['xi_nonres_obs'])
    dists['duration_ratios_mmr_AD'] = AD_stat(sss['xi_res_obs'], ssk['xi_res_obs'])
    dists['depths_AD'] = AD_stat(sss['D_obs'], ssk['D_obs'])
    dists['depths_above_AD'] = AD_stat(sss['D_above_obs'], ssk['D_above_obs'])
    dists['depths_below_AD'] = AD_stat(sss['D_below_obs'], ssk['D_below_obs'])
    dists['radius_ratios_AD'] = AD_stat(sss['D_ratio_obs'], ssk['D_ratio_obs'])
    dists['radius_ratios_above_AD'] = AD_stat(sss['D_ratio_above_obs'], ssk['D_ratio_above_obs'])
    dists['radius_ratios_below_AD'] = AD_stat(sss['D_ratio_below_obs'], ssk['D_ratio_below_obs'])
    dists['radius_ratios_across_AD'] = AD_stat(sss['D_ratio_across_obs'], ssk['D_ratio_across_obs'])
    dists['radii_partitioning_AD'] = AD_stat(sss_per_sys['radii_partitioning'], ssk_per_sys['radii_partitioning'])
    dists['radii_monotonicity_AD'] = AD_stat(sss_per_sys['radii_monotonicity'], ssk_per_sys['radii_monotonicity'])
    dists['gap_complexity_AD'] = AD_stat(sss_per_sys['gap_complexity'], ssk_per_sys['gap_complexity'])

    # To create a dict for all the weighted distances:
    dists_w = {}
    for key in dists.keys():
        dists_w[key] = dists[key]*weights[key]

    # To compute the total weighted KS and AD distances assuming default terms:
    dists_default = ['periods', 'period_ratios', 'durations', 'duration_ratios_nonmmr', 'duration_ratios_mmr', 'depths', 'radius_ratios']
    tot_dist_w_KS = dists_w['delta_f'] + dists_w['mult_CRPD_r']
    tot_dist_w_AD = dists_w['delta_f'] + dists_w['mult_CRPD_r']
    for key in dists_default:
        tot_dist_w_KS += dists_w[key+'_KS']
        tot_dist_w_AD += dists_w[key+'_AD']
    dists_w['tot_dist_KS_default'] = tot_dist_w_KS
    dists_w['tot_dist_AD_default'] = tot_dist_w_AD



    if print_dists:
        print('(Planets Kepler obs, Planet pairs Kepler obs) = (%s, %s)' % (len(ssk['P_obs']), len(ssk['Rm_obs'])))
        print('(Planets obs, Planet pairs obs) = (%s, %s)' % (len(sss['P_obs']), len(sss['Rm_obs'])))
        tot_dist_w = compute_total_weighted_dist(weights, dists, dists_w, dists_include=dists_include)
        dists_w['tot_dist_w_include'] = tot_dist_w

    return dists, dists_w
