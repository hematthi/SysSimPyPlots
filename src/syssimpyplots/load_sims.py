# To import required modules:
import numpy as np
import pandas as pd
import time

import syssimpyplots.general as gen
import syssimpyplots.compare_kepler as ckep




# Fixed parameters:

param_symbols = {
    "num_targets_sim_pass_one": r'$N_{\rm stars,sim}$',
    "max_incl_sys": r'$i_{\rm ref,max}$',
    "f_stars_with_planets_attempted": r'$f_{\rm swpa}$',
    "f_stars_with_planets_attempted_at_med_color": r'$f_{\rm swpa,med}$',
    "f_stars_with_planets_attempted_color_slope": r'$d(f_{\rm swpa})/d(b_p-r_p)$',
    "log_rate_clusters": r'$\lambda_c$',
    "max_clusters_in_sys": r'$N_{c,\rm max}$',
    "log_rate_planets_per_cluster": r'$\lambda_p$',
    "max_planets_in_clusters": r'$N_{p,\rm max}$',
    "power_law_P": r'$\alpha_P$',
    "power_law_P_at_med_color": r'$\alpha_{P,\rm med}$',
    "power_law_P_color_slope": r'$d\alpha_P/d(b_p-r_p)$',
    "min_period": r'$P_{\rm min}$',
    "max_period": r'$P_{\rm max}$',
    "power_law_r1": r'$\alpha_{R1}$',
    "power_law_r2": r'$\alpha_{R2}$',
    "break_radius (R_earth)": r'$R_{p,\rm break}$ $(R_\oplus)$',
    "min_radius (R_earth)": r'$R_{p,\rm min}$ $(R_\oplus)$',
    "max_radius (R_earth)": r'$R_{p,\rm max}$ $(R_\oplus)$',
    "f_high_incl": r'$f_{\sigma_{i,\rm high}}$',
    "sigma_incl": r'$\sigma_{i,\rm high}$',
    "sigma_incl_near_mmr": r'$\sigma_{i,\rm low}$',
    "sigma_hk": r'$\sigma_e$',
    "sigma_hk_at_med_color": r'$\sigma_e(0.95)$',
    "sigma_hk_color_slope": r'$d(\sigma_e)/d(b_p-r_p)$',
    "num_mutual_hill_radii": r'$\Delta_c$',
    "sigma_log_radius_in_cluster": r'$\sigma_R$',
    "sigma_logperiod_per_pl_in_cluster": r'$\sigma_N$',
} # dictionary of the symbols and names for all the model parameters; NOTE: although the params are named log rate of clusters and planets per cluster, we use the symbols and values for the rates

def read_targets_period_radius_bounds(file_name):
    """
    Read the number of simulated targets and bounds for the planet periods and radii from a file.

    Parameters
    ----------
    file_name : str
        The path/name of the file containing a header with simulation parameters.

    Returns
    -------
    N_sim : int
        The number of simulated systems.
    cos_factor : float
        The cosine of the maximum inclination angle (relative to the sky plane) drawn for the reference planes of the simulated systems (between 0 and 1).
    P_min : float
        The minimum orbital period (days).
    P_max : float
        The maximum orbital period (days).
    radii_min : float
        The minimum planet radius (Earth radii).
    radii_max : float
        The maximum planet radius (Earth radii).
    """
    with open(file_name, 'r') as file: #open(loadfiles_directory + 'observed_catalog_planets%s.txt' % run_number, 'r')
        for line in file:
            if line[:26] == '# num_targets_sim_pass_one':
                N_sim = int(line[28:])
            elif line[:14] == '# max_incl_sys':
                max_incl_sys = float(line[16:])
                cos_factor = np.cos(max_incl_sys*np.pi/180.)
            elif line[:12] == '# min_period':
                P_min = float(line[14:])
            elif line[:12] == '# max_period':
                P_max = float(line[14:])
            elif line[:12] == '# min_radius':
                radii_min = float(line[24:])
            elif line[:12] == '# max_radius':
                radii_max = float(line[24:])

    return N_sim, cos_factor, P_min, P_max, radii_min, radii_max

def read_sim_params(file_name):
    """
    Read the simulation parameters from a file and output them in a dictionary.

    Parameters
    ----------
    file_name : str
        The path/name of the file containing a header with simulation parameters.

    Returns
    -------
    param_vals : dict
        A dictionary containing the simulation parameters.


    The full list of possible parameters is defined in ``param_symbols`` (also exported by this module).
    """
    param_vals = {}
    with open(file_name, 'r') as file:
        for line in file:
            for param in param_symbols:
                chars = len(param)
                if line[:3+chars] == '# ' + param + ':':
                    if param[:3] == 'log':
                        param_vals[param] = np.exp(float(line[4+chars:]))
                    elif (param[:11] == 'num_targets') or (param[:11] == 'mr_max_mass'):
                        param_vals[param] = int(line[4+chars:])
                    else:
                        param_vals[param] = float(line[4+chars:])

    return param_vals





# Functions to load and analyze simulated physical catalogs:

def load_cat_phys(file_name):
    """
    Load a table with all the planets in a simulated physical catalog.

    Parameters
    ----------
    file_name : str
        The path/name of the file for the physical catalog (should end with 'physical_catalog.csv').

    Returns
    -------
    cat_phys : structured array
        A table with the physical properties of all the planets.


    The table has the following columns:

    - `target_id`: The index of the star in the simulation (e.g. 1 for the first star) which the planet orbits.
    - `star_id`: The index of the star based on where it is in the input stellar catalog.
    - `planet_mass`: The planet mass (solar masses).
    - `planet_radius`: The planet radius (solar radii).
    - `clusterid`: A cluster identifier.
    - `period`: The orbital period (days).
    - `ecc`: The orbital eccentricity.
    - `incl`: The orbital inclination (radians) relative to the sky plane.
    - `omega`: The argument of periapsis (radians) relative to the sky plane.
    - `asc_node`: The argument of ascending node (radians) relative to the sky plane.
    - `mean_anom`: The mean anomaly (radians) relative to the sky plane.
    - `incl_invariable`: The orbital inclination (radians) relative to the system invariable plane.
    - `asc_node_invariable`: The argument of ascending node (radians) relative to the system invariable plane.
    - `star_mass`: The stellar mass (solar masses).
    - `star_radius`: The stellar radius (solar radii).

    """
    start = time.time()
    #cat_phys = pd.read_csv(file_name, comment='#', dtype={'target_id': 'Int64', 'star_id': 'Int64', 'planet_mass': 'f8', 'planet_radius': 'f8', 'clusterid': 'Int64', 'period': 'f8', 'ecc': 'f8', 'incl_mut': 'f8', 'incl': 'f8', 'star_mass': 'f8', 'star_radius': 'f8'}) # faster than np.genfromtxt, BUT indexing the pandas DataFrame is much slower later!
    #'''
    with open(file_name, 'r') as file:
        lines = (line for line in file if not line.startswith('#'))
        #cat_phys = np.genfromtxt(lines, names=True, dtype=('i4', 'i4', 'f8', 'f8', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8'), delimiter=',')
        cat_phys = np.genfromtxt(lines, names=True, dtype=('i4', 'i4', 'f8', 'f8', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'), delimiter=',') # faster than np.loadtxt
    #'''
    stop = time.time()
    print('Time to load (table of planets): %s s' % (stop - start))

    return cat_phys

def load_star_phys(file_name):
    """
    Load a table of only the stars with planets in a simulated physical catalog.

    Parameters
    ----------
    file_name : str
        The path/name of the file for the stellar physical catalog (should end with 'physical_catalog_stars.csv').

    Returns
    -------
    star_phys : structured array
        A table with basic properties of the planet-hosting stars.


    The table has the following columns:

    - `target_id`: The index of the star in the simulation (e.g. 1 for the first star) which the planet orbits.
    - `star_id`: The index of the star based on where it is in the input stellar catalog.
    - `star_mass`: The stellar mass (solar masses).
    - `star_radius`: The stellar radius (solar radii).
    - `num_planets`: The number of planets in the system.

    """
    start = time.time()
    #star_phys = pd.read_csv(file_name, comment='#', dtype={'target_id': 'Int64', 'star_id': 'Int64', 'star_mass': 'f8', 'star_radius': 'f8', 'num_planets': 'Int64'}) # faster than np.genfromtxt, BUT indexing the pandas DataFrame is much slower later!
    #'''
    with open(file_name, 'r') as file:
        lines = (line for line in file if not line.startswith('#'))
        star_phys = np.genfromtxt(lines, names=True, dtype=('i4', 'i4', 'f8', 'f8', 'i4'), delimiter=',') # faster than np.loadtxt
    #'''
    stop = time.time()
    print('Time to load (table of stars with planets): %s s' % (stop - start))

    return star_phys

def load_planets_stars_phys_separate(file_name_path, run_number):
    """
    Load individual files with the properties of all the planets and stars in a simulated physical catalog.

    Note
    ----
    Faster than :py:func:`syssimpyplots.load_sims.load_cat_phys` for large catalogs, but returns individual lists instead of a single table. Each list is ordered in the same way so the planet properties can be matched to each other.

    Parameters
    ----------
    file_name_path : str
        The path to the physical catalog.
    run_number : str
        The run number appended to the file names for the physical catalog.

    Returns
    -------
    clusterids_per_sys : list[list]
        The cluster id's of each system.
    P_per_sys : list[list]
        The orbital periods (days) of each system.
    radii_per_sys : list[list]
        The planet radii (solar radii) of each system.
    mass_per_sys : list[list]
        The planet masses (solar masses) of each system.
    e_per_sys : list[list]
        The orbital eccentricities of each system.
    inclmut_per_sys : list[list]
        The orbital inclinations (radians) relative to the system invariable plane of each system.
    incl_per_sys : list[list]
        The orbital inclinations (radians) relative to the sky plane of each system.
    Mstar_all : array[float]
        The stellar mass (solar masses) of each system.
    Rstar_all : array[float]
        The stellar radius (solar radii) of each system.
    """
    start = time.time()

    clusterids_per_sys = [] # list to be filled with lists of all the cluster id's per system
    try:
        with open(file_name_path + 'clusterids_all%s.out' % run_number, 'r') as file:
            for line in file:
                if line[0] != '#':
                    line = line[1:-2].split(', ')
                    clusterids_sys = [int(i) for i in line]
                    clusterids_per_sys.append(clusterids_sys)
    except:
        print('No file with cluster ids found.')

    P_per_sys = [] # list to be filled with lists of all periods per system (days)
    try:
        with open(file_name_path + 'periods_all%s.out' % run_number, 'r') as file:
            for line in file:
                if line[0] != '#':
                    line = line[1:-2].split(', ')
                    P_sys = [float(i) for i in line]
                    P_per_sys.append(P_sys)
                    #print(P_sys)
    except:
        print('No file with periods found.')

    radii_per_sys = [] # list to be filled with lists of all planet radii per system (solar radii)
    try:
        with open(file_name_path + 'radii_all%s.out' % run_number, 'r') as file:
            for line in file:
                if line[0] != '#':
                    line = line[1:-2].split(', ')
                    radii_sys = [float(i) for i in line]
                    radii_per_sys.append(radii_sys)
                    #print(radii_sys)
    except:
        print('No file with planet radii found.')

    mass_per_sys = [] # list to be filled with lists of all planet radii per system (solar masses)
    try:
        with open(file_name_path + 'masses_all%s.out' % run_number, 'r') as file:
            for line in file:
                if line[0] != '#':
                    line = line[1:-2].split(', ')
                    mass_sys = [float(i) for i in line]
                    mass_per_sys.append(mass_sys)
                    #print(mass_sys)
    except:
        print('No file with planet masses found.')

    e_per_sys = [] # list to be filled with lists of all eccentricities per system
    try:
        with open(file_name_path + 'eccentricities_all%s.out' % run_number, 'r') as file:
            for line in file:
                if line[0] != '#':
                    line = line[1:-2].split(', ')
                    e_sys = [float(i) for i in line]
                    e_per_sys.append(e_sys)
                    #print(e_sys)
    except:
        print('No file with eccentricities found.')

    inclmut_per_sys = [] # list to be filled with lists of all mutual inclinations (rad) per system
    try:
        with open(file_name_path + 'mutualinclinations_all%s.out' % run_number, 'r') as file:
            for line in file:
                if line[0] != '#':
                    line = line[1:-2].split(', ')
                    inclmut_sys = [float(i) for i in line]
                    inclmut_per_sys.append(inclmut_sys)
                    #print(inclmut_sys)
    except:
        print('No file with mutual inclinations found.')

    incl_per_sys = [] # list to be filled with lists of all sky inclinations (rad) per system
    try:
        with open(file_name_path + 'inclinations_all%s.out' % run_number, 'r') as file:
            for line in file:
                if line[0] != '#':
                    line = line[1:-2].split(', ')
                    incl_sys = [float(i) for i in line]
                    incl_per_sys.append(incl_sys)
        #print(incl_sys)
    except:
        print('No file with sky inclinations found.')

    try:
        Mstar_all = np.loadtxt(file_name_path + 'stellar_masses_with_planets%s.out' % run_number) # array of stellar masses of all the systems with a planetary system, in solar masses
    except:
        Mstar_all = []
        print('No file with stellar masses found.')

    try:
        Rstar_all = np.loadtxt(file_name_path + 'stellar_radii_with_planets%s.out' % run_number) # array of stellar radii of all the systems with a planetary system, in solar radii
    except:
        Rstar_all = []
        print('No file with stellar radii found.')

    stop = time.time()
    print('Time to load (separate files with planets and stars): %s s' % (stop - start))

    return clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all

def compute_basic_summary_stats_per_sys_cat_phys(clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all):
    """
    Compute the basic summary statistics per system in a physical catalog.

    Note
    ----
    The input parameters should be returned by the function :py:func:`syssimpyplots.load_sims.load_planets_stars_phys_separate` and requires the individual lists to be ordered in the same way.

    Parameters
    ----------
    clusterids_per_sys : list[list]
        A list of lists with the cluster id's for each system.
    P_per_sys : list[list]
        A list of lists with the orbital periods (days) for each system.
    radii_per_sys : list[list]
        A list of lists with the planet radii (solar radii) for each system.
    mass_per_sys : list[list]
        A list of lists with the planet masses (solar masses) for each system.
    e_per_sys : list[list]
        A list of lists with the orbital eccentricities for each system.
    inclmut_per_sys : list[list]
        A list of lists with the orbital inclinations (radians) relative to the system invariable plane for each system.
    incl_per_sys : list[list]
        A list of lists with the orbital inclinations (radians) relative to the sky plane for each system.
    Mstar_all : array[float]
        The stellar mass (solar masses) for each system.
    Rstar_all : array[float]
        The stellar radius (solar radii) for each system.

    Returns
    -------
    sssp_per_sys_basic : dict
        A dictionary containing planetary and stellar properties for each system.


    The output is a dictionary containing the following fields:

    - `Mmax`: The maximum planet multiplicity of any system.
    - `Mtot_all`: The planet multiplicity of each system (1-d array).
    - `clustertot_all`: The number of planet clusters in each system (1-d array).
    - `pl_per_cluster_all`: The number of planets in each cluster (1-d array).
    - `P_all`: The orbital periods (days) of each system (2-d array).
    - `clusterids_all`: The cluster id's of each system (2-d array).
    - `e_all`: The orbital eccentricities of each system (2-d array).
    - `inclmut_all`: The orbital inclinations (radians) relative to system invariable plane of each system (2-d array).
    - `incl_all`: The orbital inclinations (radians) relative to the sky plane of each system (2-d array).
    - `radii_all`: The planet radii (Earth radii) of each system (2-d array).
    - `mass_all`: The planet masses (Earth masses) of each system (2-d array).
    - `Mstar_all`: The stellar mass (solar masses) of each system (1-d array).
    - `Rstar_all`: The stellar radius (solar radii) of each system (1-d array).
    - `mu_all`: The planet/star mass ratios of each system (2-d array).
    - `a_all`: The semi-major axes (AU) of each system (2-d array).
    - `AMD_all`: The AMDs (units of G*Mstar=1) of each system (2-d array).
    - `AMD_tot_all`: The total AMD (units of G*Mstar=1) of each system (1-d array).

    Warning
    -------
    For the 2-d arrays, each row is padded with zeros (or negative ones), since different systems have different numbers of planets.
    """
    assert len(clusterids_per_sys) != 0

    clusterids_all = []
    P_all = []
    e_all = []
    inclmut_all = []
    incl_all = []
    radii_all = []
    mass_all = []

    Pmin = 0. # set a minimum period (days), discarding planets less than this period

    Mmax = np.max([len(x) for x in clusterids_per_sys]) # maximum planet multiplicity
    Mtot_all = [] # 1d, len = number of systems
    clustertot_all = [] # 1d, len = number of systems
    pl_per_cluster_all = [] # 1d, len = number of clusters

    start = time.time()
    for i in range(len(clusterids_per_sys)):
        # Clusters and planets per cluster:
        Mtot_all.append(len(clusterids_per_sys[i]))
        clusterids_unique = list(set(clusterids_per_sys[i]))
        clustertot_all.append(len(clusterids_unique))
        for c in clusterids_unique:
            pl_per_cluster_all.append(np.sum(np.array(clusterids_per_sys[i]) == c))

        if len(P_per_sys) != 0:
            i_sorted = np.argsort(P_per_sys[i])
            P_sorted = np.array(P_per_sys[i])[i_sorted]
            P_sorted_cut = P_sorted[P_sorted > Pmin]
            cids_sorted_cut = np.array(clusterids_per_sys[i])[i_sorted][P_sorted > Pmin]
            e_sorted_cut = np.array(e_per_sys[i])[i_sorted][P_sorted > Pmin]
            incl_sorted_cut = np.array(incl_per_sys[i])[i_sorted][P_sorted > Pmin] if len(incl_per_sys) > 0 else []
            radii_sorted_cut = np.array(radii_per_sys[i])[i_sorted][P_sorted > Pmin]
            mass_sorted_cut = np.array(mass_per_sys[i])[i_sorted][P_sorted > Pmin]

            P_sys = list(P_sorted_cut) + [0]*(Mmax - len(P_sorted_cut))
            clusterids_sys = list(cids_sorted_cut) + [0]*(Mmax - len(cids_sorted_cut))
            e_sys = list(e_sorted_cut) + [0]*(Mmax - len(e_sorted_cut))
            incl_sys = list(incl_sorted_cut) + [0]*(Mmax - len(incl_sorted_cut))
            radii_sys = list(radii_sorted_cut) + [0]*(Mmax - len(radii_sorted_cut))
            mass_sys = list(mass_sorted_cut) + [0]*(Mmax - len(mass_sorted_cut))

            P_all.append(P_sys)
            clusterids_all.append(clusterids_sys)
            e_all.append(e_sys)
            incl_all.append(incl_sys)
            radii_all.append(radii_sys)
            mass_all.append(mass_sys)

            # Mutual inclinations:
            if len(inclmut_per_sys) != 0:
                inclmut_sorted_cut = np.array(inclmut_per_sys[i])[i_sorted][P_sorted > Pmin]
                inclmut_sys = list(inclmut_sorted_cut) + [0]*(Mmax - len(inclmut_sorted_cut))
                inclmut_all.append(inclmut_sys)
    stop = time.time()
    print('Time to analyze (basic): %s s' % (stop-start))

    Mtot_all = np.array(Mtot_all)
    clustertot_all = np.array(clustertot_all)
    pl_per_cluster_all = np.array(pl_per_cluster_all)

    P_all = np.array(P_all)
    clusterids_all = np.array(clusterids_all)
    e_all = np.array(e_all)
    incl_all = np.array(incl_all)
    inclmut_all = np.array(inclmut_all)
    radii_all = np.array(radii_all)
    mass_all = np.array(mass_all)

    if len(inclmut_all) == 0:
        inclmut_all = np.zeros((len(Mtot_all), Mmax)) # need this to 'compute AMD' when mutual inclinations are not available

    if len(Mstar_all) != 0:
        mu_all = mass_all/Mstar_all[:,None] # planet/star mass ratios
        a_all = gen.a_from_P(P_all, Mstar_all[:,None]) # 2d array of semimajor axes
        AMD_all = gen.AMD(mu_all, a_all, e_all, inclmut_all) # 2d array of AMD
        AMD_tot_all = np.sum(AMD_all, axis=1) # 1d array of total AMD per system
    else:
        mu_all = np.array([])
        a_all = np.array([])
        AMD_all = np.array([])
        AMD_tot_all = np.array([])

    # To convert the radii and masses to Earth units:
    radii_all = radii_all*(gen.Rsun/gen.Rearth) # planet radii in Earth radii
    mass_all = mass_all*(gen.Msun/gen.Mearth) # planet masses in Earth masses

    if len(P_all) != 0:
        Mtot_all = np.sum(P_all > 0, axis=1)
    else:
        print('Multiplicities (Mtot_all) computed from clusterids only.')

    sssp_per_sys_basic = {}
    # Total planet, cluster, and planets per cluster multiplicities:
    sssp_per_sys_basic['Mmax'] = Mmax
    sssp_per_sys_basic['Mtot_all'] = Mtot_all
    sssp_per_sys_basic['clustertot_all'] = clustertot_all
    sssp_per_sys_basic['pl_per_cluster_all'] = pl_per_cluster_all
    # Planet properties:
    sssp_per_sys_basic['P_all'] = P_all
    sssp_per_sys_basic['clusterids_all'] = clusterids_all
    sssp_per_sys_basic['e_all'] = e_all
    sssp_per_sys_basic['inclmut_all'] = inclmut_all
    sssp_per_sys_basic['incl_all'] = incl_all
    sssp_per_sys_basic['radii_all'] = radii_all
    sssp_per_sys_basic['mass_all'] = mass_all
    # Stellar dependent properties:
    sssp_per_sys_basic['Mstar_all'] = Mstar_all
    sssp_per_sys_basic['Rstar_all'] = Rstar_all
    sssp_per_sys_basic['mu_all'] = mu_all
    sssp_per_sys_basic['a_all'] = a_all
    sssp_per_sys_basic['AMD_all'] = AMD_all
    sssp_per_sys_basic['AMD_tot_all'] = AMD_tot_all

    return sssp_per_sys_basic

def load_cat_phys_separate_and_compute_basic_summary_stats_per_sys(file_name_path, run_number):
    """
    Load a physical catalog and compute the basic summary statistics per system.

    Wrapper for the functions :py:func:`syssimpyplots.load_sims.load_planets_stars_phys_separate` and :py:func:`syssimpyplots.load_sims.compute_basic_summary_stats_per_sys_cat_phys`.

    Parameters
    ----------
    file_name_path : str
        The path to the physical catalog.
    run_number : str
        The run number appended to the file names for the physical catalog.

    Returns
    -------
    sssp_per_sys_basic : dict
        A dictionary containing planetary and stellar properties for each system. See the documentation for :py:func:`syssimpyplots.load_sims.compute_basic_summary_stats_per_sys_cat_phys` for a description of the dictionary fields.
    """
    clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all = load_planets_stars_phys_separate(file_name_path, run_number)

    sssp_per_sys_basic = compute_basic_summary_stats_per_sys_cat_phys(clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all)
    return sssp_per_sys_basic

def compute_summary_stats_from_cat_phys(cat_phys=None, star_phys=None, file_name_path=None, run_number='', load_full_tables=False, compute_ratios=gen.compute_ratios_adjacent, match_observed=True):
    """
    Compute detailed summary statistics per system in a simulated physical catalog.

    Note
    ----
    This function can be used by either passing a ``cat_phys`` and ``star_phys`` for the physical catalog, or by passing a ``file_name_path`` and ``run_number`` from which to load the physical catalog. If the latter, will load the individual files with the planet and star properties using :py:func:`syssimpyplots.load_sims.load_planets_stars_phys_separate`.

    Parameters
    ----------
    cat_phys : structured array, default=None
        A table with the physical properties of all the planets.
    star_phys : structured array, default=None
        A table with the basic properties of the planet-hosting stars.
    file_name_path : str, default=None
        The path to the physical catalog.
    run_number : str, default=''
        The run number appended to the file names for the physical catalog.
    load_full_tables : bool, default=False
        Whether to load full tables of the physical catalogs. Required to be True if also want to match the physical planets to the observed planets.
    compute_ratios : func, default=compute_ratios_adjacent
        The function to use for computing ratios; can be either :py:func:`syssimpyplots.general.compute_ratios_adjacent` or :py:func:`syssimpyplots.general.compute_ratios_all`.
    match_observed : bool, default=True
        Whether to match the physical planets to the observed planets. If True, the output will also contain a field `det_all`.

    Returns
    -------
    sssp_per_sys : dict
        A dictionary containing the planetary and stellar properties for each system (2-d and 1-d arrays).
    sssp : dict
        A dictionary containing the planetary and stellar properties of all planets (1-d arrays).


    The outputs are two dictionaries. ``sssp_per_sys`` contains the following fields:

    - `det_all`: The detection flags (1=detected, 0=undetected) of the planets in each system (2-d array). Only returned if `match_observed=True`.
    - `Mtot_all`: The number of planets in each system (1-d array).
    - `clusterids_all`: The cluster id's of each system (2-d array).
    - `P_all`: The orbital periods (days) of each system (2-d array).
    - `a_all`: The semi-major axes (AU) of each system (2-d array).
    - `radii_all`: The planet radii (Earth radii) of each system (2-d array).
    - `mass_all`: The planet masses (Earth masses) of each system (2-d array).
    - `mu_all`: The planet/star mass ratios of each system (2-d array).
    - `e_all`: The orbital eccentricities of each system (2-d array).
    - `inclmut_all`: The orbital inclinations (radians) relative to the system invariable plane of each system (2-d array).
    - `incl_all`: The orbital inclinations (radians) relative to the sky plane of each system (2-d array).
    - `AMD_all`: The AMDs (units of G*Mstar=1) of each system (2-d array).
    - `Rm_all`: The period ratios of each system (2-d array).
    - `radii_ratio_all`: The planet radius ratios of each system (2-d array).
    - `N_mH_all`: The planet spacings in mutual Hill radii of each system (2-d array).
    - `dynamical_mass`: The 'dynamical mass' of each system (1-d array).
    - `radii_partitioning`: The 'radius partitioning' of each multi-planet system (1-d array).
    - `radii_monotonicity`: The 'radius monotonicity' of each multi-planet system (1-d array).
    - `gap_complexity`: The 'gap complexity' of each system with 3+ planets (1-d array).

    ``sssp`` contains the following fields:

    - `Mstar_all`: The stellar mass (solar masses) of each system (1-d array).
    - `Rstar_all`: The stellar radius (solar radii) of each system (1-d array).
    - `clustertot_all`: The number of planet clusters in each system (1-d array).
    - `AMD_tot_all`: The total AMD (units of G*Mstar=1) of each system (1-d array).
    - `pl_per_cluster_all`: The number of planets in each cluster (1-d array).
    - `P_all`: The orbital periods (days) of all planets (1-d array).
    - `radii_all`: The radii (Earth radii) of all planets (1-d array).
    - `mass_all`: The masses (Earth masses) of all planets (1-d array).
    - `e_all`: The orbital eccentricities of all planets (1-d array).
    - `inclmut_all`: The orbital inclinations (radians) of all planets relative to the system invariable planes (1-d array).
    - `incl_all`: The orbital inclinations (radians) of all planets relative to the sky plane (1-d array).
    - `radii_above_all`: The radii (Earth radii) of all planets above the photo-evaporation boundary\* (1-d array).
    - `radii_below_all`: The radii (Earth radii) of all planets below the photo-evaporation boundary\* (1-d array).
    - `Rm_all`: The orbital period ratios (1-d array).
    - `radii_ratio_all`: The planet radii ratios (1-d array).
    - `N_mH_all`: The planet spacings in mutual Hill radii (1-d array).
    - `radii_ratio_above_all`: The planet radii ratios for all planets above the photo-evaporation boundary\* (1-d array).
    - `radii_ratio_below_all`: The planet radii ratios for all planets below the photo-evaporation boundary\* (1-d array).
    - `radii_ratio_across_all`: The planet radii ratios for all planets across the photo-evaporation boundary\* (1-d array).

    Note
    ----
    \*The photo-evaporation boundary is defined by the function :py:func:`syssimpyplots.general.photoevap_boundary_Carrera2018`.

    Warning
    -------
    For the 2-d arrays, each row is padded with zeros (or negative ones), since different systems have different numbers of planets.
    """
    #This function takes in a simulated observed catalog of planets 'cat_phys' in table format and returns many arrays (1D and 2D) of the summary stats

    if load_full_tables or (cat_phys is not None and star_phys is not None):
        if cat_phys is None:
            cat_phys = load_cat_phys(file_name_path + 'physical_catalog%s.csv' % run_number)
        if star_phys is None:
            star_phys = load_star_phys(file_name_path + 'physical_catalog_stars%s.csv' % run_number)

        start = time.time()

        i_sys = np.unique(cat_phys['target_id'])
        N_sys_with_planets = len(i_sys) # number of simulated systems with planets

        clusterids_per_sys = []
        P_per_sys = [] # periods per system (days)
        e_per_sys = [] # eccentricities per system
        inclmut_per_sys = [] # mutual inclinations (rad) per system
        incl_per_sys = [] # sky inclinations (rad) per system
        radii_per_sys = [] # planet radii per system (solar radii)
        mass_per_sys = [] # planet masses per system (solar masses)
        num_planets_cumu = np.concatenate((np.array([0]), np.cumsum(np.array(star_phys['num_planets']))))
        for i in range(len(num_planets_cumu) - 1):
            cat_phys_sys = cat_phys[num_planets_cumu[i]:num_planets_cumu[i+1]]

            clusterids_per_sys.append(cat_phys_sys['clusterid'])
            P_per_sys.append(cat_phys_sys['period'])
            e_per_sys.append(cat_phys_sys['ecc'])
            #inclmut_per_sys.append(cat_phys_sys['incl_mut'])
            inclmut_per_sys.append(cat_phys_sys['incl_invariable'])
            incl_per_sys.append(cat_phys_sys['incl'])
            radii_per_sys.append(cat_phys_sys['planet_radius'])
            mass_per_sys.append(cat_phys_sys['planet_mass'])
        Mstar_all = star_phys['star_mass']
        Rstar_all = star_phys['star_radius']
        targetid_all = star_phys['target_id']

        stop = time.time()
        print('Time to sort planet table into systems: %s s' % (stop - start))
    else:
        clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all = load_planets_stars_phys_separate(file_name_path, run_number)



    sssp_per_sys_basic = compute_basic_summary_stats_per_sys_cat_phys(clusterids_per_sys, P_per_sys, radii_per_sys, mass_per_sys, e_per_sys, inclmut_per_sys, incl_per_sys, Mstar_all, Rstar_all)

    # Total planet, cluster, and planets per cluster multiplicities:
    Mmax = sssp_per_sys_basic['Mmax']
    Mtot_all = sssp_per_sys_basic['Mtot_all']
    clustertot_all = sssp_per_sys_basic['clustertot_all']
    pl_per_cluster_all = sssp_per_sys_basic['pl_per_cluster_all']

    # Planet properties:
    P_all = sssp_per_sys_basic['P_all']
    clusterids_all = sssp_per_sys_basic['clusterids_all']
    e_all = sssp_per_sys_basic['e_all']
    inclmut_all = sssp_per_sys_basic['inclmut_all']
    incl_all = sssp_per_sys_basic['incl_all']
    radii_all = sssp_per_sys_basic['radii_all']
    mass_all = sssp_per_sys_basic['mass_all']

    # Stellar dependent properties:
    mu_all = sssp_per_sys_basic['mu_all']
    a_all = sssp_per_sys_basic['a_all']
    AMD_all = sssp_per_sys_basic['AMD_all']
    AMD_tot_all = sssp_per_sys_basic['AMD_tot_all']



    # To compute which planets are detected:
    if match_observed and cat_phys is not None and star_phys is not None:
        start = time.time()
        print('Loading observed catalog to match observed planets to physical planets...')
        cat_obs = load_cat_obs(file_name_path + 'observed_catalog%s.csv' % run_number)

        ferr_match = 0.05 # fractional error in period to match
        print('Matching periods to within %s (fractional error)...' % ferr_match)

        det_all = np.zeros(np.shape(P_all))
        for i,tid in enumerate(cat_obs['target_id']):
            id_sys = np.where(targetid_all == tid)[0][0]
            fp_diff = np.abs(P_all[id_sys] - cat_obs['period'][i]) / cat_obs['period'][i]
            #print(i, ': P = ', cat_obs['period'][i], ' -- ', P_all[id_sys])
            id_pl = np.where(fp_diff < ferr_match)[0][0]
            det_all[id_sys, id_pl] = 1.0

        stop = time.time()
        print('Time to match planets: %s s' % (stop - start))



    #To calculate the underlying period ratios, radii ratios, and separations in mutual Hill radii:
    Rm_all = [] #list to be filled with all the period ratios
    radii_ratio_all = [] #list to be filled with all the radii ratios
    N_mH_all = [] #list to be filled with all the separations between adjacent planet pairs in units of mutual Hill radii

    radii_above_all_flat = [] #list to be filled with the radii of planets above the photoevaporation boundary
    radii_below_all_flat = [] #list to be filled with the radii of planets below the photoevaporation boundary
    radii_ratio_above_all_flat = [] #list to be filled with the radii ratios of adjacent planet pairs above the photoevaporation boundary
    radii_ratio_below_all_flat = [] #list to be filled with the radii ratios of adjacent planet pairs below the photoevaporation boundary
    radii_ratio_across_all_flat = [] #list to be filled with the radii ratios of adjacent planet pairs across the photoevaporation boundary

    start = time.time()
    for i in range(len(P_all)):
        Mstar_system = Mstar_all[i] #mass of the star for this system, in solar masses
        P_all_system = P_all[i][P_all[i] > 0]
        e_all_system = e_all[i][P_all[i] > 0]
        radii_all_system = radii_all[i][P_all[i] > 0]
        mass_all_system = mass_all[i][P_all[i] > 0]

        #To calculate all the period ratios:
        Rm_all_system = list(compute_ratios(P_all_system)) #list of period ratios in this system
        Rm_all_system = np.array(Rm_all_system + [0]*(Mmax - 1 - len(Rm_all_system))) #to add filler 0's to Rm_all_system to pad it to Mmax - 1 elements
        Rm_all.append(Rm_all_system)

        #To calculate all the radii ratios:
        radii_ratio_all_system = list(compute_ratios(radii_all_system)) #list of radii ratios in this system
        radii_ratio_all_system = np.array(radii_ratio_all_system + [0]*(Mmax - 1 - len(radii_ratio_all_system))) #to add filler 0's to radii_ratio_all_system to pad it to Mmax - 1 elements
        radii_ratio_all.append(radii_ratio_all_system)

        #To calculate all the separations in mutual Hill radii between adjacent planet pairs:
        a_all_system = gen.a_from_P(P_all_system, Mstar_system)
        R_mH_all_system = ((a_all_system[0:-1] + a_all_system[1:])/2.)*(gen.Mearth*(mass_all_system[0:-1] + mass_all_system[1:])/(3.*Mstar_system*gen.Msun))**(1./3.) #mutual Hill radii between adjacent planet pairs in this system, in AU
        #R_sep_all_system = a_all_system[1:] - a_all_system[0:-1] #separations between adjacent planet pairs in this system, in AU, ignoring eccentricities
        R_sep_all_system = a_all_system[1:]*(1. - e_all_system[1:]) - a_all_system[0:-1]*(1. + e_all_system[0:-1]) #separations between adjacent planet pairs in this system, in AU, including eccentricities
        N_mH_all_system = list(R_sep_all_system/R_mH_all_system) #separations between adjacent planet pairs in this system, in mutual Hill radii
        N_mH_all_system = np.array(N_mH_all_system + [0]*(Mmax - 1 - len(N_mH_all_system))) #to add filler 0's to N_mH_all_system to pad it to Mmax - 1 elements
        N_mH_all.append(N_mH_all_system)

        #To separate the planets in the system as above and below the boundary:
        system_above_bools = np.array([gen.photoevap_boundary_Carrera2018(radii_all_system[x], P_all_system[x]) for x in range(len(P_all_system))])
        #if len(system_above_bools) > 1:
        #print(system_above_bools)

        #To record the transit depths of the planets above and below the boundary:
        for j in range(len(radii_all_system)):
            radii_above_all_flat.append(radii_all_system[j]) if system_above_bools[j] == 1 else radii_below_all_flat.append(radii_all_system[j])

        #To record the transit depth ratios of the planets above, below, and across the boundary:
        radii_ratio_all_system = list(compute_ratios(radii_all_system)) #list of radii ratios in this system
        for j in range(len(radii_ratio_all_system)):
            if system_above_bools[j] + system_above_bools[j+1] == 2: #both planets are above the boundary
                radii_ratio_above_all_flat.append(radii_ratio_all_system[j])
            elif system_above_bools[j] + system_above_bools[j+1] == 1: #one planet is above, the other is below the boundary
                radii_ratio_across_all_flat.append(radii_ratio_all_system[j])
            elif system_above_bools[j] + system_above_bools[j+1] == 0: #both planets are below the boundary
                radii_ratio_below_all_flat.append(radii_ratio_all_system[j])
    stop = time.time()
    print('Time to analyze: %s s' % (stop-start))

    Rm_all = np.array(Rm_all)
    radii_ratio_all = np.array(radii_ratio_all)
    N_mH_all = np.array(N_mH_all)

    inclmut_all_flat = []
    if len(inclmut_per_sys) != 0:
        # NOTE: getting rid of all the 'mutual inclinations' for the single planets (since they would all be zero or are not real)
        inclmut_all_2plus = inclmut_all[Mtot_all > 1,:]
        P_all_2plus = P_all[Mtot_all > 1,:]
        inclmut_all_flat = inclmut_all_2plus[P_all_2plus > 0] #all the mutual inclinations of all the planets (which can be zero; in particular, all intrinsic singles are zero)
    inclmut_all_flat = np.array(inclmut_all_flat)

    radii_above_all_flat = np.array(radii_above_all_flat)
    radii_below_all_flat = np.array(radii_below_all_flat)
    radii_ratio_above_all_flat = np.array(radii_ratio_above_all_flat)
    radii_ratio_below_all_flat = np.array(radii_ratio_below_all_flat)
    radii_ratio_across_all_flat = np.array(radii_ratio_across_all_flat)

    # Create dictionaries to hold summary stats ('sssp' stands for 'summary stats simulated physical'):

    sssp_per_sys = {}
    # Planet properties:
    if match_observed and cat_phys is not None:
        sssp_per_sys['det_all'] = det_all
    sssp_per_sys['Mtot_all'] = Mtot_all
    sssp_per_sys['clusterids_all'] = clusterids_all
    sssp_per_sys['P_all'] = P_all
    sssp_per_sys['a_all'] = a_all
    sssp_per_sys['radii_all'] = radii_all
    sssp_per_sys['mass_all'] = mass_all
    sssp_per_sys['mu_all'] = mu_all
    sssp_per_sys['e_all'] = e_all
    sssp_per_sys['inclmut_all'] = inclmut_all
    sssp_per_sys['incl_all'] = incl_all
    sssp_per_sys['AMD_all'] = AMD_all
    # Planet property ratios:
    sssp_per_sys['Rm_all'] = Rm_all
    sssp_per_sys['radii_ratio_all'] = radii_ratio_all
    sssp_per_sys['N_mH_all'] = N_mH_all

    sssp = {}
    # Stellar properties:
    sssp['Mstar_all'] = Mstar_all
    sssp['Rstar_all'] = Rstar_all
    # Planet properties:
    sssp['clustertot_all'] = clustertot_all
    sssp['AMD_tot_all'] = AMD_tot_all
    sssp['pl_per_cluster_all'] = pl_per_cluster_all
    sssp['P_all'] = P_all[P_all > 0]
    sssp['radii_all'] = radii_all[radii_all > 0]
    sssp['mass_all'] = mass_all[mass_all > 0]
    sssp['e_all'] = e_all[P_all > 0] # can be zero, so use periods to get real planets
    sssp['inclmut_all'] = inclmut_all_flat
    sssp['incl_all'] = incl_all[incl_all > 0]
    sssp['radii_above_all'] = radii_above_all_flat
    sssp['radii_below_all'] = radii_below_all_flat
    # Planet property ratios:
    sssp['Rm_all'] = Rm_all[Rm_all > 0]
    sssp['radii_ratio_all'] = radii_ratio_all[radii_ratio_all > 0]
    sssp['N_mH_all'] = N_mH_all[N_mH_all > 0]
    sssp['radii_ratio_above_all'] = radii_ratio_above_all_flat
    sssp['radii_ratio_below_all'] = radii_ratio_below_all_flat
    sssp['radii_ratio_across_all'] = radii_ratio_across_all_flat

    # To compute some summary stats (system-level metrics) from GF2020:
    Nsys_all = len(Mtot_all)
    assert Nsys_all == len(radii_all) == len(P_all)

    dynamical_mass = []
    radii_partitioning = []
    radii_monotonicity = []
    gap_complexity = []

    start = time.time()
    for i in range(Nsys_all):
        P_all_system = P_all[i][P_all[i] > 0]
        radii_all_system = radii_all[i][P_all[i] > 0]
        mu_all_system = mu_all[i][P_all[i] > 0]

        dynamical_mass.append(np.sum(mu_all_system))

        if len(P_all_system) >= 2:
            radii_partitioning.append(gen.partitioning(radii_all_system))
            radii_monotonicity.append(gen.monotonicity_GF2020(radii_all_system))
        if len(P_all_system) >= 3:
            gap_complexity.append(gen.gap_complexity_GF2020(P_all_system))
    stop = time.time()
    print('Time to analyze (GF2020): %s s' % (stop-start))

    sssp_per_sys['dynamical_mass'] = np.array(dynamical_mass)
    sssp_per_sys['radii_partitioning'] = np.array(radii_partitioning)
    sssp_per_sys['radii_monotonicity'] = np.array(radii_monotonicity)
    sssp_per_sys['gap_complexity'] = np.array(gap_complexity)

    return [sssp_per_sys, sssp]





# Functions to load and analyze simulated observed catalogs:

def load_cat_obs(file_name):
    """
    Load a table with all the planets in a simulated observed catalog.

    Parameters
    ----------
    file_name : str
        The path/name of the file for the observed catalog (should end with ‘observed_catalog.csv’).

    Returns
    -------
    cat_obs : structured array
        A table with the observed properties of all the planets.


    The table has the following columns:

    - `target_id`: The index of the star in the simulation (e.g. 1 for the first star) which the planet orbits.
    - `star_id`: The index of the star based on where it is in the input stellar catalog.
    - `period`: The observed orbital period (days).
    - `period_err`: The uncertainty in observed orbital period (days).
    - `depth`: The observed transit depth.
    - `depth_err`: The uncertainty in observed transit depth.
    - `duration`: The observed transit duration (days).
    - `duration_err`: The uncertainty in observed transit duration (days).
    - `star_mass`: The stellar mass (solar masses).
    - `star_radius`: The stellar radius (solar radii).

    """
    with open(file_name, 'r') as file:
        lines = (line for line in file if not line.startswith('#'))
        cat_obs = np.loadtxt(lines, skiprows=1, dtype={'names': ('target_id', 'star_id', 'period', 'period_err', 'depth', 'depth_err', 'duration', 'duration_err', 'star_mass', 'star_radius'), 'formats': ('i4', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')}, delimiter=',')

    return cat_obs

def load_star_obs(file_name):
    """
    Load a table of only the stars with observed planets in a simulated observed catalog.

    Parameters
    ----------
    file_name : str
        The path/name of the file for the stellar physical catalog (should end with ‘observed_catalog_stars.csv’)

    Returns
    -------
    star_obs : structured array
        A table with the basic properties of the observed planet-hosting stars.


    The table has the same columns as those returned by the function :py:func:`syssimpyplots.load_sims.load_star_phys`.
    """
    with open(file_name, 'r') as file: #open(loadfiles_directory + 'observed_catalog_stars%s.txt' % run_number, 'r')
        lines = (line for line in file if not line.startswith('#'))
        star_obs = np.loadtxt(lines, skiprows=1, dtype={'names': ('target_id', 'star_id', 'star_mass', 'star_radius', 'num_obs_planets'), 'formats': ('i4', 'i4', 'f8', 'f8', 'i4')}, delimiter=',')

    return star_obs

def load_planets_stars_obs_separate(file_name_path, run_number):
    """
    Load individual files with the properties of all the planets and stars in a simulated observed catalog.

    Note
    ----
    Faster than :py:func:`syssimpyplots.load_sims.load_cat_obs` for large catalogs, but returns individual lists instead of a single table. Each list is ordered in the same way (low to high observed multiplicity) so the planet properties can be matched to each other.

    Parameters
    ----------
    file_name_path : str
        The path to the observed catalog.
    run_number : str
        The run number appended to the file names for the observed catalog.

    Returns
    -------
    P_per_sys : list[list]
        The observed orbital periods (days) of each system.
    D_per_sys : list[list]
        The observed transit depths of each system.
    tdur_per_sys : list[list]
        The observed transit durations (days) of each system.
    Mstar_per_sys : array[float]
        The stellar mass (solar masses) of each system.
    Rstar_per_sys : array[float]
        The stellar radius (solar radii) of each system.
    """
    P_per_sys = [] #list to be filled with lists of the observed periods per system (days)
    with open(file_name_path + 'periods%s.out' % run_number, 'r') as file:
        for line in file:
            if line[0] != '#':
                line = line[1:-2]
                line_per_sys = line.split('; ')
                #print(len(line_per_sys))
                for x in line_per_sys:
                    P_sys = x.split()
                    P_sys = [float(i) for i in P_sys]
                    P_per_sys.append(P_sys)

    D_per_sys = [] #list to be filled with lists of the transit depths per system
    with open(file_name_path + 'depths%s.out' % run_number, 'r') as file:
        for line in file:
            if line[0] != '#':
                line = line[1:-2]
                line_per_sys = line.split('; ')
                #print(len(line_per_sys))
                for x in line_per_sys:
                    D_sys = x.split()
                    D_sys = [float(i) for i in D_sys]
                    D_per_sys.append(D_sys)

    tdur_per_sys = [] #list to be filled with lists of the transit durations per system (days)
    with open(file_name_path + 'durations%s.out' % run_number, 'r') as file:
        for line in file:
            if line[0] != '#':
                line = line[1:-2]
                line_per_sys = line.split('; ')
                #print(len(line_per_sys))
                for x in line_per_sys:
                    tdur_sys = x.split()
                    tdur_sys = [float(i) for i in tdur_sys]
                    tdur_per_sys.append(tdur_sys)

    Mstar_per_sys = [] #list to be filled with the stellar masses of the systems with observed planets (Msun)
    with open(file_name_path + 'stellar_masses_obs%s.out' % run_number, 'r') as file:
        for line in file:
            if line[0] != '#':
                line = line[1:-2]
                Mstars = line.split(', ')
                Mstars = [float(i) for i in Mstars]
                Mstar_per_sys += Mstars
    Mstar_per_sys = np.array(Mstar_per_sys)

    Rstar_per_sys = [] #list to be filled with the stellar radii of the systems with observed planets (Rsun)
    with open(file_name_path + 'stellar_radii_obs%s.out' % run_number, 'r') as file:
        for line in file:
            if line[0] != '#':
                line = line[1:-2]
                Rstars = line.split(', ')
                Rstars = [float(i) for i in Rstars]
                Rstar_per_sys += Rstars
    Rstar_per_sys = np.array(Rstar_per_sys)

    return P_per_sys, D_per_sys, tdur_per_sys, Mstar_per_sys, Rstar_per_sys

def count_planets_from_loading_cat_obs_stars_only(file_name_path=None, run_number='', Rstar_min=0., Rstar_max=10., Mstar_min=0., Mstar_max=10., teff_min=0., teff_max=1e4, bp_rp_min=-5., bp_rp_max=5.):
    """
    Count the number of observed planets in each system (and the resulting observed multiplicity distribution), given a set of stellar cuts.

    Note
    ----
    Loads an 'observed_catalog_stars.csv' file (using :py:func:`syssimpyplots.load_sims.load_star_obs`) and a table of cleaned Kepler target stars (using :py:func:`syssimpyplots.compare_kepler.load_Kepler_stars_cleaned`).

    Parameters
    ----------
    file_name_path : str, default=None
        The path to the observed catalog.
    run_number : str, default=''
        The run number appended to the file names for the observed catalog.
    Rstar_min : float, default=0.
        The minimum stellar radius (solar radii) to include in the sample.
    Rstar_max= : float, default=10.
        The maximum stellar radius (solar radii) to include in the sample.
    Mstar_min : float, default=0.
        The minimum stellar mass (solar masses) to include in the sample.
    Mstar_max : float, default=10.
        The maximum stellar mass (solar masses) to include in the sample.
    teff_min : float, default=0.
        The minimum stellar effective temperature (K) to include in the sample.
    teff_max : float, default=1e4
        The maximum stellar effective temperature (K) to include in the sample.
    bp_rp_min : float, default=-5.
        The minimum Gaia DR2 bp-rp color to include in the sample.
    bp_rp_max : float, default=5.
        The maximum Gaia DR2 bp-rp color to include in the sample.

    Returns
    -------
    Mtot_obs : array[int]
        The number of observed planets in each system.
    Nmult_obs : array[int]
        The observed multiplicity distribution (number of observed systems at each multiplicity order).
    """
    stars_cleaned = ckep.load_Kepler_stars_cleaned() # NOTE: make sure that this is loading the same stellar catalog used for the simulations!

    star_obs = load_star_obs(file_name_path + 'observed_catalog_stars%s.csv' % run_number)
    i_stars_obs = list(star_obs['star_id']-1) # star_id were indexed in Julia, which starts at 1
    bools_stars_obs_in_custom = np.ones(len(i_stars_obs), dtype=bool)

    Mstar_per_sys = star_obs['star_mass']
    Rstar_per_sys = star_obs['star_radius']
    teff_per_sys = stars_cleaned['teff'][i_stars_obs]
    bp_rp_per_sys = stars_cleaned['bp_rp'][i_stars_obs]
    e_bp_rp_per_sys = stars_cleaned['e_bp_rp_interp'][i_stars_obs]

    indices_keep = np.arange(len(Rstar_per_sys))[bools_stars_obs_in_custom & (Rstar_per_sys >= Rstar_min) & (Rstar_per_sys < Rstar_max) & (Mstar_per_sys >= Mstar_min) & (Mstar_per_sys < Mstar_max) & (teff_per_sys >= teff_min) & (teff_per_sys < teff_max) & (bp_rp_per_sys - e_bp_rp_per_sys >= bp_rp_min) & (bp_rp_per_sys - e_bp_rp_per_sys < bp_rp_max)]

    star_obs_keep = star_obs[indices_keep]
    Mtot_obs = star_obs_keep['num_obs_planets']
    Nmult_obs = np.array([np.sum(Mtot_obs == x) for x in range(1,np.max(Mtot_obs)+1)])
    return Mtot_obs, Nmult_obs

def compute_summary_stats_from_cat_obs(cat_obs=None, star_obs=None, file_name_path=None, run_number='', P_min=0., P_max=300., Rstar_min=0., Rstar_max=10., Mstar_min=0., Mstar_max=10., teff_min=0., teff_max=1e4, bp_rp_min=-5., bp_rp_max=5., i_stars_custom=None, compute_ratios=gen.compute_ratios_adjacent):
    """
    Compute detailed summary statistics per system in a simulated observed catalog.

    Note
    ----
    This function can be used by either passing a ``cat_obs`` and ``star_obs`` for the observed catalog, or by passing a ``file_name_path`` and ``run_number`` from which to load the observed catalog. If the latter, will load the individual files with the planet and star properties using :py:func:`syssimpyplots.load_sims.load_planets_stars_obs_separate`.

    Parameters
    ----------
    cat_obs : structured array, default=None
        A table with the observed properties of the planets.
    star_obs : structured array, default=None
        A table with the basic properties of the observed planet-hosting stars.
    file_name_path : str, default=None
        The path to the observed catalog.
    run_number : str, default=''
        The run number appended to the file names for the observed catalog.
    P_min : float, default=0.
        The minimum orbital period to include in the sample.
    P_max : float, default=300.
        The maximum orbital period to include in the sample.
    Rstar_min : float, default=0.
        The minimum stellar radius (solar radii) to include in the sample.
    Rstar_max : float, default=10.
        The maximum stellar radius (solar radii) to include in the sample.
    Mstar_min : float, default=0.
        The minimum stellar mass (solar masses) to include in the sample.
    Mstar_max : float, default=10.
        The maximum stellar mass (solar masses) to include in the sample.
    teff_min : float, default=0.
        The minimum stellar effective temperature (K) to include in the sample.
    teff_max : float, default=10.
        The maximum stellar effective temperature (K) to include in the sample.
    bp_rp_min : float, default=-5.
        The minimum Gaia DR2 bp-rp color to include in the sample.
    bp_rp_max : float, default=5.
        The maximum Gaia DR2 bp-rp color to include in the sample.
    i_stars_custom : array[int], default=None
        An array of indices for the stars in the Kepler stellar catalog to be included in the sample.
    compute_ratios : func, default=compute_ratios_adjacent
        The function to use for computing ratios; can be either :py:func:`syssimpyplots.general.compute_ratios_adjacent` or :py:func:`syssimpyplots.general.compute_ratios_all`.

    Returns
    -------
    sss_per_sys : dict
        A dictionary containing the planetary and stellar properties for each observed system (2-d and 1-d arrays).
    sss : dict
        A dictionary containing the planetary and stellar properties of all observed planets (1-d arrays).


    The outputs are two dictionaries. ``sss_per_sys`` contains the following fields:

    - `Mstar_obs`: The stellar mass (solar masses) of each system (1-d array).
    - `Rstar_obs`: The stellar radius (solar radii) of each system (1-d array).
    - `teff_obs`: The stellar effective temperature (K) of each system (1-d array).
    - `bp_rp_obs`: The Gaia DR2 bp-rp color (mag) of each system (1-d array).
    - `e_bp_rp_obs`: The extinction in bp-rp color interpolated from a model, of each system (1-d array).
    - `cdpp4p5_obs`: The 4.5 hr duration combined differential photometric precision of each system (1-d array).
    - `Mtot_obs`: The observed number of planets in each system (1-d array).
    - `P_obs`: The observed orbital periods (days) of each system (2-d array).
    - `D_obs`: The observed transit depths of each system (2-d array).
    - `tdur_obs`: The observed transit durations of each system (2-d array).
    - `tdur_tcirc_obs`: The observed transit durations normalized by the durations of the circular orbits of each system (2-d array).
    - `radii_obs`: The observed planet radii (Earth radii) of each system (2-d array).
    - `Rm_obs`: The observed period ratios of each system (2-d array).
    - `D_ratio_obs`: The observed transit depth ratios of each system (2-d array).
    - `xi_obs`: The observed period-normalized transit duration ratios ('xi' values) of each system (2-d array).
    - `xi_res_obs`: The observed 'xi' values of planet-pairs near a resonance\* (2-d array).
    - `xi_res32_obs`: The observed 'xi' values of planet-pairs near the 3:2 resonance (2-d array).
    - `xi_res21_obs`: The observed 'xi' values of planet-pairs near the 2:1 resonance (2-d array).
    - `xi_nonres_obs`: The observed 'xi' values of planet-pairs not near any resonances\* (2-d array).
    - `radii_star_ratio`: The observed sum of planet/star radius ratios of each system (1-d array).
    - `radii_partitioning`: The observed 'radius partitioning' of each multi-planet system (1-d array).
    - `radii_monotonicity`: The observed 'radius monotonicity' of each multi-planet system (1-d array).
    - `gap_complexity`: The observed 'gap complexity' of each system with 3+ planets (1-d array).

    ``sss`` contains the following fields:

    - `Mstar_obs`: The stellar mass (solar masses) of each system, repeated for each planet in the system (1-d array).
    - `Rstar_obs`: The stellar radius (solar radii) of each system, repeated for each planet in the system (1-d array).
    - `teff_obs`: The stellar effective temperature (K) of each system, repeated for each planet in the system (1-d array).
    - `bp_rp_obs`: The Gaia DR2 bp-rp color (mag) of each system, repeated for each planet in the system (1-d array).
    - `e_bp_rp_obs`: The extinction in bp-rp color interpolated from a model, of each system, repeated for each planet in the system (1-d array).
    - `cdpp4p5_obs`: The 4.5 hr duration combined differential photometric precision of each system, repeated for each planet in the system (1-d array).
    - `Nmult_obs`: The observed multiplicity distribution (1-d array).
    - `P_obs`: The observed periods (days) of all observed planets (1-d array).
    - `D_obs`: The observed transit depths of all observed planets (1-d array).
    - `tdur_obs`: The observed transit duration (hrs) of all observed planets (1-d array).
    - `tdur_tcirc_obs`: The observed transit durations normalized by the durations of the circular orbits, of all observed planets (1-d array).
    - `tdur_tcirc_1_obs`: The observed transit durations normalized by the durations of the circular orbits, of all observed single planets (1-d array).
    - `tdur_tcirc_2p_obs`: The observed transit durations normalized by the durations of the circular orbits, of all observed multi-planets (1-d array).
    - `radii_obs`: The observed radii (Earth radii) of all observed planets (1-d array).
    - `D_above_obs`: The observed transit depths of all planets above the photo-evaporation boundary\** (1-d array).
    - `D_below_obs`: The observed transit depths of all planets below the photo-evaporation boundary\** (1-d array).
    - `Rm_obs`: The observed period ratios (1-d array).
    - `D_ratio_obs`: The observed transit depth ratios (1-d array).
    - `xi_obs`: The observed 'xi' values (1-d array).
    - `xi_res_obs`: The observed 'xi' values of all planet-pairs near a resonance\* (1-d array).
    - `xi_res32_obs`: The observed 'xi' values of all planet-pairs near the 3:2 resonance (1-d array).
    - `xi_res21_obs`: The observed 'xi' values of all planet-pairs near the 2:1 resonance (1-d array).
    - `xi_nonres_obs`: The observed 'xi' values of all planet-pairs not near any resonances\* (1-d array).
    - `D_ratio_above_obs`: The observed transit depth ratios of all planets above the photo-evaporation boundary\** (1-d array).
    - `D_ratio_below_obs`: The observed transit depth ratios of all planets below the photo-evaporation boundary\** (1-d array).
    - `D_ratio_across_obs`: The observed transit depth ratios of all planets across the photo-evaporation boundary\** (1-d array).

    Note
    -----
    \*As defined by ``res_ratios`` and ``res_width`` in :ref:`general.py <api_general>`.

    \**The photo-evaporation boundary defined by the function :py:func:`syssimpyplots.general.photoevap_boundary_Carrera2018`.

    Warnings
    --------
    1. For the 2-d arrays, each row is padded with zeros (or negative ones), since different systems have different numbers of observed planets.

    2. The observed transit durations (`tdur_obs`, and thus also fields involving `tdur_tcirc_obs` and `xi_obs`) can be zero!
    """
    if cat_obs is not None and star_obs is not None:
        i_sys = np.unique(cat_obs['target_id'])

        P_per_sys = [] #list to be filled with lists of the observed periods per system (days)
        D_per_sys = [] #list to be filled with lists of the transit depths per system
        tdur_per_sys = [] #list to be filled with lists of the transit durations per system (days)
        Mstar_per_sys = [] #list to be filled with the stellar masses of the systems with observed planets (Msun)
        Rstar_per_sys = [] #list to be filled with the stellar radii of the systems with observed planets (Rsun)
        for i in i_sys:
            P_sys = cat_obs['period'][cat_obs['target_id'] == i]
            D_sys = cat_obs['depth'][cat_obs['target_id'] == i]
            tdur_sys = cat_obs['duration'][cat_obs['target_id'] == i]
            Mstar_sys = star_obs['star_mass'][star_obs['target_id'] == i][0]
            Rstar_sys = star_obs['star_radius'][star_obs['target_id'] == i][0]

            P_per_sys.append(P_sys)
            D_per_sys.append(D_sys)
            tdur_per_sys.append(tdur_sys)
            Mstar_per_sys.append(Mstar_sys)
            Rstar_per_sys.append(Rstar_sys)
        Mstar_per_sys = np.array(Mstar_per_sys)
        Rstar_per_sys = np.array(Rstar_per_sys)

    if file_name_path is not None:
        P_per_sys, D_per_sys, tdur_per_sys, Mstar_per_sys, Rstar_per_sys = load_planets_stars_obs_separate(file_name_path, run_number)
        star_obs = load_star_obs(file_name_path + 'observed_catalog_stars%s.csv' % run_number)

    # For loading and matching other stellar properties:
    stars_cleaned = ckep.load_Kepler_stars_cleaned() # NOTE: make sure that this is loading the same stellar catalog used for the simulations!

    i_stars_obs = list(star_obs['star_id']-1) # star_id were indexed in Julia, which starts at 1
    if i_stars_custom is not None:
        i_stars_obs_custom = np.array(list(set(i_stars_obs).intersection(i_stars_custom))) # star_id of stars in both 'i_stars_obs' and 'i_stars_custom'

        bools_stars_obs_in_custom = np.zeros(len(i_stars_obs), dtype=bool)
        for i,id in enumerate(i_stars_obs):
            bools_stars_obs_in_custom[i] = 1 if (id in i_stars_obs_custom) else 0
    else:
        bools_stars_obs_in_custom = np.ones(len(i_stars_obs), dtype=bool)

    teff_per_sys = stars_cleaned['teff'][i_stars_obs]
    bp_rp_per_sys = stars_cleaned['bp_rp'][i_stars_obs]
    e_bp_rp_per_sys = stars_cleaned['e_bp_rp_interp'][i_stars_obs]
    cdpp4p5_per_sys = stars_cleaned['rrmscdpp04p5'][i_stars_obs]





    #To make cuts on stellar properties:
    #NOTE: this assumes that the ordering in Mstar_per_sys and Rstar_per_sys are the same as those in the P-, D-, and tdur-per_sys!
    indices_keep = np.arange(len(Rstar_per_sys))[bools_stars_obs_in_custom & (Rstar_per_sys >= Rstar_min) & (Rstar_per_sys < Rstar_max) & (Mstar_per_sys >= Mstar_min) & (Mstar_per_sys < Mstar_max) & (teff_per_sys >= teff_min) & (teff_per_sys < teff_max) & (bp_rp_per_sys - e_bp_rp_per_sys >= bp_rp_min) & (bp_rp_per_sys - e_bp_rp_per_sys < bp_rp_max)]

    P_per_sys = [P_per_sys[i] for i in indices_keep]
    D_per_sys = [D_per_sys[i] for i in indices_keep]
    tdur_per_sys = [tdur_per_sys[i] for i in indices_keep]
    Mstar_per_sys = Mstar_per_sys[indices_keep]
    Rstar_per_sys = Rstar_per_sys[indices_keep]
    teff_per_sys = teff_per_sys[indices_keep]
    bp_rp_per_sys = bp_rp_per_sys[indices_keep]
    e_bp_rp_per_sys = e_bp_rp_per_sys[indices_keep]
    cdpp4p5_per_sys = cdpp4p5_per_sys[indices_keep]

    Mstar_obs = []
    Rstar_obs = []
    teff_obs = []
    bp_rp_obs = []
    e_bp_rp_obs = []
    cdpp4p5_obs = []
    for i in range(len(P_per_sys)):
        pl_in_sys = len(P_per_sys[i])
        Mstar_obs += [Mstar_per_sys[i]]*pl_in_sys
        Rstar_obs += [Rstar_per_sys[i]]*pl_in_sys
        teff_obs += [teff_per_sys[i]]*pl_in_sys
        bp_rp_obs += [bp_rp_per_sys[i]]*pl_in_sys
        e_bp_rp_obs += [e_bp_rp_per_sys[i]]*pl_in_sys
        cdpp4p5_obs += [cdpp4p5_per_sys[i]]*pl_in_sys
    Mstar_obs = np.array(Mstar_obs)
    Rstar_obs = np.array(Rstar_obs)
    teff_obs = np.array(teff_obs)
    bp_rp_obs = np.array(bp_rp_obs)
    e_bp_rp_obs = np.array(e_bp_rp_obs)
    cdpp4p5_obs = np.array(cdpp4p5_obs)



    P_obs = [] #list to be zero-padded so each list of periods is sorted and has the same length, and then converted to an array
    D_obs = [] #list to be zero-padded so each list of depths is sorted (by period) and has the same length, and then converted to an array
    tdur_obs = [] #list to be zero-padded so each list of transit durations is sorted (by period) and has the same length, and then converted to an array
    tdur_tcirc_obs = [] #list to be zero-padded so each list of circular-normalized transit durations is sorted (by period) and has the same length, and then converted to an array

    Mmax = np.max([len(x) for x in P_per_sys]) #maximum planet multiplicity generated by the clustering method
    for i in range(len(P_per_sys)):
        Mstar = Mstar_per_sys[i]
        Rstar = Rstar_per_sys[i]
        i_sorted = np.argsort(P_per_sys[i]) #array of indices which would sort the system by period
        P_sorted = np.array(P_per_sys[i])[i_sorted]
        P_sorted_cut = P_sorted[(P_sorted > P_min) & (P_sorted < P_max)]
        D_sorted_cut = np.array(D_per_sys[i])[i_sorted][(P_sorted > P_min) & (P_sorted < P_max)]
        tdur_sorted_cut = np.array(tdur_per_sys[i])[i_sorted][(P_sorted > P_min) & (P_sorted < P_max)] * 24. # transit durations converted to hours
        tdur_tcirc_sorted_cut = tdur_sorted_cut/gen.tdur_circ(P_sorted_cut, Mstar, Rstar)

        P_sys = list(P_sorted_cut) + [-1]*(Mmax - len(P_sorted_cut)) #zero-pad the list up to Mmax elements
        D_sys = list(D_sorted_cut) + [0]*(Mmax - len(D_sorted_cut)) #zero-pad the list up to Mmax elements
        tdur_sys = list(tdur_sorted_cut) + [-1]*(Mmax - len(tdur_sorted_cut)) #zero-pad the list up to Mmax elements
        tdur_tcirc_sys = list(tdur_tcirc_sorted_cut) + [-1]*(Mmax - len(tdur_tcirc_sorted_cut)) #zero-pad the list up to Mmax elements

        P_obs.append(P_sys)
        D_obs.append(D_sys)
        tdur_obs.append(tdur_sys)
        tdur_tcirc_obs.append(tdur_tcirc_sys)
    P_obs = np.array(P_obs)
    D_obs = np.array(D_obs)
    tdur_obs = np.array(tdur_obs) # can be zero; padded values are -1
    tdur_tcirc_obs = np.array(tdur_tcirc_obs) # can be zero; padded values are -1

    Mtot_obs = np.sum(P_obs > 0, axis=1) #array of observed planet multiplicites
    Nmult_obs = np.array([np.sum(Mtot_obs == x) for x in range(1,Mmax+1)]) #array of total numbers of systems with observed planet multiplicities of 1,2,3,...,Mmax planets
    radii_obs = np.sqrt(D_obs)*np.transpose([Rstar_per_sys])*(gen.Rsun/gen.Rearth) #array of planet radii, in Earth radii

    # To split 'tdur_tcirc_obs' into arrays for observed singles and multis:
    tdur_tcirc_1_obs = tdur_tcirc_obs[Mtot_obs == 1, 0] # observed singles, 1d
    tdur_tcirc_2p_obs = tdur_tcirc_obs[Mtot_obs > 1] # observed multis, but 2d
    tdur_tcirc_2p_obs = tdur_tcirc_2p_obs[~np.isnan(tdur_tcirc_2p_obs) & (tdur_tcirc_2p_obs >= 0)] # observed multis, 1d



    #To calculate the observed period ratios, period-normalized transit duration ratios, and transit depth ratios:
    Rm_obs = [] #list to be filled with the observed period ratios
    D_ratio_obs = [] #list to be filled with the observed transit depth ratios
    xi_obs = [] #list to be filled with the period-normalized transit duration ratios
    xi_res_obs = [] #list to be filled with the period-normalized transit duration ratios for planet pairs near resonance
    xi_res32_obs = []
    xi_res21_obs = []
    xi_nonres_obs = [] #list to be filled with the period_normalized transit duration ratios for planet pairs not in resonance

    D_above_obs_flat = [] #list to be filled with the transit depths of observed planets above the photoevaporation boundary
    D_below_obs_flat = [] #list to be filled with the transit depths of observed planets below the photoevaporation boundary
    D_ratio_above_obs_flat = [] #list to be filled with the transit depth ratios of adjacent observed planet pairs above the photoevaporation boundary
    D_ratio_below_obs_flat = [] #list to be filled with the transit depth ratios of adjacent observed planet pairs below the photoevaporation boundary
    D_ratio_across_obs_flat = [] #list to be filled with the transit depth ratios of adjacent observed planet pairs across the photoevaporation boundary

    pad_extra = 100 #####

    for i in range(len(P_obs)):
        P_obs_system = P_obs[i][(P_obs[i] > P_min) & (P_obs[i] < P_max)]
        radii_obs_system = radii_obs[i][(P_obs[i] > P_min) & (P_obs[i] < P_max)]
        tdur_obs_system = tdur_obs[i][(P_obs[i] > P_min) & (P_obs[i] < P_max)]
        D_obs_system = D_obs[i][(P_obs[i] > P_min) & (P_obs[i] < P_max)]

        #To calculate all the observed period ratios:
        Rm_obs_system = list(compute_ratios(P_obs_system)) #list of period ratios observed in this system
        Rm_obs_system = np.array(Rm_obs_system + [-1]*(pad_extra - 1 - len(Rm_obs_system))) #to add filler 0's to Rm_obs_system to pad it to Mmax - 1 elements
        Rm_obs.append(Rm_obs_system)

        #To calculate all the observed transit depth ratios:
        D_ratio_obs_system = list(compute_ratios(D_obs_system)) #list of transit depth ratios observed in this system
        D_ratio_obs_system = np.array(D_ratio_obs_system + [-1]*(pad_extra - 1 - len(D_ratio_obs_system))) #to add filler 0's to D_ratio_obs_system to pad it to Mmax - 1 elements
        D_ratio_obs.append(D_ratio_obs_system)

        #To calculate all the period-normalized transit duration ratios:
        xi_obs_system = list(compute_ratios(tdur_obs_system, inverse=True, avoid_div_zeros=True)*(compute_ratios(P_obs_system)**(1./3.))) #list of period-normalized transit duration ratios in this system
        xi_obs_system = np.array(xi_obs_system + [-1]*(pad_extra - 1 - len(xi_obs_system))) #to add filler 0's to xi_obs_system to pad it to Mmax - 1 elements
        xi_obs.append(xi_obs_system)

        #To separate the period-normalized transit duration ratios for planet pairs near vs. not in resonance:
        mask_res_system = np.zeros(len(Rm_obs_system), dtype=bool)
        mask_res32_system = np.zeros(len(Rm_obs_system), dtype=bool)
        mask_res21_system = np.zeros(len(Rm_obs_system), dtype=bool)

        for ratio in gen.res_ratios:
            mask_res_system[(Rm_obs_system >= ratio) & (Rm_obs_system <= ratio*(1.+gen.res_width))] = 1

        mask_res32_system[(Rm_obs_system >= 1.5) & (Rm_obs_system <= 1.5*(1.+gen.res_width))] = 1
        mask_res21_system[(Rm_obs_system >= 2.) & (Rm_obs_system <= 2.*(1.+gen.res_width))] = 1

        xi_res_obs_system = list(xi_obs_system[mask_res_system])
        xi_res32_obs_system = list(xi_obs_system[mask_res32_system])
        xi_res21_obs_system = list(xi_obs_system[mask_res21_system])
        xi_nonres_obs_system = list(xi_obs_system[~mask_res_system])
        xi_res_obs_system = np.array(xi_res_obs_system + [-1]*(pad_extra - len(xi_res_obs_system)))
        xi_res32_obs_system = np.array(xi_res32_obs_system + [-1]*(pad_extra - len(xi_res32_obs_system)))
        xi_res21_obs_system = np.array(xi_res21_obs_system + [-1]*(pad_extra - len(xi_res21_obs_system)))
        xi_nonres_obs_system = np.array(xi_nonres_obs_system + [-1]*(pad_extra - len(xi_nonres_obs_system)))
        xi_res_obs.append(xi_res_obs_system)
        xi_res32_obs.append(xi_res32_obs_system)
        xi_res21_obs.append(xi_res21_obs_system)
        xi_nonres_obs.append(xi_nonres_obs_system)

        #To separate the planets in the system as above and below the boundary:
        system_above_bools = np.array([gen.photoevap_boundary_Carrera2018(radii_obs_system[x], P_obs_system[x]) for x in range(len(P_obs_system))])
        #if len(system_above_bools) > 1:
        #print(system_above_bools)

        #To record the transit depths of the planets above and below the boundary:
        for j in range(len(D_obs_system)):
            D_above_obs_flat.append(D_obs_system[j]) if system_above_bools[j] == 1 else D_below_obs_flat.append(D_obs_system[j])

        #To record the transit depth ratios of the planets above, below, and across the boundary:
        D_ratio_obs_system = list(compute_ratios(D_obs_system)) #list of transit depth ratios observed in this system
        if compute_ratios == gen.compute_ratios_adjacent:
            for j in range(len(D_ratio_obs_system)):
                if system_above_bools[j] + system_above_bools[j+1] == 2: #both planets are above the boundary
                    D_ratio_above_obs_flat.append(D_ratio_obs_system[j])
                elif system_above_bools[j] + system_above_bools[j+1] == 1: #one planet is above, the other is below the boundary
                    D_ratio_across_obs_flat.append(D_ratio_obs_system[j])
                elif system_above_bools[j] + system_above_bools[j+1] == 0: #both planets are below the boundary
                    D_ratio_below_obs_flat.append(D_ratio_obs_system[j])
        elif compute_ratios == gen.compute_ratios_all:
            ncum_ratio_j = np.concatenate((np.array([0]),np.cumsum(np.arange(len(D_obs_system))[::-1][:-2])))
            for j in range(len(D_ratio_obs_system)):
                for k in range(j+1,len(D_obs_system)):
                    i_ratio = ncum_ratio_j[j] + k - 1 - j
                    if system_above_bools[j] + system_above_bools[k] == 2: #both planets are above the boundary
                        D_ratio_above_obs_flat.append(D_ratio_obs_system[i_ratio])
                    elif system_above_bools[j] + system_above_bools[k] == 1: #one planet is above, the other is below the boundary
                        D_ratio_across_obs_flat.append(D_ratio_obs_system[i_ratio])
                    elif system_above_bools[j] + system_above_bools[k] == 0: #both planets are below the boundary
                        D_ratio_below_obs_flat.append(D_ratio_obs_system[i_ratio])

    Rm_obs = np.array(Rm_obs)
    D_ratio_obs = np.array(D_ratio_obs)
    xi_obs = np.array(xi_obs)
    xi_res_obs = np.array(xi_res_obs)
    xi_res32_obs = np.array(xi_res32_obs)
    xi_res21_obs = np.array(xi_res21_obs)
    xi_nonres_obs = np.array(xi_nonres_obs)

    D_above_obs_flat = np.array(D_above_obs_flat)
    D_below_obs_flat = np.array(D_below_obs_flat)
    D_ratio_above_obs_flat = np.array(D_ratio_above_obs_flat)
    D_ratio_below_obs_flat = np.array(D_ratio_below_obs_flat)
    D_ratio_across_obs_flat = np.array(D_ratio_across_obs_flat)

    # Create dictionaries to hold summary stats ('sss' stands for 'summary stats simulated'):

    sss_per_sys = {}
    # Stellar properties:
    sss_per_sys['Mstar_obs'] = Mstar_per_sys
    sss_per_sys['Rstar_obs'] = Rstar_per_sys
    sss_per_sys['teff_obs'] = teff_per_sys
    sss_per_sys['bp_rp_obs'] = bp_rp_per_sys
    sss_per_sys['e_bp_rp_obs'] = e_bp_rp_per_sys
    sss_per_sys['cdpp4p5_obs'] = cdpp4p5_per_sys
    # Planet properties:
    sss_per_sys['Mtot_obs'] = Mtot_obs
    sss_per_sys['P_obs'] = P_obs
    sss_per_sys['D_obs'] = D_obs
    sss_per_sys['tdur_obs'] = tdur_obs
    sss_per_sys['tdur_tcirc_obs'] = tdur_tcirc_obs
    sss_per_sys['radii_obs'] = radii_obs
    # Planet property ratios:
    sss_per_sys['Rm_obs'] = Rm_obs
    sss_per_sys['D_ratio_obs'] = D_ratio_obs
    sss_per_sys['xi_obs'] = xi_obs
    sss_per_sys['xi_res_obs'] = xi_res_obs
    sss_per_sys['xi_res32_obs'] = xi_res32_obs
    sss_per_sys['xi_res21_obs'] = xi_res21_obs
    sss_per_sys['xi_nonres_obs'] = xi_nonres_obs

    sss = {}
    # Stellar properties (repeated to match number of planets):
    sss['Mstar_obs'] = Mstar_obs
    sss['Rstar_obs'] = Rstar_obs
    sss['teff_obs'] = teff_obs
    sss['bp_rp_obs'] = bp_rp_obs
    sss['e_bp_rp_obs'] = e_bp_rp_obs
    sss['cdpp4p5_obs'] = cdpp4p5_obs
    # Planet multiplicities:
    sss['Nmult_obs'] = Nmult_obs
    # Planet properties:
    sss['P_obs'] = P_obs[P_obs > 0]
    sss['D_obs'] = D_obs[D_obs > 0]
    sss['tdur_obs'] = tdur_obs[tdur_obs >= 0] # our durations can be zero
    sss['tdur_tcirc_obs'] = tdur_tcirc_obs[tdur_tcirc_obs >= 0] # can also be zero
    sss['tdur_tcirc_1_obs'] = tdur_tcirc_1_obs
    sss['tdur_tcirc_2p_obs'] = tdur_tcirc_2p_obs
    sss['radii_obs'] = radii_obs[radii_obs > 0]
    sss['D_above_obs'] = D_above_obs_flat
    sss['D_below_obs'] = D_below_obs_flat
    # Planet property ratios:
    sss['Rm_obs'] = Rm_obs[Rm_obs > 0]
    sss['D_ratio_obs'] = D_ratio_obs[D_ratio_obs > 0]
    sss['xi_obs'] = xi_obs[xi_obs >= 0]
    sss['xi_res_obs'] = xi_res_obs[xi_res_obs >= 0]
    sss['xi_res32_obs'] = xi_res32_obs[xi_res32_obs >= 0]
    sss['xi_res21_obs'] = xi_res21_obs[xi_res21_obs >= 0]
    sss['xi_nonres_obs'] = xi_nonres_obs[xi_nonres_obs >= 0]
    sss['D_ratio_above_obs'] = D_ratio_above_obs_flat
    sss['D_ratio_below_obs'] = D_ratio_below_obs_flat
    sss['D_ratio_across_obs'] = D_ratio_across_obs_flat

    # To compute some summary stats (system-level metrics) from GF2020:
    Nsys_obs = len(Mtot_obs)
    assert Nsys_obs == len(radii_obs) == len(P_obs)

    radii_star_ratio = []
    radii_partitioning = []
    radii_monotonicity = []
    gap_complexity = []
    for i in range(Nsys_obs):
        P_obs_system = P_obs[i][P_obs[i] > 0]
        radii_obs_system = radii_obs[i][P_obs[i] > 0]

        radii_star_ratio.append(gen.radii_star_ratio(radii_obs_system, Rstar_per_sys[i]))

        if len(P_obs_system) >= 2:
            radii_partitioning.append(gen.partitioning(radii_obs_system))
            radii_monotonicity.append(gen.monotonicity_GF2020(radii_obs_system))
        if len(P_obs_system) >= 3:
            gap_complexity.append(gen.gap_complexity_GF2020(P_obs_system))
    sss_per_sys['radii_star_ratio'] = np.array(radii_star_ratio)
    sss_per_sys['radii_partitioning'] = np.array(radii_partitioning)
    sss_per_sys['radii_monotonicity'] = np.array(radii_monotonicity)
    sss_per_sys['gap_complexity'] = np.array(gap_complexity)

    return [sss_per_sys, sss]





# Functions to combine catalogs:

def combine_sss_or_sssp_per_sys(s1, s2):
    """
    Combine two dictionaries of summary statistics (e.g., two simulated catalogs).

    Note
    ----
    Requires both dictionaries to have the exact same fields.

    Parameters
    ----------
    s1, s2 : dict
        A dictionary of summary statistics.

    Returns
    -------
    scombined : dict
        The combined dictionary of summary statistics.
    """
    assert s1.keys() == s2.keys()

    scombined = {}
    for key in s1.keys():
        #print(key, ': ', np.shape(s1[key]), ', ', np.shape(s2[key]))

        if key == 'Nmult_obs':
            m_max = max(len(s1[key]), len(s2[key])) # highest multiplicity of the two catalogs
            Nmult_obs1 = np.append(s1[key], np.zeros(m_max - len(s1[key]), dtype=int))
            Nmult_obs2 = np.append(s2[key], np.zeros(m_max - len(s2[key]), dtype=int))
            scombined[key] = Nmult_obs1 + Nmult_obs2
            continue

        # For 2d arrays, need to pad them equally before concatenating:
        if np.ndim(s1[key]) == 2:
            npad1, npad2 = np.shape(s1[key])[1], np.shape(s2[key])[1]
            npad = max(npad1, npad2)
            if npad1 != npad2:
                s1_key_pad = np.concatenate((s1[key], -1.*np.ones((len(s1[key]),npad-npad1))), axis=1)
                s2_key_pad = np.concatenate((s2[key], -1.*np.ones((len(s2[key]),npad-npad2))), axis=1)
                scombined[key] = np.concatenate((s1_key_pad, s2_key_pad), axis=0)
                continue

        scombined[key] = np.concatenate((s1[key], s2[key]), axis=0)
    return scombined

def load_cat_phys_multiple_and_compute_combine_summary_stats(file_name_path, run_numbers=range(1,11), load_full_tables=False, compute_ratios=gen.compute_ratios_adjacent, match_observed=True):
    """
    Load multiple simulated physical catalogs and compute detailed summary statistic for all the catalogs combined.

    Parameters
    ----------
    file_name_path : str
        The path to the physical catalogs.
    run_number : range, default=range(1,11)
        The range of catalog run numbers over which we want to load and combine.
    load_full_tables : bool, default=False
        Whether to load full tables of the physical catalogs. Required to be True if also want to match the physical planets to the observed planets.
    compute_ratios : func, default=compute_ratios_adjacent
        The function to use for computing ratios; can be either :py:func:`syssimpyplots.general.compute_ratios_adjacent` or :py:func:`syssimpyplots.general.compute_ratios_all`.
    match_observed : bool, default=True
        Whether to match the physical planets to the observed planets. If True, the output will also contain a field `det_all`.

    Returns
    -------
    sssp_per_sys : dict
        A dictionary containing the planetary and stellar properties for each system (2-d and 1-d arrays).
    sssp : dict
        A dictionary containing the planetary and stellar properties of all planets (1-d arrays).


    The fields of ``sssp_per_sys`` and ``sssp`` are the same as those returned by :py:func:`syssimpyplots.load_sims.compute_summary_stats_from_cat_phys`.
    """
    print('Attempting to load %s physical catalogs to compute and combine their summary statistics...' % len(run_numbers))

    sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=file_name_path, run_number=run_numbers[0], load_full_tables=load_full_tables, compute_ratios=compute_ratios, match_observed=match_observed)
    for i in run_numbers[1:]:
        print(i)
        sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=file_name_path, run_number=i, load_full_tables=load_full_tables, compute_ratios=compute_ratios, match_observed=match_observed)

        sssp_per_sys = combine_sss_or_sssp_per_sys(sssp_per_sys, sssp_per_sys_i)
        sssp = combine_sss_or_sssp_per_sys(sssp, sssp_i)

    return sssp_per_sys, sssp
