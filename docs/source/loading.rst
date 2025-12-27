Loading catalogs
================

The data you download from following the :ref:`"Downloading simulated catalogs" section <downloading_catalogs>` contains a number of catalog pairs, each consisting of:

- a "**physical catalog**": a set of intrinsic, physical planetary systems (before any observations; contains properties like the true orbital periods, planet radii, etc.)
- an "**observed catalog**": a set of transiting and detected planet candidates derived from a physical catalog (after a Kepler-like mission; contains properties like the measured orbital periods, transit depths, etc.)


.. _loading_physical_catalogs:

Loading physical catalogs
-------------------------

First, you need to import some required packages:

.. code-block:: python

   from syssimpyplots.general import *
   from syssimpyplots.load_sims import *

Then specify the path to where you saved your data and load it as follows, for example:

.. code-block:: python

   load_dir = '/path/to/a/simulated/catalog/' # replace with your path!

   cat_phys = load_cat_phys(load_dir + 'physical_catalog.csv')

This returns a table (which we stored in ``cat_phys``) with the properties of all the planets in the physical catalog, where each row corresponds to one planet. See the documentation for :py:func:`load_cat_phys <syssimpyplots.load_sims.load_cat_phys>` for a detailed description of the table columns.

While you could directly work with the ``cat_phys`` table, it would be convenient if the data was processed in some more useful ways. For example, you may want to know which planets belong to the same system, their period ratios, etc. You don't have to compute these yourself -- we've defined functions for computing these and *many* other summary statistics from the catalogs!

.. code-block:: python

   sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=load_dir)

(This may take a minute or two, depending on the size of the catalog.) The function above outputs two dictionary objects, which we have stored in ``sssp_per_sys`` and ``sssp``. They contain mostly the same information but summarized in different ways.

.. collapse:: What do they contain?

   ``sssp_per_sys`` includes the detailed properties of each individual planetary system. Most of its data fields are two-dimensional arrays, with the first dimension (i.e. indexing rows) running through the different systems and the second dimension (i.e. indexing columns) running through the different planets in a system. For example, ``sssp_per_sys['P_all']`` gives a 2-d array of orbital periods.

   .. warning::

      Each row is padded with zeros, since different systems have different numbers of planets.

   Some fields in ``sssp_per_sys`` are one-dimensional arrays, i.e. for system-level quantities such as the multiplicity of each system (``sssp_per_sys['Mtot_all']``).

   On the other hand, ``sssp`` contains only one-dimensional arrays, such as ``sssp['P_all']`` for the orbital periods of all the planets in the catalog. This loses information about which planet(s) belong to which system, but is very convenient for plotting histograms, or performing simple calculations like computing the median period or the number of planets with periods less than 10 days.

   For a complete list of all the data fields, see the documentation for the :py:func:`compute_summary_stats_from_cat_phys <syssimpyplots.load_sims.compute_summary_stats_from_cat_phys>` function.

|

Notice that we only passed the path to the simulated catalog, ``load_dir``, to the function ``compute_summary_stats_from_cat_phys()``. This tells it to load several ancillary files containing the same information as what's in "physical_catalog.csv", which is actually faster than loading the catalog itself using ``load_cat_phys()``. The function :py:func:`compute_summary_stats_from_cat_phys <syssimpyplots.load_sims.compute_summary_stats_from_cat_phys>` also accepts a physical catalog table (i.e. the ``cat_phys`` object) as input, but we recommend using the load directory path.


Loading observed catalogs
-------------------------

The process for loading simulated observed catalogs is similar; after importing the packages and defining the path to the data as above, simply do:

.. code-block:: python

   cat_obs = load_cat_obs(load_dir + 'observed_catalog.csv')

for loading a table with all of the observed planets, or

.. code-block:: python

   sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=load_dir)

for computing the summary statistics from the observed catalog.

.. collapse:: What do they contain?

   Analogous to the dictionaries for the physical catalogs, ``sss_per_sys`` includes the detailed properties of each individual planetary system (mostly two-dimensional arrays), while ``sss`` includes only one-dimensional arrays. For example, ``sss_per_sys['P_obs']`` gives a 2-d array of the observed orbital periods, while ``sss['P_obs']`` gives the same periods as a 1-d array.

   .. warning::

      Again, each row in a 2-d array is padded with either zeros or negative ones, since different systems have different numbers of observed planets!

   For a complete list of all the data fields, see the documentation for the :py:func:`compute_summary_stats_from_cat_obs <syssimpyplots.load_sims.compute_summary_stats_from_cat_obs>` function.

|

As before, we only passed the path to the simulated catalog to the function ``compute_summary_stats_from_cat_obs()``, which loads several ancillary files containing the same information instead of "observed_catalog.csv". You can also pass the ``cat_obs`` object into the function but we recommend the load directory path approach.


Reading simulation parameters
-----------------------------

You may want to read the number of simulated targets and the period and radius bounds for the simulated planets, without loading the full catalog (which may take several minutes for larger physical catalogs):

.. code-block:: python

   # Contains 'N_sim' (number of simulated planets), 'P_min' and 'P_max' (period bounds), etc.
   sim_settings = read_targets_period_radius_bounds(load_dir + 'periods.out')

You may also want to read the parameters of the model that went into the simulation:

.. code-block:: python

   param_vals_all = read_sim_params(load_dir + 'periods.out')

In these examples, you can replace the ``periods.out`` file with any of the other simulation files -- they all have the same header information.


Loading the Kepler catalog
--------------------------

Analogous to the functions for loading and summarizing an observed catalog, there are also functions for loading and processing the real Kepler data:

.. code-block:: python

   from syssimpyplots.compare_kepler import *

   koi_table = load_Kepler_planets_cleaned()

   ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max)

The function :py:func:`compute_summary_stats_from_Kepler_catalog <syssimpyplots.compare_kepler.compute_summary_stats_from_Kepler_catalog>` requires the arguments ``P_min``, ``P_max``, ``radii_min``, and ``radii_max`` for selecting a sample of exoplanets that is restricted to a given orbital period and planet radius range, in order to be comparable to the simulated planets -- parameters which are conveniently provided by the :py:func:`read_targets_period_radius_bounds <syssimpyplots.load_sims.read_targets_period_radius_bounds>` function shown earlier.

The outputs stored in ``ssk_per_sys`` and ``ssk`` contain the same summary statistics as those in ``sss_per_sys`` and ``sss``, respectively.

.. tip::

   The variable names ``sss`` and ``ssk`` were chosen to stand for "summary statistics simulated" and "summary statistics Kepler", respectively (and ``sssp`` for "summary statistics simulated physical"). Of course, you are free to choose whatever variable names you prefer.

You are now ready to use the catalogs to explore the models!
