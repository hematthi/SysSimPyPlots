Loading catalogs
================

The data you download from following the :ref:`"Downloading simulated catalogs" section <downloading_catalogs>` contains a number of catalog pairs, each consisting of:

- a "physical catalog": a set of intrinsic, physical planetary systems (before any observations; contains properties like the true orbital periods, planet radii, etc.)
- an "observed catalog": a set of transiting and detected planet candidates derived from a physical catalog (after a Kepler-like mission; contains properties like the measured orbital periods, transit depths, etc.)


Loading physical catalogs
-------------------------

First, you need to import some required packages:

.. code-block:: python

   import numpy as np
   from syssimpyplots.general import *
   from syssimpyplots.load_sims import *

Then specify the path to where you saved your data and load it as follows, for example:

.. code-block:: python

   load_dir = '/path/to/a/simulated/catalog/' # replace with your path!

   sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=load_dir)

The function outputs two dictionary objects, which we have stored in ``sssp_per_sys`` and ``sssp``. They contain mostly the same information but summarized in different ways.

.. collapse:: What do they contain?

   ``sssp_per_sys`` includes the detailed properties of each individual planetary system. Most of its data fields are two-dimensional arrays, with the first dimension (i.e. indexing rows) running through the different systems and the second dimension (i.e. indexing columns) running through the different planets in a system. For example, ``sssp_per_sys['P_all']`` gives a 2-d array of orbital periods.

   .. warning::

      Each row is padded with zeros, since different systems have different numbers of planets.

   Some fields in ``sssp_per_sys`` are one-dimensional arrays, i.e. for system-level quantities such as the multiplicity of each system (``sssp_per_sys['Mtot_all']``).

   On the other hand, ``sssp`` contains only one-dimensional arrays, such as ``sssp['P_all']`` for the orbital periods of all the planets in the catalog. This loses information about which planet(s) belong to which system, but is very convenient for plotting histograms, or performing simple calculations like computing the median period or the number of planets with periods less than 10 days.

   For a complete list of all the data fields, see the documentation for the :py:func:`syssimpyplots.load_sims.compute_summary_stats_from_cat_phys` function.

|


Loading observed catalogs
-------------------------

The process for loading simulated observed catalogs is similar; after importing the packages and defining the path to the data as above, simply do:

.. code-block:: python

   sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=load_dir)

.. collapse:: What do they contain?

   Analogous to the dictionaries for the physical catalogs, ``sss_per_sys`` includes the detailed properties of each individual planetary system (mostly two-dimensional arrays), while ``sss`` includes only one-dimensional arrays. For example, ``sss_per_sys['P_obs']`` gives a 2-d array of the observed orbital periods, while ``sss['P_obs']`` gives the same periods as a 1-d array.

   .. warning::

      Again, each row in a 2-d array is padded with either zeros or negative ones, since different systems have different numbers of observed planets!

   For a complete list of all the data fields, see the documentation for the :py:func:`syssimpyplots.load_sims.compute_summary_stats_from_cat_phys` function.

|


Reading simulation parameters
-----------------------------

You may want to read the number of simulated targets and the period and radius bounds for the simulated planets, without loading the full catalog (which may take up to a minute for larger physical catalogs):

.. code-block:: python

   N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(load_dir + 'periods.out')

You may also want to read the parameters of the model that went into the simulation:

.. code-block:: python

   param_vals_all = read_sim_params(load_dir + 'periods.out')

In these examples, you can replace the ``periods.out`` file with any of the other simulation files -- they all have the same header information.


Loading the Kepler catalog
--------------------------

Analogous to the function for summarizing an observed catalog, there is also a function for loading and processing the real Kepler data:

.. code-block:: python

   ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max)

This function requires the arguments ``P_min``, ``P_max``, ``radii_min``, and ``radii_max`` for selecting a sample of exoplanets that is restricted to a given orbital period and planet radius range, in order to be comparable to the simulated planets -- parameters which are conveniently provided by the :py:func:`syssimpyplots.load_sims.read_targets_period_radius_bounds` function shown earlier.

The outputs stored in ``ssk_per_sys`` and ``ssk`` contain the same summary statistics as those in ``sss_per_sys`` and ``sss``, respectively.

.. hint::

   The variable names ``sss`` and ``ssk`` were chosen to stand for "summary statistics simulated" and "summary statistics Kepler", respectively (and ``sssp`` for "summary statistics simulated physical"). Of course, you are free to choose whatever variable names you prefer.

You are now ready to use the catalogs to explore the models!
