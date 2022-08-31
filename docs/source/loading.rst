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

The function outputs two objects, which we have stored in ``sssp_per_sys`` and ``sssp``. They contain mostly the same information but summarized in different ways.

*More details on what these two objects contain...*


Loading observed catalogs
-------------------------

The process for loading simulated observed catalogs is similar; after importing the packages and defining the path to the data as above, simply do:

.. code-block:: python

   sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=load_dir)

*More details on what these two objects contain...*


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


.. note::

   Currently under construction!
