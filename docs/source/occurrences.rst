Computing occurrence rates
==========================

.. note::

   This page is currently under construction!


With the *Architectures of Exoplanetary Systems IV* paper, SysSimPyPlots is now updated to include user-friendly functions for calculating planet occurrence rates from the existing models!

The multi-planet models enables us to compute both:

- planet occurrence rates (the mean number of planets per star), and
- fractions of stars with planets (the fraction of stars with at least one such planet).

These are computed from a large number of simulated catalogs from a given model by counting the number of planets within a given set of period, radius, and/or mass bounds provided by the user. Our functions described below also automatically calculates the uncertainties in the estimated quantities arising from uncertainties in the model parameters and stochastic noise during the simulation process.


A simple example
----------------

First, we need to load the simulated catalogs from a given model. We have a faster function for loading just the periods, radii, and masses from the physical catalogs, which are sufficient for computing simple occurrence rates:

.. code-block:: python

   # Load modules and catalogs as before:
   from syssimpyplots.general import *
   from syssimpyplots.load_sims import *
   from syssimpyplots.compute_occurrence import *

   load_dir = '/path/to/a/simulated/catalog/' # replace with your path!

   # Load a simulated physical catalog:
   sssp_per_sys_all = load_planets_periods_radii_masses_as_summary_stats_per_sys_many_fast(load_dir)


.. Note::

   The function :py:func:`load_planets_periods_radii_masses_as_summary_stats_per_sys_many_fast <syssimpyplots.load_sims.load_planets_periods_radii_masses_as_summary_stats_per_sys_many_fast>` will load the first 100 simulated catalogs in the directory provided by ``load_dir``, by default, but this can be changed by setting the optional argument ``runs=``.
   
   This is also much faster than loading each full catalog by repeatedly calling the function :py:func:`compute_summary_stats_from_cat_phys <syssimpyplots.load_sims.compute_summary_stats_from_cat_phys>`, as was done in the :ref:`"Loading physical catalogs" section <loading_physical_catalogs>`.

The output ``sssp_per_sys_all`` is a list of dictionaries (one for each simulated physical catalog), each containing the basic summary statistics (mainly things like ``P_all`` for all the orbital periods, ``radii_all`` for all the planet radii, and ``mass_all`` for all the planet masses).

To compute the occurrence rates, we can call the following function providing simple bounds in period, radius, and/or mass. For example, to compute the occurrence of planets between :math:`P = 3`-100 days and :math:`R_p = 2`-:math:`4 R_\oplus`:

.. code-block:: python

   occurrence_all, fswp_all = compute_occurrence_rates_and_fswps_in_period_radius_mass_bounds_many_catalogs(sssp_per_sys_all, period_bounds=(3,100), radius_bounds=(2,4))

The outputs are an array of occurrence rates (``occurrence_all``) and an array of fractions of stars with planets (``fswp_all``), with each value in the arrays corresponding to a single simulated catalog. The function will also print the credible regions (central 68%) of each quantity.

.. tip::

   For bounds that are not provided (the default is set to `None`), the full simulation range will be assumed. In the example above, no `mass_bounds` were provided and thus planets of all masses are included (as long as they still satisfy the period and radius bounds).
