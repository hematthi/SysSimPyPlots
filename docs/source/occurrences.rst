Computing occurrence rates
==========================

.. note::

   This page is currently under construction!


With the *Architectures of Exoplanetary Systems IV* paper, SysSimPyPlots is now updated to include user-friendly functions for calculating planet occurrence rates from the existing models!

The multi-planet models enables us to compute both:

- planet occurrence rates (the mean number of planets per star), and
- fractions of stars with planets (the fraction of stars with at least one such planet).

These are computed from a large number of simulated catalogs from a given model by counting the number of planets within a given set of period, radius, and/or mass bounds provided by the user. Our functions described below also automatically calculates the uncertainties in the estimated quantities arising from uncertainties in the model parameters and stochastic noise during the simulation process.


A few simple examples
---------------------

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

To compute the occurrence rates, we can call the following function providing the list of simulated catalogs and simple bounds in period, radius, and/or mass. Some examples are given below:

:math:`R_p = 2`-:math:`4 R_\oplus` and :math:`P = 3`-:math:`100` days:

.. code-block:: python

   occurrence_all, fswp_all = compute_occurrence_rates_and_fswps_in_period_radius_mass_bounds_many_catalogs(sssp_per_sys_all, radius_bounds=(2,4), period_bounds=(3,100))

.. code-block:: console

    # Printed output:
    No mass bounds given; assuming the full simulation range of (0.1,1000.0) Earth masses.
    Occurrence rate = 0.422_{-0.030}^{+0.049}
    f_swp = 0.295_{-0.019}^{+0.023}


:math:`M_p = 50`-:math:`1000 M_\oplus` and :math:`P = 3`-:math:`11` days:

.. code-block:: python

   occurrence_all, fswp_all = compute_occurrence_rates_and_fswps_in_period_radius_mass_bounds_many_catalogs(sssp_per_sys_all, mass_bounds=(50,1000), period_bounds=(3,11))

.. code-block:: console

    # Printed output:
    No radius bounds given; assuming the full simulation range of (0.5,10.0) Earth radii.
    Occurrence rate = 0.008_{-0.004}^{+0.006}
    f_swp = 0.007_{-0.004}^{+0.005}


:math:`R_p = 0.8`-:math:`1.2 R_\oplus`, :math:`M_p = 0.65`-:math:`0.98 M_\oplus`, and :math:`P = 180`-:math:`270` days (e.g., Venus analogs):

.. code-block:: python

   occurrence_all, fswp_all = compute_occurrence_rates_and_fswps_in_period_radius_mass_bounds_many_catalogs(sssp_per_sys_all, radius_bounds=(0.8,1.2), mass_bounds=(0.65,0.98), period_bounds=(180,270))

.. code-block:: console

    # Printed output:
    Occurrence rate = 0.006_{-0.002}^{+0.003}
    f_swp = 0.006_{-0.002}^{+0.003}


The outputs of the function calls are an array of occurrence rates (``occurrence_all``) and an array of fractions of stars with planets (``fswp_all``), with each value in the arrays corresponding to a single simulated catalog. The function will also print the credible regions (central 68%) of each quantity.

.. tip::

   For bounds that are not provided, the full simulation range will be assumed.
