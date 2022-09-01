Plotting histograms
===================

.. note::

   Currently under construction!


On the previous page, you learned how to load a catalog (physical or observed) in the form of dictionaries containing various planetary (and stellar) properties (sometimes referred to as *summary statistics*). One of the most basic yet illuminating ways of visualizing a catalog is to plot histograms of the various properties. For example, you may want to look at the *marginal* distribution of orbital periods, planet radii, transit durations, etc. SysSimPyPlots provides several flexible functions for plotting histograms, as demonstrated below:

.. code-block:: python

   fig_size = (8,4) # size of each figure (width, height)

   # To plot a histogram of the observed multiplicities (number of planets per system):
   ax = plot_fig_counts_hist_simple(fig_size, [sss_per_sys['Mtot_obs']], [], x_min=0, x_max=7, y_max=2e3, x_llim=0.5, log_y=True, xlabel_text='Observed planets per system')

   # To plot a histogram of the observed orbital periods:
   ax = plot_fig_pdf_simple(fig_size, [sss['P_obs']], [], x_min=3., x_max=300., y_min=1e-3, y_max=0.1, log_x=True, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)')

The :py:func:`syssimpyplots.plot_catalogs.plot_fig_counts_hist_simple` function should be used for quantities taking on discrete, integer values, as it is designed to center each bin on an integer. The multiplicity distribution is a perfect example of this case!

For continuous distributions, the :py:func:`syssimpyplots.plot_catalogs.plot_fig_pdf_simple` function should be used.

.. tip::

   The two functions above are actually wrappers of the functions :py:func:`syssimpyplots.plot_catalogs.plot_panel_counts_hist_simple` and :py:func:`syssimpyplots.plot_catalogs.plot_panel_pdf_simple`, respectively, which do most of the work and create a single panel (requiring an axes subplot object to plot on) instead of a figure. These are useful for making multi-panel figures!


Plotting CDFs
-------------

Similarly, you can use the following functions to plot (empirical) cumulative distribution functions (CDFs):

.. code-block:: python

   # To plot a CDF of the observed multiplicities:
   ax = plot_fig_mult_cdf_simple(fig_size, [sss_per_sys['Mtot_obs']], [], y_min=0.6, y_max=1., xlabel_text='Observed planets per system')

   # To plot a CDF of the observed orbital periods:
   ax = plot_fig_cdf_simple(fig_size, [sss['P_obs']], [], x_min=3., x_max=300., log_x=True, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)')
