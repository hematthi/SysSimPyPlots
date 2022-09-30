Welcome to SysSimPyPlots's documentation!
=========================================

**SysSimPyPlots** is a codebase for loading, analyzing, and plotting catalogs generated from the SysSim models.

.. tip:: What's behind the name?

   **SysSim** -- a comprehensive forward modeling framework for studying planetary systems based on the *Kepler* mission. In particular, `SysSimExClusters <https://github.com/ExoJulia/SysSimExClusters>`_ provides clustered planetary system models that characterize the underlying occurrence and intra-system correlations of multi-planet systems.

   **Py** -- this package is written almost entirely in Python 3. This is unlike the SysSim codebase which is written in `Julia <https://julialang.org>`_.

   **Plots** -- while the SysSim codebase provides the workhorse for simulating catalogs of planetary systems and the *Kepler* mission, this package allows you to plot, visualize, and explore those catalogs!

.. note::

   Feel free to `create an issue on Github <https://github.com/hematthi/SysSimPyPlots/issues>`_ if you find any problems.


Contents
--------

.. toctree::
   :maxdepth: 2
   :includehidden:

   installation
   loading
   histograms
   distances
   galleries
   api


.. _publications:

Publications
------------

The scripts in this package have been used to produce the figures and results in several papers:

- `Architectures of Exoplanetary Systems. I: A Clustered Forward Model for Exoplanetary Systems around Kepler's FGK Stars <https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.4575H/abstract>`_

  **Matthias Y. He**, Eric B. Ford, Darin Ragozzine, 2019, MNRAS, 490, 4575-4605

- `Architectures of Exoplanetary Systems. II: An Increase in Inner Planetary System Occurrence Towards Later Spectral Types for Kepler's FGK Dwarfs <https://ui.adsabs.harvard.edu/abs/2021AJ....161...16H/abstract>`_

  **Matthias Y. He**, Eric B. Ford, Darin Ragozzine, 2021a, AJ, 161, 16-40

- `Architectures of Exoplanetary Systems. III: Eccentricity and Mutual Inclination Distributions of AMD-stable Planetary Systems <https://ui.adsabs.harvard.edu/abs/2020arXiv200714473H/abstract>`_

  **Matthias Y. He**, Eric B. Ford, Darin Ragozzine, Daniel Carrera, 2020, AJ, 160, 276-314

- `Friends and Foes: Conditional Occurrence Rates of Exoplanet Companions and their Impact on Radial Velocity Follow-up Surveys <https://ui.adsabs.harvard.edu/abs/2021arXiv210504703H/abstract>`_

  **Matthias Y. He**, Eric B. Ford, Darin Ragozzine, 2021b, AJ, 162, 216-238
