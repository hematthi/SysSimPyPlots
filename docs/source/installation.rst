Installation
============


Using pip
---------

.. attention::

   SysSimPyPlots will be pip installable soon! Stay tuned.


From source
-----------

All of the code is publicly available on `Github <https://github.com/hematthi/SysSimPyPlots>`_. Thus an alternative to pip is to download or clone the repository (make sure to fork it first if you might want to make your own changes) and install it from there:

.. code-block:: bash

   git clone git@github.com:hematthi/SysSimPyPlots.git
   python -m pip install .


Dependencies
------------

SysSimPyPlots has been tested on Python >3.7 and uses:

- ``numpy`` (for almost everything)
- ``matplotlib`` (for making plots)
- ``scipy`` (for some miscellaneous functions)
- ``corner`` (for plotting multi-dimensional parameter spaces)


Downloading simulated catalogs
------------------------------

You can download many simulated catalogs from the `SysSimExClusters Simulated Catalog folder <https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/myh7_psu_edu/Ei7QJqnmaCBGipPM4uMzrusBjw_hUwo0KfIDBe-0UTYyMw>`_. Each catalog pair consists of:

- a "physical catalog": a set of intrinsic, physical planetary systems (before any observations; contains properties like the true orbital periods, planet radii, etc.)
- an "observed catalog": a set of transiting and detected planet candidates derived from a *physical catalog* (after a Kepler-like mission; contains properties like the measured orbital periods, transit depths, etc.)

More details are provided in the `SysSimExClusters repository <https://github.com/ExoJulia/SysSimExClusters>`_. Check the READMEs of the individual branches for each paper.

That's it -- you are now ready to use SysSimPyPlots!


If you want to go further
-------------------------

If you want to simulate new or additional catalogs, you will need to download `Julia <https://julialang.org/downloads/>`_ and install `SysSimExClusters <https://github.com/ExoJulia/SysSimExClusters>`_ which also requires installing `ExoplanetsSysSim <https://github.com/ExoJulia/SysSimExClusters>`_. Please check those pages for instructions.
