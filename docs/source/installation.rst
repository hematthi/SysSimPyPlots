Installation
============


Using pip
---------

You can install the most recent stable version of SysSimPyPlots using `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: bash

   python -m pip install syssimpyplots


From source
-----------

All of the code is publicly available on `Github <https://github.com/hematthi/SysSimPyPlots>`_. Thus an alternative to pip is to download or clone the repository and install it from there:

.. code-block:: bash

   git clone https://github.com/hematthi/SysSimPyPlots.git
   python -m pip install .

There are multiple ways of cloning; if you have trouble with the above or prefer another, see `this Github guide <https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories>`_. You should also fork the repository first if you want to make your own changes to the source code later.


.. _downloading_catalogs:

Downloading simulated catalogs
------------------------------

After you have installed SysSimPyPlots, you will still need to download some simulated catalogs in order to explore any models. You can download a single, representative catalog generated from the latest model (the "maximum AMD model" described in `He et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020arXiv200714473H/abstract>`_) using `this link <https://drive.google.com/drive/folders/18-PMwzqDeCDQtXStAG4u_EWTOx-T4gML?usp=sharing>`_.

.. note::

   This simulated catalog contains five times as many stars as the Kepler catalog in our sample, and is about 450 MB in size.

If you want to perform more robust analyses, you can download many more simulated catalogs from the `SysSimExClusters Simulated Catalog folder <https://drive.google.com/drive/folders/1KGY7E1fK48O9mDuF6mDDhpXlyErkkd7t?usp=sharing>`_.

.. note::

   This folder contains 100 individual catalogs, each with the same number of stars as the Kepler catalog in our sample. The total file size is about 3.5 GB compressed, and roughly 8.2 GB when opened/uncompressed.

More details are provided in the `SysSimExClusters repository <https://github.com/ExoJulia/SysSimExClusters>`_. Check the READMEs of the individual branches for each paper.

That's it -- you are now ready to use SysSimPyPlots!


Dependencies
------------

SysSimPyPlots has been tested on Python >3.7 and uses:

- ``numpy`` (for almost everything)
- ``matplotlib`` (for making plots)
- ``scipy`` (for some miscellaneous functions)
- ``corner.py`` (for plotting multi-dimensional parameter spaces; see `how to install <https://corner.readthedocs.io/en/latest/install/>`_)


If you want to go further
-------------------------

If you want to simulate even more additional or new catalogs, you will need to download `Julia <https://julialang.org/downloads/>`_ and install `SysSimExClusters <https://github.com/ExoJulia/SysSimExClusters>`_ which also requires installing `ExoplanetsSysSim <https://github.com/ExoJulia/SysSimExClusters>`_. Please check those pages for instructions.
