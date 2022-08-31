# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))  # Source code dir relative to this file



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SysSimPyPlots'
copyright = '2022, Matthias Yang He'
author = 'Matthias Yang He'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.duration',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.mathjax',
   'sphinx.ext.napoleon',
   'sphinx_toolbox.collapse',
]
autosummary_generate = True
add_module_names = False # whether to prepend module names to functions/objects
autodoc_member_order = 'bysource' # sort docs for members by the order in which they appear in the module; default is 'alphabetical'

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = [
   "numpy",
   "matplotlib",
   "scipy",
   "mpl_toolkits",
   "pandas",
   "corner",
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = [] # ['_static']
