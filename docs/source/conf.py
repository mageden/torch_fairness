import os
import sys
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- General configuration ---------------------------------------------------

project = "Torch Fairness"
copyright = "2022, Michael Geden"
author = "Michael Geden"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

# This sets the name of a function/class to not include module (e.g., Covariance, not torch_fairness.covariance.Covariance)
add_module_names = False

templates_path = ["_templates"]

# These are related to NB
exclude_patterns = ["build", "**.ipynb_checkpoints"]

# ---- Autodoc ----
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_docstring_signature = True

# ---- Autosummary ----
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = 'pydata_sphinx_theme'
html_static_path = ["_static"]

# ---- NUMPBY ----
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
