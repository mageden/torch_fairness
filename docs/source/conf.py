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
    # Add when examples are ready
    "sphinx_gallery.gen_gallery",
    # 'sphinx.ext.autosectionlabel',
    "sphinx.ext.mathjax",
    # 'sphinx.ext.ifconfig',
    # 'sphinx.ext.githubpages'
]

# This sets the name of a function/class to not include module (e.g., Covariance, not torch_fairness.covariance.Covariance)
add_module_names = False

templates_path = ["_templates"]

# These are related to NB
exclude_patterns = ["build", "**.ipynb_checkpoints"]

# ---- Sphinx Gallary ----
sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": r"..\..\examples",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    "image_scrapers": ("matplotlib",),
}

# ---- Napoleon settings ----
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = True
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True

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
