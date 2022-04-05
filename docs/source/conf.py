# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '4.4.0'

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories thtml_themeo sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../..'))
sys.path.insert(0, os.path.abspath('../../../..'))

import inspect
__location__ = os.path.join(os.getcwd(), os.path.dirname(
        inspect.getfile(inspect.currentframe())))

sys.path.insert(0, os.path.join(__location__, "../.."))
sys.path.insert(0, os.path.join(__location__, "../../.."))
sys.path.insert(0, os.path.join(__location__, "../../../.."))

# -- Project information -----------------------------------------------------

project = 'AutoX'
copyright = '2022, caihengxing'
author = 'caihengxing'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.autosummary', 'sphinx.ext.viewcode', 'sphinx.ext.coverage',
              'sphinx.ext.doctest', 'sphinx.ext.ifconfig', 'sphinx.ext.imgmath',
              'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']