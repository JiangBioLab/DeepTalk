# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DeepTalktest'
copyright = '2023, wenyiyang'
author = 'wenyiyang'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys
import os
import sphinx

extensions = [
    'sphinx_mdinclude',
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_size',
    'nbsphinx',
    'sphinx_gallery.load_style',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh-en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
master_doc = 'index'