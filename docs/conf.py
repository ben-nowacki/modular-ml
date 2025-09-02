import os
import sys
# If your code lives in src/your_package/, keep this:
sys.path.insert(0, os.path.abspath(".."))

# If it lives directly at repo-root/your_package/, use this instead:
# sys.path.insert(0, os.path.abspath(".."))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ModularML'
copyright = '2025, Benjamin Nowacki, Tingkai Li, and Chao Hu'
author = 'Benjamin Nowacki, Tingkai Li, and Chao Hu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = [
    "torch", "tensorflow", "pandas", "numpy", "scipy", "sklearn",
    "skopt", "optuna", "networkx", "plotly", "matplotlib", "joblib"
]
# Show both the class docstring and the __init__ docstring
autoclass_content = "both"
# If you use Google/NumPy style via napoleon, include __init__ in class docs
napoleon_include_init_with_doc = True
# Good defaults so class/module pages aren’t empty
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# --- Notebook execution settings ---
# Start simple: don't execute notebooks during build (fast & predictable).
nb_execution_mode = "off"  # change to "auto" later if you want live runs
# Optional niceties:
nb_execution_timeout = 120
nb_merge_streams = True  # cleaner stdout/stderr in output

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "style_external_links": True,
}

html_context = {
    "display_github": True,  # enables the GitHub icon
    "github_user": "REIL-UConn",
    "github_repo": "modular-ml",
    "github_version": "main",
    "conf_py_path": "/docs/",
}