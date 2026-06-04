# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Magenta RealTime 2'
copyright = '2026 Google LLC'  # pylint: disable=redefined-builtin
author = 'Google LLC'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'README.md',
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_title = 'Magenta RealTime'
html_static_path = ['_static']

# Theme-specific options
# https://sphinx-book-theme.readthedocs.io/en/stable/reference.html
html_theme_options = {
    'show_navbar_depth': 1,
    'show_toc_level': 3,
    'repository_url': 'https://github.com/magenta/magenta-realtime',
    'use_issues_button': True,
    'use_repository_button': True,
    'path_to_docs': 'docs/',
    'navigation_with_keys': True,
}

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = ['colon_fence', 'linkify']

# We ship plain Markdown docs only; don't execute notebooks at build time.
nb_execution_mode = 'off'
