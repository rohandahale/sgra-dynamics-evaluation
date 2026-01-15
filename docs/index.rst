.. Sgr A* Dynamics Evaluation documentation master file

======================================
Sgr A* Dynamics Evaluation
======================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

**Evaluation tools for black hole video reconstructions for real and synthetic VLBI data.**

This package provides comprehensive metrics and visualization tools for evaluating 
dynamic reconstructions of Sgr A* from Event Horizon Telescope (EHT) observations.

.. note::
   
   This documentation is auto-generated from the source code and updated 
   with every push to the main branch.

----

Quick Start
-----------

.. code-block:: bash

   # Install from source
   git clone https://github.com/rohandahale/sgra-dynamics-evaluation.git
   cd sgra-dynamics-evaluation
   pip install -e .

   # Run an evaluation
   python evaluate.py --params params.yml

----

Features
--------

🔬 **Comprehensive Metrics**
   Chi-squared fitting, normalized cross-correlation, pattern speed analysis, and more.

📊 **Rich Visualization**
   Publication-quality plots with variance maps, time series, and comparison figures.

🌀 **Bayesian Support**
   Full support for analyzing multiple reconstruction samples with uncertainty quantification.

⚡ **Parallel Processing**
   Efficient multi-core processing for large datasets.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   parameters

.. toctree::
   :maxdepth: 2
   :caption: Theory & Methods

   theory

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
