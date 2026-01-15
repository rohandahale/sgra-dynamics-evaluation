=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
^^^^^
- Sphinx documentation with auto-build on GitHub Pages
- Comprehensive API reference with autodoc

[0.1.0] - 2025-01-15
--------------------

Initial release with core evaluation functionality.

Added
^^^^^
- Chi-squared analysis (``chisq.py``)
- Normalized cross-correlation (``nxcorr.py``)
- Ring extraction (``rex.py``)
- Hotspot tracking (``hotspot.py``)
- Pattern speed analysis (``patternspeed.py``, ``patternspeed_v2.py``)
- Visibility variance visualization (``visualize.py``)
- VIDA polarimetric analysis (``vida_pol.py``, ``vida_pol.jl``)
- HDF5 preprocessing utilities
- Bayesian mode support for all metrics
- Parallel processing with configurable worker count
- YAML-based configuration system

Documentation
^^^^^^^^^^^^^
- README with installation and usage instructions
- Mathematical documentation for all metrics
- Example parameter files
