=============
API Reference
=============

This section documents the available modules in the evaluation package.

.. note::

   Full API documentation with auto-generated docstrings will be available 
   once the package is properly structured as an installable module.
   For now, please refer to the source code and the :doc:`../theory` section 
   for detailed information about each module.

Available Modules
-----------------

The following modules are available in the ``src/`` directory:

Core Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``chisq.py``
     - Chi-squared analysis for closure quantities and polarization ratios
   * - ``nxcorr.py``
     - Normalized cross-correlation for image fidelity assessment
   * - ``rex.py``
     - Ring extraction using the REx algorithm
   * - ``hotspot.py``
     - Hotspot tracking and Gaussian fitting
   * - ``patternspeed.py``
     - Pattern speed analysis via autocorrelation
   * - ``patternspeed_v2.py``
     - Enhanced pattern speed with MCMC uncertainty estimation

Visualization & Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``visualize.py``
     - Visibility variance maps and plotting utilities
   * - ``vida_pol.py``
     - VIDA polarimetric analysis (Python interface)
   * - ``vida_pol.jl``
     - VIDA polarimetric analysis (Julia implementation)
   * - ``preprocess_hdf5.py``
     - HDF5 preprocessing and regridding
   * - ``mean_image_extraction.py``
     - Mean image and polarization extraction

Usage Example
-------------

Each module can be run as a command-line script:

.. code-block:: bash

   # Chi-squared analysis
   python src/chisq.py -d observation.uvfits --input "recon/*.h5" -o results/

   # Normalized cross-correlation  
   python src/nxcorr.py --truth truth.h5 --input "recon/*.h5" -o results/

   # Pattern speed analysis
   python src/patternspeed.py --truth truth.h5 --input "recon/*.h5" -o results/

See the :doc:`../usage` guide for more detailed examples.
