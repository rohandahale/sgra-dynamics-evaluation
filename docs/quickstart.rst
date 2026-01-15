===========
Quick Start
===========

This guide will get you running your first evaluation in minutes.

Basic Usage
-----------

The main entry point is the ``evaluate.py`` script, which coordinates all 
evaluation metrics:

.. code-block:: bash

   python evaluate.py --params params.yml

Configuration File
------------------

Create a ``params.yml`` configuration file:

.. code-block:: yaml

   # Input data
   data: path/to/observation.uvfits
   truth: path/to/truth_movie.h5
   input:
     - path/to/reconstruction_*.h5
   
   # Output directory
   outpath: ./results
   
   # Time range (optional)
   tstart: 10.5
   tstop: 14.0
   
   # Parallel processing
   ncores: 16
   
   # Metrics to compute
   metrics:
     - chisq
     - nxcorr
     - rex
     - patternspeed
     - hotspot
     - vida

Example Workflow
----------------

1. **Prepare your data**

   .. code-block:: bash

      # Ensure your reconstruction HDF5 files are ready
      ls reconstructions/*.h5

2. **Configure the evaluation**

   .. code-block:: bash

      # Copy the example params file
      cp params.yml my_evaluation.yml
      
      # Edit paths and settings
      vim my_evaluation.yml

3. **Run the evaluation**

   .. code-block:: bash

      python evaluate.py --params my_evaluation.yml

4. **View results**

   .. code-block:: bash

      # Results are saved as CSV and PNG files
      ls results/

Output Files
------------

The evaluation produces:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``*_chisq.csv``
     - Chi-squared metrics over time
   * - ``*_nxcorr.csv``
     - Normalized cross-correlation values
   * - ``*_rex.csv``
     - Ring extraction parameters (diameter, width, etc.)
   * - ``*_patternspeed.csv``
     - Pattern speed measurements with uncertainties
   * - ``*_hotspot.csv``
     - Hotspot tracking results
   * - ``*.png``
     - Publication-quality visualization plots

Next Steps
----------

- Learn about all :doc:`parameters <parameters>` in detail
- Understand the :doc:`theory <theory>` behind each metric
- Explore the :doc:`API reference <api/index>` for programmatic use
