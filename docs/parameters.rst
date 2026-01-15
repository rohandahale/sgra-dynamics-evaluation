==========
Parameters
==========

This page documents all configuration parameters for the evaluation pipeline.

Configuration File Format
-------------------------

The configuration uses YAML format:

.. code-block:: yaml

   # params.yml
   parameter_name: value
   nested:
     sub_parameter: value

Core Parameters
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``data``
     - string
     - Path to the UVFITS observation file
   * - ``truth``
     - string
     - Path to the ground truth HDF5 movie (optional for real data)
   * - ``input``
     - list
     - Glob pattern(s) for reconstruction HDF5 files
   * - ``outpath``
     - string
     - Output directory for results
   * - ``ncores``
     - int
     - Number of parallel workers (default: 32)

Time Selection
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``tstart``
     - float
     - Start time in UT hours (optional)
   * - ``tstop``
     - float
     - End time in UT hours (optional)

Metric Selection
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``metrics``
     - list
     - List of metrics to compute

Available metrics:

- ``chisq``: Chi-squared fitting to visibilities
- ``nxcorr``: Normalized cross-correlation
- ``rex``: Ring extraction (diameter, width, asymmetry)
- ``patternspeed``: Pattern speed from autocorrelation
- ``hotspot``: Hotspot tracking
- ``vida``: VIDA polarimetric analysis
- ``visualize``: Variance maps

Threshold Parameters
--------------------

Each metric has configurable thresholds:

Chi-squared Thresholds
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   chisq:
     flag_sites: ["AA", "AP"]  # Baselines to flag
     cp_uv_min: 1e8            # Minimum UV distance

NXCORR Thresholds
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   nxcorr:
     evpa_threshold: 20.0      # EVPA tolerance in degrees
     blur_threshold: true      # Use adaptive blur threshold

REx Thresholds
^^^^^^^^^^^^^^

.. code-block:: yaml

   rex:
     diameter_tol: 5.0         # μas
     width_tol: 5.0            # μas
     pa_tol: 26.0              # degrees
     relative_tol: 0.10        # 10% for flux quantities

Hotspot Thresholds
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   hotspot:
     position_tol: 5.0         # μas
     angle_tol: 20.0           # degrees
     flux_tol: 0.25            # 25% relative

Pattern Speed Thresholds
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   patternspeed:
     f_pipe:
       truth_hotspot: 2.0
       truth_turbulence: 3.0
       modeling: 0.6
       resolve: 0.2
       ehtim: 0.6
