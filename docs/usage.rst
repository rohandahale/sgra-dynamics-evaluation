=====
Usage
=====

This page explains how to configure and run the evaluation pipeline.

----

Quick Start
-----------

Run the evaluation pipeline with:

.. code-block:: bash

   python evaluate.py params.yml

The script reads all configuration from ``params.yml`` and processes the 
specified models through the enabled evaluation steps.

----

Configuration (params.yml)
--------------------------

The entire pipeline is configured via a single YAML file. Below is a 
complete reference of all parameters.


Directories
^^^^^^^^^^^

.. code-block:: yaml

   submission_dir: "/path/to/submissions"
   results_dir: "/path/to/results"

- **submission_dir**: Directory containing input files (UVFITS observations, 
  HDF5 reconstructions, truth movies)
- **results_dir**: Directory where output files will be saved


Computational Resources
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   ncores: 32

Number of CPU cores for parallel processing.


Time Range
^^^^^^^^^^

.. code-block:: yaml

   tstart: null
   tstop: null

- **tstart**: Start time in UT hours (e.g., ``10.5``). Set to ``null`` for 
  full range.
- **tstop**: Stop time in UT hours. Set to ``null`` for full range.


Overwrite
^^^^^^^^^

.. code-block:: yaml

   overwrite: False

If ``True``, existing results will be overwritten. If ``False``, existing 
files are skipped.


Models
^^^^^^

.. code-block:: yaml

   models:
     - "crescent"
     - "grmhd2"
     - "mring+hsCW"

List of model names to evaluate. The script constructs file paths using 
these names with the format templates below.

Available models include:

- Geometric: ``crescent``, ``disk``, ``double``, ``edisk``, ``ring``, ``point``
- GRMHD: ``grmhd1`` through ``grmhd8``
- Hotspot: ``mring+hsCW``, ``mring+hsCCW``, ``mring+hs-*`` variants
- Calibrators: ``J1924-2914``, ``SGRA``


Data Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   data_band: "LO"
   scattering: "onsky"

- **data_band**: Frequency band of observation data
  
  - ``LO``: Low band
  - ``HI``: High band
  - ``LO+HI``: Combined

- **scattering**: Scattering treatment
  
  - ``onsky``: As observed (with scattering)
  - ``dsct``: Descattered
  - ``deblur``: Deblurred


File Format Templates
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   data_format: "{model}_{data_band}_{scattering}.uvfits"
   truth_format: "{model}_{data_band}_{scattering}_truth.hdf5"

Templates for constructing file paths. Available placeholders:

- ``{model}``: Model name from the list
- ``{data_band}``: From ``data_band`` setting
- ``{scattering}``: From ``scattering`` setting
- ``{recon_band}``: From ``recon_band`` setting
- ``{pipeline}``: From ``pipeline`` setting


Reconstruction Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   recon_band: "LO+HI"
   pipeline: "resolve"
   is_bayesian: True

- **recon_band**: Frequency band used for reconstruction
- **pipeline**: Reconstruction pipeline name (``kine``, ``resolve``, 
  ``ehtim``, ``doghit``, ``ngmem``, ``modeling``)
- **is_bayesian**: 
  
  - ``True``: Expects multiple HDF5 samples (uses glob pattern)
  - ``False``: Expects single HDF5 file


Reconstruction File Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # For is_bayesian: False
   recon_format_non_bayesian: "{model}_{recon_band}_{scattering}_{pipeline}.hdf5"

   # For is_bayesian: True
   recon_format_bayesian: "{model}_{recon_band}_{scattering}_{pipeline}_0*"
   recon_format_bayesian_mean: "{model}_{recon_band}_{scattering}_{pipeline}_mean.hdf5"

- **recon_format_non_bayesian**: Single reconstruction file
- **recon_format_bayesian**: Glob pattern for multiple samples (e.g., 
  ``*_001.hdf5``, ``*_002.hdf5``, ...)
- **recon_format_bayesian_mean**: Mean reconstruction for visualization


Evaluation Steps
^^^^^^^^^^^^^^^^

.. code-block:: yaml

   run_steps:
     preprocess_hdf5: True
     chisq: True
     hotspot: True
     nxcorr: True
     patternspeed: True
     patternspeed_v2: True
     rex: True
     vida_pol: True
     vizualize: True

Enable/disable individual evaluation steps:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Step
     - Description
   * - ``preprocess_hdf5``
     - Regrid and prepare HDF5 files, extract static/dynamic components
   * - ``chisq``
     - Compute chi-squared metrics against observation data
   * - ``hotspot``
     - Track and compare hotspot features
   * - ``nxcorr``
     - Compute normalized cross-correlation metrics
   * - ``patternspeed``
     - Calculate pattern speed via autocorrelation
   * - ``patternspeed_v2``
     - Alternative pattern speed calculation
   * - ``rex``
     - Extract ring parameters using REx algorithm
   * - ``vida_pol``
     - Fit parametric ring models (requires Julia)
   * - ``vizualize``
     - Generate visualization outputs (movies, variance maps)

----

Output Structure
----------------

Results are saved to:

::

   <results_dir>/<model>_<pipeline>/

Each evaluation step produces:

- **CSV files**: Tabular results with metrics per time step
- **PNG plots**: Visualization of metrics over time
- **HDF5 files**: Preprocessed images (from ``preprocess_hdf5``)
- **GIF animations**: Movie visualizations (from ``vizualize``)

The configuration file used is copied to the results directory for 
reproducibility.


Example Output Files
^^^^^^^^^^^^^^^^^^^^

For a model ``crescent`` with pipeline ``resolve``:

::

   results/crescent_resolve/
   ├── crescent_resolve_chisq.csv
   ├── crescent_resolve_chisq.png
   ├── crescent_resolve_chisq_flagAAAP.csv
   ├── crescent_resolve_nxcorr.csv
   ├── crescent_resolve_nxcorr.png
   ├── crescent_resolve_hotspot.csv
   ├── crescent_resolve_rex.csv
   ├── crescent_resolve_rex.png
   ├── crescent_resolve_vida.csv
   ├── crescent_resolve_vida.png
   ├── crescent_resolve_patternspeed.csv
   ├── crescent_resolve_total.gif
   ├── crescent_resolve_lp.gif
   └── params.yml

----

Important Notes
---------------

.. warning::

   **Time Alignment**: Observation UVFITS and HDF5 movies must be 
   time-aligned. The script uses nearest-neighbor interpolation to match 
   observation times to movie frames.

.. warning::

   **Automatic Flagging**:
   
   - **Chi-squared**: AA (ALMA) and AP (APEX) baselines are automatically 
     flagged to avoid calibration artifacts
   - **Polarization chi-squared**: JC (JCMT) sites are additionally flagged

.. note::

   **Bayesian Mode**: When ``is_bayesian: True``, the script:
   
   - Loads all HDF5 files matching the glob pattern
   - Computes mean and standard deviation across samples
   - Reports pass percentages based on whether truth falls within 
     reconstruction uncertainty
   
.. note::

   **Pattern Speed**: Critical threshold factors are pipeline-specific. 
   The script automatically selects the appropriate factor based on the 
   ``pipeline`` setting.

.. note::

   **VIDA Polarimetric**: This step requires Julia to be installed with 
   the VIDA package. If Julia is not available, set ``vida_pol: False``.

----

Example Configuration
---------------------

Complete working example:

.. code-block:: yaml

   # Directories
   submission_dir: "/data/sgra/submissions"
   results_dir: "/data/sgra/results"

   # Resources
   ncores: 16
   tstart: 10.5
   tstop: 14.0
   overwrite: False

   # Models
   models:
     - "crescent"
     - "mring+hsCW"

   # Data
   data_band: "LO"
   scattering: "onsky"
   data_format: "{model}_{data_band}_{scattering}.uvfits"

   # Reconstruction
   recon_band: "LO+HI"
   pipeline: "resolve"
   is_bayesian: True
   recon_format_bayesian: "{model}_{recon_band}_{scattering}_{pipeline}_0*"
   recon_format_bayesian_mean: "{model}_{recon_band}_{scattering}_{pipeline}_mean.hdf5"

   # Truth
   truth_format: "{model}_{data_band}_{scattering}_truth.hdf5"

   # Steps
   run_steps:
     preprocess_hdf5: True
     chisq: True
     hotspot: True
     nxcorr: True
     patternspeed: True
     patternspeed_v2: False
     rex: True
     vida_pol: False
     vizualize: True
