=====
Usage
=====

This section provides detailed usage information for all evaluation modules.

Command-Line Interface
----------------------

The main evaluation script supports the following arguments:

.. code-block:: bash

   python evaluate.py --params params.yml [OPTIONS]

Options:

- ``--params``: Path to YAML configuration file (required)
- ``--metrics``: Override which metrics to compute
- ``--ncores``: Number of parallel workers
- ``--verbose``: Enable verbose output

Running Individual Metrics
--------------------------

Each metric module can be run independently:

Chi-squared Analysis
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m src.chisq \
       -d observation.uvfits \
       --input "reconstructions/*.h5" \
       -o results/prefix \
       --ncores 16

Normalized Cross-Correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m src.nxcorr \
       --truth truth_movie.h5 \
       --input "reconstructions/*.h5" \
       --obs observation.uvfits \
       -o results/prefix

Pattern Speed
^^^^^^^^^^^^^

.. code-block:: bash

   python -m src.patternspeed \
       --truth truth_movie.h5 \
       --input "reconstructions/*.h5" \
       -o results/prefix

Bayesian Mode
-------------

When multiple reconstruction samples are provided (e.g., from MCMC posterior),
the tools automatically switch to Bayesian mode:

.. code-block:: yaml

   input:
     - reconstructions/sample_*.h5  # Multiple samples

In Bayesian mode:

- Mean and standard deviation are computed across samples
- Error bars are shown on plots
- Pass/fail criteria account for reconstruction uncertainty

Working with HDF5 Files
-----------------------

The package uses ehtim's HDF5 movie format:

.. code-block:: python

   import ehtim as eh
   
   # Load a movie
   movie = eh.movie.load_hdf5("reconstruction.h5")
   
   # Get frame at specific time
   frame = movie.get_image(time_utc=12.5)
   
   # Save as HDF5
   movie.save_hdf5("output.h5")

Output Format
-------------

All metrics output CSV files with consistent structure:

.. code-block:: text

   time,metric_value,metric_mean,metric_std,threshold,pass
   10.5,0.95,0.93,0.02,0.85,True
   10.6,0.91,0.92,0.03,0.84,True
   ...

Visualization outputs are saved as high-resolution PNG files (300 DPI).
