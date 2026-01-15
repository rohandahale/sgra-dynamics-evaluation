============
Installation
============

This guide covers how to install the Sgr A* Dynamics Evaluation package.

Requirements
------------

* Python 3.9 or higher
* Julia 1.6+ (for VIDA polarimetric analysis)
* Conda (recommended for environment management)

Using Conda (Recommended)
-------------------------

We recommend using Conda to manage the environment, especially for the 
scientific dependencies.

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/rohandahale/sgra-dynamics-evaluation.git
   cd sgra-dynamics-evaluation

   # Create conda environment
   conda env create -f environment.yml

   # Activate environment
   conda activate sgra-dynamics-eval

   # Install the package in development mode
   pip install -e .

Using pip
---------

If you prefer pip, you can install directly:

.. code-block:: bash

   pip install git+https://github.com/rohandahale/sgra-dynamics-evaluation.git

.. warning::
   
   Some dependencies like ``ehtim`` may require additional system libraries. 
   The Conda installation handles these automatically.

Development Installation
------------------------

For contributing or development:

.. code-block:: bash

   # Clone with full history
   git clone https://github.com/rohandahale/sgra-dynamics-evaluation.git
   cd sgra-dynamics-evaluation

   # Install with development dependencies
   pip install -e ".[dev,docs]"

   # Run tests
   pytest

Julia Setup
-----------

For the VIDA polarimetric analysis module, you need Julia installed:

.. code-block:: bash

   # Install Julia (if not already installed)
   # See: https://julialang.org/downloads/

   # Install required Julia packages (automatic on first run)
   julia -e 'using Pkg; Pkg.instantiate()'

Verification
------------

Verify your installation:

.. code-block:: python

   import src
   print(src.__version__)
   # Should print: 0.1.0

.. code-block:: bash

   # Or run the test suite
   pytest tests/
