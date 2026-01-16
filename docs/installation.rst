============
Installation
============

Requirements
------------

- Python 3.9+
- Conda (recommended for environment management)
- Julia 1.10.9 (optional, for VIDA polarimetric analysis)


Conda Environment
-----------------

Create and activate the conda environment:

.. code-block:: bash

   git clone https://github.com/rohandahale/sgra-dynamics-evaluation.git
   cd sgra-dynamics-evaluation
   conda env create -f environment.yml
   conda activate evaluation


Julia Installation (Optional)
-----------------------------

Julia is only required if you plan to use ``vida_pol.py`` for polarimetric 
ring fitting. Skip this section if you don't need this feature.

Install Julia 1.10.9 via `juliaup <https://github.com/JuliaLang/juliaup>`_:

.. code-block:: bash

   curl -fsSL https://install.julialang.org | sh
   source ~/.bashrc
   juliaup add 1.10.9
   juliaup default 1.10.9

Install the required Julia packages:

.. code-block:: bash

   cd src
   julia --project=. -e 'using Pkg; Pkg.instantiate()'

This reads ``src/Project.toml`` and ``src/Manifest.toml`` to install:

- VIDA
- VLBISkyModels  
- Comrade
- Other optimization packages


Verify Installation
-------------------

.. code-block:: bash

   # Test Python environment
   python -c "import ehtim; import numpy; import pandas; print('Python OK')"

   # Test Julia packages (if installed)
   julia --project=src -e 'using VIDA; using VLBISkyModels; println("Julia OK")'
