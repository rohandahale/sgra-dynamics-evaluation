============
Contributing
============

We welcome contributions to the Sgr A* Dynamics Evaluation project!

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/YOUR-USERNAME/sgra-dynamics-evaluation.git
      cd sgra-dynamics-evaluation

3. Create a development environment:

   .. code-block:: bash

      conda env create -f environment.yml
      conda activate sgra-dynamics-eval
      pip install -e ".[dev,docs]"

Development Workflow
--------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes and add tests
3. Run the test suite:

   .. code-block:: bash

      pytest tests/

4. Format your code (we follow PEP 8)
5. Commit and push:

   .. code-block:: bash

      git add .
      git commit -m "Add your feature"
      git push origin feature/your-feature-name

6. Open a Pull Request on GitHub

Code Style
----------

- Follow PEP 8 for Python code
- Use NumPy-style docstrings
- Add type hints where possible
- Keep lines under 88 characters

Documentation
-------------

To build the documentation locally:

.. code-block:: bash

   cd docs
   make html
   
   # View in browser
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Adding New Metrics
------------------

When adding a new evaluation metric:

1. Create a new module in ``src/``
2. Add docstrings following NumPy style
3. Add tests in ``tests/``
4. Document the theory in ``docs/theory.rst``
5. Update the API reference in ``docs/api/index.rst``

Questions?
----------

Open an issue on GitHub or reach out to the maintainers!
