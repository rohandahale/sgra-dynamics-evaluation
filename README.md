# Evaluation and validation scripts for black hole video reconstructions

A unified driver to automate the evaluation pipeline for Sgr A* Dynamics project submissions.

## Installation

### 1. Create Conda Environment

Create and activate the conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate evaluation
```

### 2. Julia Package Installation (for VIDA metrics)

The `vida_pol.py` script uses Julia for polarimetric analysis. Install the required Julia packages:

```bash
cd src
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This reads the `src/Project.toml` and `src/Manifest.toml` to install packages including:
- VIDA
- VLBISkyModels
- Comrade
- Other optimization and data handling packages

### 3. Verify Installation

```bash
# Test Python environment
python -c "import ehtim; import numpy; import pandas; print('Python OK')"

# Test Julia packages
julia --project=src -e 'using VIDA; using VLBISkyModels; println("Julia OK")'
```

## Configuration

The entire run is configured via `params.yml`. Key parameters include:

- **Models**: List of models to process
- **Run Settings**: `data_band`, `scattering`, and `pipeline` (single choice per run)
- **Mode**: Set `is_bayesian: True` for multiple-sample inputs (glob pattern) or `False` for single HDF5 inputs
- **Steps**: Toggle specific evaluation modules (e.g., `chisq`, `nxcorr`, `vida`) in the `run_steps` section

## Usage

Run the evaluation pipeline:

```bash
python evaluate.py params.yml
```

## Output

Results are saved to: `<results_dir>/<model>_<pipeline>/`

The script copies the configuration file used into the results directory for reproducibility.

## Documentation

For detailed documentation on each evaluation module, see [docs.md](docs.md).