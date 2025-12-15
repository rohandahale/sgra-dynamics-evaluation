# Evaluation and validation scripts for black hole video reconstructions

A unified driver to automate the evaluation pipeline for Sgr A* Dynamics project submissions.

## Setup

Ensure you are in the `evaluation` conda environment:
```bash
conda activate evaluation
```

## Configuration

The entire run is configured via `params.yml`. Key parameters include:
- **Models**: List of models to process.
- **Run Settings**: `data_band`, `scattering`, and `pipeline` (single choice per run).
- **Mode**: Set `is_bayesian: True` for multiple-sample inputs (glob pattern) or `False` for single HDF5 inputs.
- **Steps**: Toggle specific evaluation modules (e.g., `chisq`, `nxcorr`) in the `run_steps` section.

## Usage

To run the evaluation:

```bash
python evaluate.py params.yml
```

## Output

Results are saved to: `<results_dir>/<model>_<pipeline>/`

The script will also copy the configuration file used for the run into the results directory for reproducibility.