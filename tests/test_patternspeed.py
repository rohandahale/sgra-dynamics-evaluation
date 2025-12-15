import os
import sys
import subprocess

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_patternspeed_execution():
    """Test that patternspeed.py runs without error on sample data."""
    
    # Paths (relative to repo root, assuming test is run from repo root or tests dir)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base_dir)
    
    script_path = os.path.join(repo_root, 'src', 'patternspeed.py')
    
    data_dir = os.path.join(base_dir, 'data')
    uvfits_path = os.path.join(data_dir, 'mring+hsCW_LO_onsky.uvfits')
    truth_hdf5_path = os.path.join(data_dir, 'mring+hsCW_LO_onsky_truth.hdf5')
    # Output directory
    out_dir = os.path.join(repo_root, 'tests')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_prefix = os.path.join(out_dir, 'test_run')
    
    # Construct command
    cmd = [
        sys.executable, script_path,
        '-d', uvfits_path,
        '-i', truth_hdf5_path,
        '--truthmv', truth_hdf5_path,
        '--nsamples', '10',
        '-o', out_prefix
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run script
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"
    
    # Verify outputs
    expected_files = [
        out_prefix + '_patternspeed_summary.png',
        out_prefix + '_patternspeed_truth_smoothed_norm_cylinder.npy',
        out_prefix + '_patternspeed_truth_uncertainty.npy',
        out_prefix + '_patternspeed_truth_autocorr_xlim.npy'
    ]
    
    for f in expected_files:
        assert os.path.exists(f), f"Output file missing: {f}"
    
    print("Test passed successfully!")
    
    # Cleanup: Delete test outputs
    print("\nCleaning up test outputs...")
    suffixes = [
        '_patternspeed_summary.png', 
        '_patternspeed_truth_smoothed_norm_cylinder.npy',
        '_patternspeed_truth_uncertainty.npy',
        '_patternspeed_truth_autocorr_xlim.npy',
        '_patternspeed_truth_autocorr.npy',
        '_patternspeed_truth_autocorr_ylim.npy',
        '_patternspeed_truth_mcmc_samples.npy',
        '_patternspeed_recon_autocorr.npy',
        '_patternspeed_recon_smoothed_norm_cylinder.npy',
        '_patternspeed_recon_autocorr_xlim.npy',
        '_patternspeed_recon_autocorr_ylim.npy',
        '_patternspeed_recon_mcmc_samples.npy',
        '_patternspeed_recon_uncertainty.npy',
        '_patternspeed_stats.npy'
    ]
    
    for suffix in suffixes:
        fpath = out_prefix + suffix
        if os.path.exists(fpath):
            os.remove(fpath)

if __name__ == "__main__":
    test_patternspeed_execution()
