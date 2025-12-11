import ehtim as eh
import numpy as np
import os
import subprocess
import pandas as pd
import sys

def test_nxcorr():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Use relative paths for portability
    uvfits_path = os.path.join(data_dir, 'crescent_LO_onsky.uvfits')
    truth_path = os.path.join(data_dir, 'crescent_LO_onsky_truth.hdf5')
    movie_path = os.path.join(data_dir, 'crescent_LO_onsky_truth.hdf5')
    out_prefix = os.path.join(data_dir, 'test_nxcorr_output')
    
    # Run nxcorr.py
    script_path = os.path.join(base_dir, '../src/nxcorr.py')
    
    cmd = [
        sys.executable, script_path,
        '-d', uvfits_path,
        '--truthmv', truth_path,
        '--input', movie_path, movie_path,  # Pass twice to trigger Bayesian mode
        '-o', out_prefix,
        '-n', '2'  # Use fewer cores for testing
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, "Script failed to run"
    
    # Check outputs for all modes (consolidated files)
    modes = ['total', 'static', 'dynamic']
    pols = ['I', 'P', 'X']
    
    for mode in modes:
        csv_file = f"{out_prefix}_{mode}.csv"
        png_file = f"{out_prefix}_{mode}.png"
        
        assert os.path.exists(csv_file), f"CSV output missing: {csv_file}"
        assert os.path.exists(png_file), f"PNG output missing: {png_file}"
        
        # Check CSV content
        df = pd.read_csv(csv_file)
        print(f"\n{mode} CSV Columns:", df.columns.tolist())
        
        # Check for all pols in the same file
        for pol in pols:
            # Bayesian columns
            expected_cols = [
                'time',
                f'nxcorr_{pol}_mean', f'nxcorr_{pol}_std',
                f'nxcorr_{pol}_thres_mean', f'nxcorr_{pol}_thres_std'
            ]
            
            # pass_rate exists for total and dynamic modes, but not static
            if mode != 'static':
                expected_cols.extend([
                    f'pass_rate_{pol}'
                ])
            
            for col in expected_cols:
                assert col in df.columns, f"Missing column in {mode} for {pol}: {col}"
        
        assert len(df) > 0, f"No data in {mode} CSV"
    
    print("\nAll tests passed!")
    
    # Cleanup: Delete test outputs
    print("\nCleaning up test outputs...")
    for mode in modes:
        csv_file = f"{out_prefix}_{mode}.csv"
        png_file = f"{out_prefix}_{mode}.png"
        
        if os.path.exists(csv_file):
            os.remove(csv_file)
        if os.path.exists(png_file):
            os.remove(png_file)

def test_nxcorr_single_movie():
    """Test non-Bayesian mode with a single movie"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    uvfits_path = os.path.join(data_dir, 'crescent_LO_onsky.uvfits')
    truth_path = os.path.join(data_dir, 'crescent_LO_onsky_truth.hdf5')
    movie_path = os.path.join(data_dir, 'crescent_LO_onsky_truth.hdf5')
    out_prefix = os.path.join(data_dir, 'test_nxcorr_single_output')
    
    script_path = os.path.join(base_dir, '../src/nxcorr.py')
    
    cmd = [
        sys.executable, script_path,
        '-d', uvfits_path,
        '--truthmv', truth_path,
        '--input', movie_path,  # Single movie
        '-o', out_prefix,
        '-n', '2'
    ]
    
    print(f"\nRunning single movie test: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, "Single movie test failed"
    
    # Check outputs
    modes = ['total', 'static', 'dynamic']
    pols = ['I', 'P', 'X']
    
    for mode in modes:
        csv_file = f"{out_prefix}_{mode}.csv"
        png_file = f"{out_prefix}_{mode}.png"
        
        assert os.path.exists(csv_file), f"CSV output missing: {csv_file}"
        assert os.path.exists(png_file), f"PNG output missing: {png_file}"
        
        df = pd.read_csv(csv_file)
        print(f"\n{mode} CSV Columns (single movie):", df.columns.tolist())
        
        for pol in pols:
            # Non-Bayesian columns
            expected_cols = [
                'time',
                f'nxcorr_{pol}',
                f'nxcorr_{pol}_thres'
            ]
            
            # pass_rate exists for total and dynamic modes, but not static
            if mode != 'static':
                expected_cols.append(f'pass_rate_{pol}')
            
            for col in expected_cols:
                assert col in df.columns, f"Missing column in {mode} for {pol}: {col}"
        
        assert len(df) > 0
    
    print("\nSingle movie test passed!")
    
    # Cleanup
    for mode in modes:
        csv_file = f"{out_prefix}_{mode}.csv"
        png_file = f"{out_prefix}_{mode}.png"
        
        if os.path.exists(csv_file):
            os.remove(csv_file)
        if os.path.exists(png_file):
            os.remove(png_file)

if __name__ == "__main__":
    test_nxcorr()
    test_nxcorr_single_movie()
