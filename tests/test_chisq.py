import ehtim as eh
import numpy as np
import os
import subprocess
import pandas as pd
import sys

def test_chisq():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Use relative paths for portability
    uvfits_path = os.path.join(data_dir, 'crescent_LO_onsky.uvfits')
    movie_path = os.path.join(data_dir, 'crescent_LO_onsky_truth.hdf5')
    out_prefix = os.path.join(data_dir, 'test_output')
    
    # Run chisq.py
    script_path = os.path.join(base_dir, '../src/chisq.py')
    
    cmd = [
        sys.executable, script_path,
        '-d', uvfits_path,
        '--input', movie_path, movie_path, # Pass twice to trigger Bayesian mode
        '-o', out_prefix
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, "Script failed to run"
    
    # Check outputs
    assert os.path.exists(out_prefix + ".csv"), "CSV output missing"
    assert os.path.exists(out_prefix + ".png"), "PNG output missing"
    
    # Check CSV content
    df = pd.read_csv(out_prefix + ".csv")
    print("CSV Columns:", df.columns)
    
    # Since we passed 2 movies, we expect Bayesian columns
    expected_cols = [
        'time', 
        'chisq_cp_mean', 'chisq_cp_std', 
        'chisq_lca_mean', 'chisq_lca_std', 
        'chisq_m_mean', 'chisq_m_std',
        'chisq_cp_avg_mean', 'chisq_cp_avg_std',
        'chisq_lca_avg_mean', 'chisq_lca_avg_std',
        'chisq_m_avg_mean', 'chisq_m_avg_std'
    ]
    
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
        
    assert len(df) > 0
    
    print("Test passed!")
    
    # Cleanup: Delete test outputs
    print("Cleaning up test outputs...")
    if os.path.exists(out_prefix + ".csv"):
        os.remove(out_prefix + ".csv")
        print(f"Deleted {out_prefix}.csv")
    if os.path.exists(out_prefix + ".png"):
        os.remove(out_prefix + ".png")
        print(f"Deleted {out_prefix}.png")

if __name__ == "__main__":
    test_chisq()
