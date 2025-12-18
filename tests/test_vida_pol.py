
import os
import subprocess
import pandas as pd
import glob
import sys
import shutil

# Test constants
DATA_DIR = 'tests/data'
OUT_PREFIX = 'tests/data/test_output'

def get_data_paths():
    # Helper to find data relative to current dir
    uvfits = os.path.join(DATA_DIR, 'crescent_LO_onsky.uvfits')
    hdf5 = os.path.join(DATA_DIR, 'crescent_LO_onsky_truth.hdf5')
    
    if not os.path.exists(uvfits):
        pass
    return uvfits, hdf5

def test_single_input():
    print("Testing Single Input (Non-Bayesian)...")
    uvfits, hdf5 = get_data_paths()
    out = f"{OUT_PREFIX}_single"
    
    cmd = [
        'python', 'src/vida_pol.py',
        '-d', uvfits,
        '--input', hdf5,
        '-o', out,
        '-n', '4',
        '--tstart', '11.0', 
        '--tstop', '11.2', # Short time range
        '--maxiters', '100', # Fast run
        '--stride', '50' # Batching
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    csv_file = out + "_vida.csv"
    assert os.path.exists(csv_file), "Single input CSV not found"
    
    # Check Plots
    assert os.path.exists(out + "_vida.png"), "Plot not found"
    
    df = pd.read_csv(csv_file)
    required = ['d', 'w', 'm_net', 'true_D', 'A', 'PA']
    for col in required:
        assert col in df.columns, f"Missing column {col}"
    print("Single Input Test Passed.")

def test_with_truth():
    print("Testing With Truth...")
    uvfits, hdf5 = get_data_paths()
    out = f"{OUT_PREFIX}_truth"
    
    cmd = [
        'python', 'src/vida_pol.py',
        '-d', uvfits,
        '--truthmv', hdf5,
        '--input', hdf5,
        '-o', out,
        '-n', '4',
        '--tstart', '11.0', 
        '--tstop', '11.2',
        '--maxiters', '100',
        '--stride', '50'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    csv_file = out + "_vida.csv"
    assert os.path.exists(csv_file), "Truth CSV output not found"
    
    # Check intermediate truth file deletion
    truth_csv = out + "_truth.csv"
    assert not os.path.exists(truth_csv), "Truth CSV should have been deleted"
    
    df = pd.read_csv(csv_file)
    assert 'd_truth' in df.columns, "Missing truth column"
    assert 'pass_percent_d' in df.columns, "Missing pass percent column"
    
    print("With Truth Test Passed.")

def cleanup():
    # Cleanup output files
    files = glob.glob(f"{OUT_PREFIX}*")
    for f in files:
        if os.path.exists(f): 
            if os.path.isdir(f): shutil.rmtree(f)
            else: os.remove(f)

if __name__ == "__main__":
    try:
        cleanup()
        if not os.path.exists('tests'):
             os.makedirs('tests')
             
        test_single_input()
        test_with_truth()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    finally:
        cleanup()
