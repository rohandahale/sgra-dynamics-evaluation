import os
import sys
import pandas as pd
import subprocess
import glob

def test_single_input():
    print("Testing Single Input (Non-Bayesian)...")
    
    # Paths relative to tests/ folder or repo root
    # Assuming running from repo root
    data_dir = 'tests/data'
    uvfits = os.path.join(data_dir, 'mring+hsCW_LO_onsky.uvfits')
    hdf5 = os.path.join(data_dir, 'mring+hsCW_LO_onsky_truth.hdf5')
    out_prefix = 'tests/test_output_single'
    
    if not os.path.exists(uvfits):
        # Try running from tests/ folder
        data_dir = 'data'
        uvfits = os.path.join(data_dir, 'mring+hsCW_LO_onsky.uvfits')
        hdf5 = os.path.join(data_dir, 'mring+hsCW_LO_onsky_truth.hdf5')
        out_prefix = 'test_output_single'
        
    if not os.path.exists(uvfits):
        raise FileNotFoundError(f"Data file not found: {uvfits}")

    cmd = [
        'python', 'src/rex.py',
        '-d', uvfits,
        '--input', hdf5,
        '-o', out_prefix,
        '-n', '1'
    ]
    
    # Adjust python path if running from tests/
    if not os.path.exists('src/rex.py'):
        cmd[1] = '../src/rex.py'
        
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    # Check outputs
    csv_path = out_prefix + '_rex.csv'
    png_path = out_prefix + '_rex.png'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Single input CSV not found")
    if not os.path.exists(png_path):
        raise FileNotFoundError("Single input PNG not found")
        
    df = pd.read_csv(csv_path)
    required_cols = ['D', 'W', 'mnet', 'evpa']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in single input CSV")
    print("Single Input Test Passed.")
    return uvfits, hdf5, out_prefix

def test_multi_input(uvfits, hdf5):
    print("Testing Multi Input (Bayesian)...")
    
    # For multi input, we can just use the same file twice to simulate multiple samples
    # This checks the logic of aggregation
    
    out_prefix = 'tests/test_output_multi'
    if not os.path.exists('tests'):
        out_prefix = 'test_output_multi'

    cmd = [
        'python', 'src/rex.py',
        '-d', uvfits,
        '--input', hdf5, hdf5, # Pass twice
        '-o', out_prefix,
        '-n', '2'
    ]
    
    if not os.path.exists('src/rex.py'):
        cmd[1] = '../src/rex.py'

    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    # Check outputs
    csv_path = out_prefix + '_rex.csv'
    png_path = out_prefix + '_rex.png'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Multi input CSV not found")
    if not os.path.exists(png_path):
        raise FileNotFoundError("Multi input PNG not found")
        
    df = pd.read_csv(csv_path)
    required_cols = ['D_mean', 'D_std', 'mnet_mean', 'mnet_std']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in multi input CSV")
    print("Multi Input Test Passed.")

def test_optional_truth(uvfits, hdf5):
    print("Testing Optional Truth (No --truthmv)...")
    
    out_prefix = 'tests/test_output_no_truth'
    if not os.path.exists('tests'):
        out_prefix = 'test_output_no_truth'

    cmd = [
        'python', 'src/rex.py',
        '-d', uvfits,
        '--input', hdf5,
        '-o', out_prefix,
        '-n', '1'
    ]
    
    if not os.path.exists('src/rex.py'):
        cmd[1] = '../src/rex.py'

    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    # Check outputs
    csv_path = out_prefix + '_rex.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Optional truth CSV not found")
        
    df = pd.read_csv(csv_path)
    # Check that truth columns are NOT present
    if 'D_truth' in df.columns:
        raise ValueError("Truth column present when --truthmv not provided")
        
    print("Optional Truth Test Passed.")

def test_with_truth(uvfits, hdf5):
    print("Testing With Truth (--truthmv)...")
    
    out_prefix = 'tests/test_output_with_truth'
    if not os.path.exists('tests'):
        out_prefix = 'test_output_with_truth'

    cmd = [
        'python', 'src/rex.py',
        '-d', uvfits,
        '--truthmv', hdf5, # Use same file as truth for testing
        '--input', hdf5,
        '-o', out_prefix,
        '-n', '1'
    ]
    
    if not os.path.exists('src/rex.py'):
        cmd[1] = '../src/rex.py'

    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    # Check outputs
    csv_path = out_prefix + '_rex.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError("With truth CSV not found")
        
    df = pd.read_csv(csv_path)
    # Check that truth columns ARE present
    if 'D_truth' not in df.columns:
        raise ValueError("Truth column MISSING when --truthmv provided")
        
    # Check for Pass Percentage columns
    expected_pass_cols = ['pass_percent_PAori', 'pass_percent_D', 'pass_percent_W', 'pass_percent_fc', 'pass_percent_A', 'pass_percent_papeak', 'pass_percent_true_D']
    
    # Add pol metrics if they exist in truth
    pol_mags = ['mnet', 'mavg', 'beta2_abs', 'vnet']
    for pm in pol_mags:
        if f'{pm}_truth' in df.columns:
            expected_pass_cols.append(f'pass_percent_{pm}')

    for col in expected_pass_cols:
        if col not in df.columns:
            raise ValueError(f"{col} column MISSING")
    
    # Note: pass_percent_beta2 might not be present if polarization data is missing or empty
    # But for our test data, we expect it if pol data exists.
    # Let's check if 'beta2_angle_truth' exists, then pass_percent_beta2_angle should exist.
    if 'beta2_angle_truth' in df.columns and 'pass_percent_beta2_angle' not in df.columns:
         raise ValueError("pass_percent_beta2_angle column MISSING but beta2_angle_truth is present")
        
    print("With Truth Test Passed.")

def cleanup():
    files = glob.glob('tests/test_output*') + glob.glob('test_output*')
    for f in files:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    try:
        cleanup()
        uvfits, hdf5, _ = test_single_input()
        test_multi_input(uvfits, hdf5)
        test_optional_truth(uvfits, hdf5)
        test_with_truth(uvfits, hdf5)
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    finally:
        cleanup()
