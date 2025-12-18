import os
import subprocess
import pandas as pd
import sys

def test_hotspot():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Use relative paths for portability
    uvfits_path = os.path.join(data_dir, 'mring+hsCW_LO_onsky.uvfits')
    truth_path = os.path.join(data_dir, 'mring+hsCW_LO_onsky_truth.hdf5')
    movie_path = os.path.join(data_dir, 'mring+hsCW_LO_onsky_truth.hdf5')
    out_prefix = os.path.join(data_dir, 'test_hotspot_output')
    
    # Run hotspot.py
    script_path = os.path.join(base_dir, '../src/hotspot.py')
    
    # Test Bayesian mode (2 inputs)
    cmd = [
        sys.executable, script_path,
        '-d', uvfits_path,
        '--truthmv', truth_path,
        '--input', movie_path, movie_path,
        '-o', out_prefix,
        '-n', '2'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, "Script failed to run"
    
    # Check outputs
    csv_file = f"{out_prefix}_hotspot.csv"
    png_file = f"{out_prefix}_hotspot.png"
    
    assert os.path.exists(csv_file), f"CSV output missing: {csv_file}"
    assert os.path.exists(png_file), f"PNG output missing: {png_file}"
    
    # Check CSV content
    df = pd.read_csv(csv_file)
    print(f"\nCSV Columns:", df.columns.tolist())
    
    quantities = ["x", "y", "distance", "angle", "fwhm", "flux"]
    for q in quantities:
        # Check for mean, std, truth, threshold, pass_percent
        expected_cols = [
            f'{q}_mean', f'{q}_std',
            f'{q}_truth',
            f'{q}_threshold',
            f'{q}_pass_percent'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column for {q}: {col}"
    
    assert len(df) > 0, "No data in CSV"
    print("\nBayesian test passed!")
    
    # Cleanup
    if os.path.exists(csv_file): os.remove(csv_file)
    if os.path.exists(png_file): os.remove(png_file)

def test_hotspot_single():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    uvfits_path = os.path.join(data_dir, 'mring+hsCW_LO_onsky.uvfits')
    truth_path = os.path.join(data_dir, 'mring+hsCW_LO_onsky_truth.hdf5')
    movie_path = os.path.join(data_dir, 'mring+hsCW_LO_onsky_truth.hdf5')
    out_prefix = os.path.join(data_dir, 'test_hotspot_single_output')
    
    script_path = os.path.join(base_dir, '../src/hotspot.py')
    
    # Test Non-Bayesian mode (1 input)
    cmd = [
        sys.executable, script_path,
        '-d', uvfits_path,
        '--truthmv', truth_path,
        '--input', movie_path,
        '-o', out_prefix,
        '-n', '2'
    ]
    
    print(f"Running single input command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, "Script failed to run"
    
    # Check outputs
    csv_file = f"{out_prefix}_hotspot.csv"
    png_file = f"{out_prefix}_hotspot.png"
    
    assert os.path.exists(csv_file), f"CSV output missing: {csv_file}"
    assert os.path.exists(png_file), f"PNG output missing: {png_file}"
    
    # Check CSV content
    df = pd.read_csv(csv_file)
    print(f"\nCSV Columns:", df.columns.tolist())
    
    quantities = ["x", "y", "distance", "angle", "fwhm", "flux"]
    for q in quantities:
        # Check for value (no suffix), truth, threshold, pass_percent
        expected_cols = [
            f'{q}',
            f'{q}_truth',
            f'{q}_threshold',
            f'{q}_pass_percent'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column for {q}: {col}"
            
        # Verify NO mean/std columns exist
        assert f'{q}_mean' not in df.columns, f"Column {q}_mean should not exist"
        assert f'{q}_std' not in df.columns, f"Column {q}_std should not exist"

    assert len(df) > 0, "No data in CSV"
    print("\nNon-Bayesian test passed!")
    
    # Cleanup
    if os.path.exists(csv_file): os.remove(csv_file)
    if os.path.exists(png_file): os.remove(png_file)

if __name__ == "__main__":
    test_hotspot()
    test_hotspot_single()
