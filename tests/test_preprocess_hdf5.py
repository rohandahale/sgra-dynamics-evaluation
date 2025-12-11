import os
import sys
import subprocess
import glob
import shutil
import ehtim as eh

def test_preprocess_hdf5():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # Ensure data exists (it should based on previous checks)
    uvfits_path = os.path.join(data_dir, 'crescent_LO_onsky.uvfits')
    movie_path = os.path.join(data_dir, 'crescent_LO_onsky_truth.hdf5')
    
    if not os.path.exists(uvfits_path) or not os.path.exists(movie_path):
        print("Skipping test: Data files not found in tests/data")
        return

    # Define output directory
    out_dir = os.path.join(data_dir, 'test_preprocess_output')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Output prefix (full path as requested by user logic)
    out_prefix = os.path.join(out_dir, 'result')
    
    # Script path
    script_path = os.path.join(base_dir, '../src/preprocess_hdf5.py')
    
    # Construct command
    cmd = [
        sys.executable, script_path,
        '-d', uvfits_path,
        '--input', movie_path,
        '-o', out_prefix,
        '-n', '2'  # Use 2 cores for testing
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"
    
    # Verify outputs
    basename = os.path.basename(movie_path)
    basename_noext = os.path.splitext(basename)[0]
    
    # Expected output files
    regrid_file = os.path.join(out_dir, 'regrid_hdf5', basename)
    static_file = os.path.join(out_dir, 'static_part', f'{basename_noext}.fits')
    dynamic_file = os.path.join(out_dir, 'dynamic_part', basename)
    uniform_file = os.path.join(out_dir, 'uniform_hdf5', basename)
    
    assert os.path.exists(regrid_file), f"Regridded output missing: {regrid_file}"
    assert os.path.exists(static_file), f"Static output missing: {static_file}"
    assert os.path.exists(dynamic_file), f"Dynamic output missing: {dynamic_file}"
    assert os.path.exists(uniform_file), f"Uniform output missing: {uniform_file}"
    
    # Verify content (basic checks)
    print("Verifying output content...")
    
    # Check regridded movie
    mov_regrid = eh.movie.load_hdf5(regrid_file)
    assert mov_regrid.xdim == 200
    assert mov_regrid.ydim == 200
    print("Regridded movie dimensions correct.")
    
    # Check static FITS
    im_static = eh.image.load_fits(static_file)
    assert im_static.xdim == 200
    assert im_static.ydim == 200
    print("Static image dimensions correct.")
    
    # Check dynamic movie
    mov_dynamic = eh.movie.load_hdf5(dynamic_file)
    assert mov_dynamic.xdim == 200
    assert mov_dynamic.ydim == 200
    print("Dynamic movie dimensions correct.")
    
    # Check uniform movie
    mov_uniform = eh.movie.load_hdf5(uniform_file)
    assert len(mov_uniform.times) == 100
    print("Uniform movie frame count correct.")
    
    print("\nAll tests passed!")
    
    # Cleanup
    print("Cleaning up...")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

if __name__ == "__main__":
    test_preprocess_hdf5()
