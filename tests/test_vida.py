######################################################################
# Test for vida.py
######################################################################

import os
import glob
import numpy as np
import ehtim as eh
import pandas as pd
import subprocess
import shutil

def test_vida_execution():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "../src")
    vida_script = os.path.join(src_dir, "vida.py")
    
    # Use provided test data using relative paths
    test_data_dir = os.path.join(base_dir, "data")
    input_h5 = os.path.join(test_data_dir, "mring+hsCW_LO_onsky_truth.hdf5")
    dummy_uvfits = os.path.join(test_data_dir, "mring+hsCW_LO_onsky.uvfits")
    
    if not os.path.exists(input_h5):
        print(f"Test data not found: {input_h5}")
        exit(1)
    
    # Truncate input file for speed
    print("Truncating input file...")
    mv = eh.movie.load_hdf5(input_h5)
    print(f"Loaded movie with {len(mv.im_list)} frames. Range: {mv.start_hr:.2f}-{mv.end_hr:.2f} hr")
    
    # Load UVFITS to get target times
    obs = eh.obsdata.load_uvfits(os.path.join(test_data_dir, "mring+hsCW_LO_onsky.uvfits"))
    obs_times = np.unique(obs.data['time'])
    print(f"UVFITS times: {len(obs_times)}. Range: {obs_times[0]:.2f}-{obs_times[-1]:.2f} hr")
    
    # Find overlapping times
    # We want to keep 3 frames that overlap with obs_times
    # Or just keep frames that COVER the range of some obs_times
    # Actually, mv.get_image(t) in vida.py needs t to be within [start, end] of truncated movie.
    # So we should pick a chunk of movie that COVERS some obs_times.
    
    start_t, end_t = obs_times[0], obs_times[-1]
    
    # Find frames in movie roughly within this range
    # Or just pick indices around the middle of obs_times
    
    # The truncated movie limits the VALID range for vida.py
    # if truncated movie is [10.85, 10.88] and obs times are [10.9, 11.0], no overlap.
    # So we must pick frames such that [min(frames), max(frames)] intersects obs_times.
    
    # Let's pick frames closest to first few obs_times
    target_ts = obs_times[:3]
    indices = []
    import bisect
    sorted_mv_times = sorted(mv.times)
    
    # Just grab frames that bracket the first few obs times
    # Or better: create a movie that spans the ENTIRE range but only has a few frames?
    # No, that's dangerous for interpolation.
    # Let's just find frames nearest to the first 3 obs times.
    
    selected_frames = []
    
    # We need a range that COVERS at least one obs time.
    # Check if ANY obs time is in the original movie range
    valid_obs = [t for t in obs_times if mv.start_hr <= t <= mv.end_hr]
    if not valid_obs:
        print("Warning: No overlap between original movie and UVFITS. Creating synthetic frames.")
        # If no overlap, synthesize frames at obs_times[0]..[2]
        # Just copy frame 0 and set its time
        base_im = mv.im_list[0]
        for i in range(3):
            im = base_im.copy()
            im.time = obs_times[i]
            selected_frames.append(im)
    else:
        # Pick 3 frames around the first valid obs time
        t_center = valid_obs[0]
        # find closest frame
        idx = np.argmin(np.abs(np.array(mv.times) - t_center))
        # slice around it
        start = max(0, idx - 1)
        end = min(len(mv.im_list), start + 3)
        selected_frames = mv.im_list[start:end]
        
    print(f"Merging {len(selected_frames)} frames from {selected_frames[0].time:.2f} to {selected_frames[-1].time:.2f} hr")
    mv_trunc = eh.movie.merge_im_list(selected_frames)
    truncated_h5 = "test_vida_input_trunc.hdf5"
    truncated_h5_2 = "test_vida_input_trunc_2.hdf5"
    mv_trunc.save_hdf5(truncated_h5)
    shutil.copy(truncated_h5, truncated_h5_2)
    
    output_prefix = "test_vida_out"
    
    # Use 2 files to trigger aggregation
    cmd = [
        "python3", vida_script,
        "-d", dummy_uvfits,
        "--input", truncated_h5, truncated_h5_2,
        "--truthmv", truncated_h5, # Use same file as truth
        "-o", output_prefix,
        "--ncores", "1",
        "--julia_threads", "4"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed: {e}")
        exit(1)
        
    # Check output CSV
    expected_csv = output_prefix + ".csv"
    if not os.path.exists(expected_csv):
        print("Output CSV not created")
        exit(1)
    
    df = pd.read_csv(expected_csv)
    print("Columns:", df.columns)
    
    # Verify content
    if 'time' not in df.columns:
        print("Missing time column")
        exit(1)
    if 'mnet_mean' not in df.columns:
        print("Missing mnet_mean column")
        exit(1)
    # Check for truth columns
    if 'mnet_truth' not in df.columns:
        print("Missing mnet_truth column")
        exit(1)
        
    # Check for pass percentages
    if 'pass_percent_mnet' not in df.columns:
         print("Missing pass_percent_mnet column")
         exit(1)
         
    # Check values (should be 100% since truth=input)
    # Actually input is mean of 2 identical files, so mean=truth.
    pass_val = df['pass_percent_mnet'].iloc[0]
    if pass_val < 99.0:
        print(f"Pass percentage too low: {pass_val}")
        # Note: uncertainty is 0, thresholds are >0. diff is 0. Condition: 0 <= thresh. Pass.
        exit(1)
    
    # Check Plots
    expected_metrics_plot = output_prefix + "_metrics.png"
    
    if not os.path.exists(expected_metrics_plot):
        print(f"Metrics plot not created: {expected_metrics_plot}")
        exit(1)

    print(f"Generated plot {expected_metrics_plot}")
    print(f"Generated {len(df)} rows")
    
    print("Test PASSED")
    
    # Cleanup
    if os.path.exists(truncated_h5): os.remove(truncated_h5)
    if os.path.exists(truncated_h5_2): os.remove(truncated_h5_2)
    if os.path.exists(expected_csv): os.remove(expected_csv)
    if os.path.exists(expected_metrics_plot): os.remove(expected_metrics_plot)
    if os.path.exists("./vida_temp"): shutil.rmtree("./vida_temp")

if __name__ == "__main__":
    test_vida_execution()
