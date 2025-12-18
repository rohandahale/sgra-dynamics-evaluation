######################################################################
# Author: Rohan Dahale, Date: 10 December 2025
# Discussions with: Kotaro Moriyama
######################################################################

import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set environment variables to single-threaded before importing numpy/scipy/ehtim
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import ehtim as eh
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor
import functools
import glob
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr

# Set plot style
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.family"] = "sans-serif"

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, required=True, help='UVFITS data file')
    p.add_argument('--input', type=str, nargs='+', required=True, help='Glob pattern(s) or list of HDF5 model files (e.g., "path/*.h5")')
    p.add_argument('-o', '--outpath', type=str, default='./chi2', help='Output prefix (without extension)')
    p.add_argument('--tstart', type=float, default=None, help='Start time (in UT hours) for data')
    p.add_argument('--tstop', type=float, default=None, help='Stop time (in UT hours) for data')
    p.add_argument('-n', '--ncores', type=int, default=32, help='Number of cores to use for parallel processing')
    return p

def compute_metrics_for_obs(obs, im):
    """
    Compute chi-squared metrics for a given observation and image.
    Returns a dictionary of metrics.
    """
    res = {}
    
    # cphase
    res['chicp'] = obs.chisq(im, dtype='cphase', pol='I', ttype='direct', cp_uv_min=1e8)
    
    # logcamp
    res['chilca'] = obs.chisq(im, dtype='logcamp', pol='I', ttype='direct', cp_uv_min=1e8)
    
    # mbreve (polarization)
    # Flag sites for specific metrics (JC for polchisq)
    obs_pol = obs.flag_sites(['JC'])
    
    # Remove nan values
    mask_nan = np.isnan(obs_pol.data['vis']) \
                 + np.isnan(obs_pol.data['qvis']) \
                 + np.isnan(obs_pol.data['uvis']) \
                 + np.isnan(obs_pol.data['vvis'])
    obs_pol.data = obs_pol.data[~mask_nan]

    mask = im.ivec != 0
    res['chim'] = obs_pol.polchisq(im, dtype='m', ttype='direct', mask=mask, cp_uv_min=1e8)
    
    return res

def process_movie(m_path, obslist_t, times):
    """
    Process a single movie file to calculate chi-squared metrics for both:
    1. Full baselines (Standard)
    2. Flagged AA/AP baselines (Flagged)
    """
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            mv = eh.movie.load_hdf5(m_path)
            mv.reset_interp(bounds_error=False)

            metrics_full = {'chicp': [], 'chilca': [], 'chim': []}
            metrics_flag = {'chicp': [], 'chilca': [], 'chim': []}
            
            for i, t in enumerate(times):
                # Get image at time t
                im = mv.get_image(t)
                current_obs = obslist_t[i]

                # --- 1. FULL ---
                res_full = compute_metrics_for_obs(current_obs, im)
                metrics_full['chicp'].append(res_full['chicp'])
                metrics_full['chilca'].append(res_full['chilca'])
                metrics_full['chim'].append(res_full['chim'])

                # --- 2. FLAGGED BASELINES (AA, AP) ---
                current_obs_flag = current_obs.flag_bl(["AA", "AP"])
                res_flag = compute_metrics_for_obs(current_obs_flag, im)
                metrics_flag['chicp'].append(res_flag['chicp'])
                metrics_flag['chilca'].append(res_flag['chilca'])
                metrics_flag['chim'].append(res_flag['chim'])
                    
            
    return metrics_full, metrics_flag

def save_and_plot(movie_metrics_list, times, num_arr, total_num, outpath_prefix, suffix):
    """
    Aggregates metrics from multiple movies, saves to CSV, and plots results.
    """
    if not movie_metrics_list:
        print(f"No metrics to process for {suffix}")
        return

    # We need to collect all time-series data to calculate mean/std across movies if Bayesian
    # Shape: (n_movies, n_times)
    all_chicp = np.array([m['chicp'] for m in movie_metrics_list])
    all_chilca = np.array([m['chilca'] for m in movie_metrics_list])
    all_chim = np.array([m['chim'] for m in movie_metrics_list])
    
    # Calculate weighted averages for each movie
    # Shape: (n_movies,)
    avg_chicp = np.sum(all_chicp * num_arr, axis=1) / total_num
    avg_chilca = np.sum(all_chilca * num_arr, axis=1) / total_num
    avg_chim = np.sum(all_chim * num_arr, axis=1) / total_num
    
    results = {'time': times}
    
    is_bayesian = len(movie_metrics_list) > 1

    if is_bayesian:
        # Per-time step statistics
        results['chisq_cp_mean'] = np.mean(all_chicp, axis=0)
        results['chisq_cp_std'] = np.std(all_chicp, axis=0)
        
        results['chisq_lca_mean'] = np.mean(all_chilca, axis=0)
        results['chisq_lca_std'] = np.std(all_chilca, axis=0)
        
        results['chisq_m_mean'] = np.mean(all_chim, axis=0)
        results['chisq_m_std'] = np.std(all_chim, axis=0)
        
        # Time-averaged statistics
        results['chisq_cp_avg_mean'] = np.full(len(times), np.mean(avg_chicp))
        results['chisq_cp_avg_std'] = np.full(len(times), np.std(avg_chicp))
        
        results['chisq_lca_avg_mean'] = np.full(len(times), np.mean(avg_chilca))
        results['chisq_lca_avg_std'] = np.full(len(times), np.std(avg_chilca))
        
        results['chisq_m_avg_mean'] = np.full(len(times), np.mean(avg_chim))
        results['chisq_m_avg_std'] = np.full(len(times), np.std(avg_chim))
        
    else:
        # Non-Bayesian Mode (Single Movie)
        results['chisq_cp'] = all_chicp[0]
        results['chisq_lca'] = all_chilca[0]
        results['chisq_m'] = all_chim[0]
        
        results['chisq_cp_avg'] = np.full(len(times), avg_chicp[0])
        results['chisq_lca_avg'] = np.full(len(times), avg_chilca[0])
        results['chisq_m_avg'] = np.full(len(times), avg_chim[0])

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = outpath_prefix + suffix + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 6), sharex=True)
    
    ax[0].set_ylabel(r'$\chi^{2}$ cphase')
    ax[1].set_ylabel(r'$\chi^{2}$ logcamp')
    ax[2].set_ylabel(r'$\chi^{2}$ mbreve')
    
    for a in ax:
        a.set_xlabel('Time (UTC)')
        a.set_yscale('log')
    
    if is_bayesian:
        # Plot Mean with Error Bars
        ax[0].errorbar(times, results['chisq_cp_mean'], yerr=results['chisq_cp_std'], fmt='-o', label='Mean', color='tab:orange', capsize=3, alpha=0.8)
        ax[1].errorbar(times, results['chisq_lca_mean'], yerr=results['chisq_lca_std'], fmt='-o', color='tab:orange', capsize=3, alpha=0.8)
        ax[2].errorbar(times, results['chisq_m_mean'], yerr=results['chisq_m_std'], fmt='-o', color='tab:orange', capsize=3, alpha=0.8)
        
        # Add horizontal lines
        ax[0].axhline(1, color='black', linestyle='-')
        ax[1].axhline(1, color='black', linestyle='-')
        ax[2].axhline(1, color='black', linestyle='-')

        # Add horizontal lines for global average
        ax[0].axhline(results['chisq_cp_avg_mean'][0], color='tab:orange', linestyle='--', label='$\chi^{2}_{cp}$ avg:'+f'{results["chisq_cp_avg_mean"][0]:.2f} $\pm$ {results["chisq_cp_avg_std"][0]:.2f}')
        ax[1].axhline(results['chisq_lca_avg_mean'][0], color='tab:orange', linestyle='--', label='$\chi^{2}_{lca}$ avg:'+f'{results["chisq_lca_avg_mean"][0]:.2f} $\pm$ {results["chisq_lca_avg_std"][0]:.2f}')
        ax[2].axhline(results['chisq_m_avg_mean'][0], color='tab:orange', linestyle='--', label='$\chi^{2}_{m}$ avg:'+f'{results["chisq_m_avg_mean"][0]:.2f} $\pm$ {results["chisq_m_avg_std"][0]:.2f}')
        
    else:
        # Plot Single Line
        ax[0].plot(times, results['chisq_cp'], '-o', color='tab:blue')
        ax[1].plot(times, results['chisq_lca'], '-o', color='tab:blue')
        ax[2].plot(times, results['chisq_m'], '-o', color='tab:blue')
        
        # Add horizontal lines
        ax[0].axhline(1, color='black', linestyle='-')
        ax[1].axhline(1, color='black', linestyle='-')
        ax[2].axhline(1, color='black', linestyle='-')

        # Add horizontal lines for average
        ax[0].axhline(results['chisq_cp_avg'][0], color='tab:blue', linestyle='--', label='$\chi^{2}_{cp}$ avg:'+f'{results["chisq_cp_avg"][0]:.2f}')
        ax[1].axhline(results['chisq_lca_avg'][0], color='tab:blue', linestyle='--', label='$\chi^{2}_{lca}$ avg:'+f'{results["chisq_lca_avg"][0]:.2f}')
        ax[2].axhline(results['chisq_m_avg'][0], color='tab:blue', linestyle='--', label='$\chi^{2}_{m}$ avg:'+f'{results["chisq_m_avg"][0]:.2f}')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()
    png_path = outpath_prefix + suffix + ".png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {png_path}")
    plt.close(fig) # Close figure to free memory

def main():
    args = create_parser().parse_args()
    
    # Expand glob patterns for input files
    input_files = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        if matched:
            input_files.extend(matched)
        else:
            # If no match, treat as literal filename
            input_files.append(pattern)
    
    if not input_files:
        print("No input files found matching the pattern(s).")
        return
    
    print(f"Found {len(input_files)} input file(s): {input_files}")
    
    # Load observation
    print(f"Loading observation: {args.data}")
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            obs = eh.obsdata.load_uvfits(args.data)
            obs.add_scans()
            obslist = obs.split_obs()
    
    # Calculate data time range
    obs_times = np.array([o.data['time'][0] for o in obslist])
    data_min_t = obs_times.min()
    data_max_t = obs_times.max()
    print(f"Data time range: {data_min_t:.3f} - {data_max_t:.3f} h")

    # Flag data if tstart/tstop provided
    if args.tstart is not None or args.tstop is not None:
        tstart = args.tstart if args.tstart is not None else data_min_t
        tstop = args.tstop if args.tstop is not None else data_max_t
        print(f"Time flagging data to use in range: {tstart:.3f} - {tstop:.3f} h")
        
        with open(os.devnull, 'w') as devnull:
             with redirect_stdout(devnull), redirect_stderr(devnull):
                 obs = obs.flag_UT_range(UT_start_hour=tstart, UT_stop_hour=tstop, output='flagged')
                 obs.add_scans()
                 obslist = obs.split_obs()

        if not obslist:
            print("No data remaining after time flagging.")
            return
        
        obs_times = np.array([o.data['time'][0] for o in obslist])
        print(f"New data time range: {obs_times.min():.3f} - {obs_times.max():.3f} h")

    times = obs_times
    obslist_t = obslist

    min_t_list = []
    max_t_list = []
    
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for m_path in input_files:
                mv = eh.movie.load_hdf5(m_path)
                min_t_list.append(min(mv.times))
                max_t_list.append(max(mv.times))
            
    if not min_t_list:
        print("No valid movies found.")
        return

    movie_min_t = max(min_t_list)
    movie_max_t = min(max_t_list)
    print(f"Movie time range: {movie_min_t:.3f} - {movie_max_t:.3f} h")

    if movie_min_t > times.min() or movie_max_t < times.max():
         print("Warning: Movie times do not span the whole duration of data. Extrapolation will be used.")
    
    print(f"Processing {len(times)} time steps.")

    # Number of data points per time step (needed for weighted avg)
    num_list = [len(o.data) for o in obslist_t]
    num_arr = np.array(num_list)
    total_num = np.sum(num_arr)

    # Parallel Processing
    max_workers = args.ncores
    print(f"Starting parallel processing with {max_workers} workers...")
    
    movie_metrics_full = []
    movie_metrics_flag = []
    
    # We pass obslist_t and times to all workers.
    # functools.partial can fix these arguments.
    process_func = functools.partial(process_movie, obslist_t=obslist_t, times=times)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_gen = executor.map(process_func, input_files)
        
        for res_full, res_flag in tqdm(results_gen, total=len(input_files), desc="Processing hdf5 files"):
            if res_full is not None:
                movie_metrics_full.append(res_full)
            if res_flag is not None:
                movie_metrics_flag.append(res_flag)
    
    if not movie_metrics_full:
        print("No metrics calculated.")
        return
    
    # --- PROCESS AND SAVE RESULTS ---
    
    # 1. Standard (All baselines)
    # Output: _chisq.csv / .png
    print("\n Saving results for STANDARD baselines (chisq.csv)...")
    save_and_plot(movie_metrics_full, times, num_arr, total_num, args.outpath, "_chisq")
    
    # 2. Flagged (AA/AP)
    # Output: _chisq_flagAAAP.csv / .png
    print("\n Saving results for FLAGGED baselines (chisq_flagAAAP.csv)...")
    save_and_plot(movie_metrics_flag, times, num_arr, total_num, args.outpath, "_chisq_flagAAAP")

    print("\nDone.")

if __name__ == "__main__":
    main()