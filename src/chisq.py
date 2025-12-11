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
    p.add_argument('-n', '--ncores', type=int, default=32, help='Number of cores to use for parallel processing')
    return p

def process_movie(m_path, obslist_t, times):
    """
    Process a single movie file to calculate chi-squared metrics.
    """
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            mv = eh.movie.load_hdf5(m_path)

            metrics = {'chicp': [], 'chilca': [], 'chim': []}
            
            for i, t in enumerate(times):
                # Get image at time t
                # Note: mv.get_image(t) interpolates if t is not exactly in mv.times
                # We assume the times are covered by the movie.
                im = mv.get_image(t)

                current_obs = obslist_t[i]

                #flag short baselines before forming closure quantities
                current_obs = current_obs.flag_bl(["AA", "AP"])
                
                # Flag sites for specific metrics (JC for polchisq)
                current_obs_pol = current_obs.flag_sites(['JC'])
                
                # Compute Chi-squared
                # cphase
                chicp = current_obs.chisq(im, dtype='cphase', pol='I', ttype='direct', cp_uv_min=1e8)
                metrics['chicp'].append(chicp)
                
                # logcamp
                chilca = current_obs.chisq(im, dtype='logcamp', pol='I', ttype='direct', cp_uv_min=1e8)
                metrics['chilca'].append(chilca)
                
                # mbreve (polarization)
                
                #Remove nan values
                mask_nan = np.isnan(current_obs_pol.data['vis']) \
                             + np.isnan(current_obs_pol.data['qvis']) \
                             + np.isnan(current_obs_pol.data['uvis']) \
                             + np.isnan(current_obs_pol.data['vvis'])
                current_obs_pol.data = current_obs_pol.data[~mask_nan]

                mask = im.ivec != 0
                chim = current_obs_pol.polchisq(im, dtype='m', ttype='direct', mask=mask, cp_uv_min=1e8)
                metrics['chim'].append(chim)
            
    return metrics


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
            #obs = obs.add_fractional_noise(0.01)
            obs.add_scans()
            obslist = obs.split_obs()
    
    obs_times = np.array([o.data['time'][0] for o in obslist])
    
    min_t_list = []
    max_t_list = []
    
    print("Scanning movies for time range...")
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for m_path in input_files:
                mv = eh.movie.load_hdf5(m_path)
                min_t_list.append(min(mv.times))
                max_t_list.append(max(mv.times))
            
    if not min_t_list:
        print("No valid movies found.")
        return

    min_t = max(min_t_list)
    max_t = min(max_t_list)
    
    valid_indices = np.where((obs_times >= min_t) & (obs_times <= max_t))[0]
    obslist_t = [obslist[i] for i in valid_indices]
    times = obs_times[valid_indices]
    
    print(f"Processing {len(times)} time steps from {min_t} to {max_t}.")

    # Number of data points per time step (needed for weighted avg)
    num_list = [len(o.data) for o in obslist_t]
    num_arr = np.array(num_list)
    total_num = np.sum(num_arr)

    # Parallel Processing
    max_workers = args.ncores
    print(f"Starting parallel processing with {max_workers} workers...")
    
    movie_metrics = []
    
    # We pass obslist_t and times to all workers.
    # functools.partial can fix these arguments.
    process_func = functools.partial(process_movie, obslist_t=obslist_t, times=times)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_gen = executor.map(process_func, input_files)
        
        for res in tqdm(results_gen, total=len(input_files), desc="Processing hdf5 files"):
            if res is not None:
                movie_metrics.append(res)
    
    if not movie_metrics:
        print("No metrics calculated.")
        return
    
    # We need to collect all time-series data to calculate mean/std across movies if Bayesian
    # Shape: (n_movies, n_times)
    all_chicp = np.array([m['chicp'] for m in movie_metrics])
    all_chilca = np.array([m['chilca'] for m in movie_metrics])
    all_chim = np.array([m['chim'] for m in movie_metrics])
    
    # Calculate weighted averages for each movie
    # Shape: (n_movies,)
    avg_chicp = np.sum(all_chicp * num_arr, axis=1) / total_num
    avg_chilca = np.sum(all_chilca * num_arr, axis=1) / total_num
    avg_chim = np.sum(all_chim * num_arr, axis=1) / total_num
    
    results = {'time': times}
    
    if len(movie_metrics) > 1:
        # Bayesian Mode
        print("Calculating Bayesian statistics...")
        
        # Per-time step statistics
        results['chisq_cp_mean'] = np.mean(all_chicp, axis=0)
        results['chisq_cp_std'] = np.std(all_chicp, axis=0)
        
        results['chisq_lca_mean'] = np.mean(all_chilca, axis=0)
        results['chisq_lca_std'] = np.std(all_chilca, axis=0)
        
        results['chisq_m_mean'] = np.mean(all_chim, axis=0)
        results['chisq_m_std'] = np.std(all_chim, axis=0)
        
        # Time-averaged statistics (scalar values, repeated for dataframe)
        # Mean and Std of the weighted averages across movies
        results['chisq_cp_avg_mean'] = np.full(len(times), np.mean(avg_chicp))
        results['chisq_cp_avg_std'] = np.full(len(times), np.std(avg_chicp))
        
        results['chisq_lca_avg_mean'] = np.full(len(times), np.mean(avg_chilca))
        results['chisq_lca_avg_std'] = np.full(len(times), np.std(avg_chilca))
        
        results['chisq_m_avg_mean'] = np.full(len(times), np.mean(avg_chim))
        results['chisq_m_avg_std'] = np.full(len(times), np.std(avg_chim))
        
    else:
        # Non-Bayesian Mode (Single Movie)
        print("Processing single movie results...")
        
        results['chisq_cp'] = all_chicp[0]
        results['chisq_lca'] = all_chilca[0]
        results['chisq_m'] = all_chim[0]
        
        results['chisq_cp_avg'] = np.full(len(times), avg_chicp[0])
        results['chisq_lca_avg'] = np.full(len(times), avg_chilca[0])
        results['chisq_m_avg'] = np.full(len(times), avg_chim[0])

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = args.outpath + ".csv"
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
    
    if len(movie_metrics) > 1:
        # Plot Mean with Error Bars
        ax[0].errorbar(times, results['chisq_cp_mean'], yerr=results['chisq_cp_std'], fmt='-o', label='Mean', color='tab:orange', capsize=3, alpha=0.8)
        ax[1].errorbar(times, results['chisq_lca_mean'], yerr=results['chisq_lca_std'], fmt='-o', color='tab:orange', capsize=3, alpha=0.8)
        ax[2].errorbar(times, results['chisq_m_mean'], yerr=results['chisq_m_std'], fmt='-o', color='tab:orange', capsize=3, alpha=0.8)
        
        # Add horizontal lines for the chisq=1
        ax[0].axhline(1, color='black', linestyle='-')
        ax[1].axhline(1, color='black', linestyle='-')
        ax[2].axhline(1, color='black', linestyle='-')

        # Add horizontal lines for the global average (mean of avg)
        ax[0].axhline(results['chisq_cp_avg_mean'][0], color='tab:orange', linestyle='--', label='$\chi^{2}_{cp}$ avg:'+f'{results["chisq_cp_avg_mean"][0]:.2f} $\pm$ {results["chisq_cp_avg_std"][0]:.2f}')
        ax[1].axhline(results['chisq_lca_avg_mean'][0], color='tab:orange', linestyle='--', label='$\chi^{2}_{lca}$ avg:'+f'{results["chisq_lca_avg_mean"][0]:.2f} $\pm$ {results["chisq_lca_avg_std"][0]:.2f}')
        ax[2].axhline(results['chisq_m_avg_mean'][0], color='tab:orange', linestyle='--', label='$\chi^{2}_{m}$ avg:'+f'{results["chisq_m_avg_mean"][0]:.2f} $\pm$ {results["chisq_m_avg_std"][0]:.2f}')
        
    else:
        # Plot Single Line
        ax[0].plot(times, results['chisq_cp'], '-o', color='tab:blue')
        ax[1].plot(times, results['chisq_lca'], '-o', color='tab:blue')
        ax[2].plot(times, results['chisq_m'], '-o', color='tab:blue')
        
        # Add horizontal lines for the chisq=1
        ax[0].axhline(1, color='black', linestyle='-')
        ax[1].axhline(1, color='black', linestyle='-')
        ax[2].axhline(1, color='black', linestyle='-')

        # Add horizontal lines for the average
        ax[0].axhline(results['chisq_cp_avg'][0], color='tab:blue', linestyle='--', label='$\chi^{2}_{cp}$ avg:'+f'{results["chisq_cp_avg"][0]:.2f}')
        ax[1].axhline(results['chisq_lca_avg'][0], color='tab:blue', linestyle='--', label='$\chi^{2}_{lca}$ avg:'+f'{results["chisq_lca_avg"][0]:.2f}')
        ax[2].axhline(results['chisq_m_avg'][0], color='tab:blue', linestyle='--', label='$\chi^{2}_{m}$ avg:'+f'{results["chisq_m_avg"][0]:.2f}')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()
    png_path = args.outpath + ".png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {png_path}")

if __name__ == "__main__":
    main()