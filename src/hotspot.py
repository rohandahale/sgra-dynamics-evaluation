######################################################################
# Author: Rohan Dahale, Date: 2 December 2025
# Based on: Antonio Fuentes' original script
######################################################################

import os

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import sys
import warnings
import argparse
import glob
import numpy as np
import pandas as pd
import ehtim as eh
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
import matplotlib as mpl

# Suppress warnings
warnings.filterwarnings('ignore')


# Plotting style
mpl.rcParams['figure.dpi'] = 300
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 18

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, required=True, help='UVFITS data file')
    p.add_argument('--truthmv', type=str, required=True, help='Truth HDF5 movie file')
    p.add_argument('--input', type=str, nargs='+', required=True, help='Glob pattern(s) or list of HDF5 model files (e.g., "path/*.h5")')
    p.add_argument('-o', '--outpath', type=str, default='./hotspot', help='Output prefix (without extension)')
    p.add_argument('--tstart', type=float, default=None, help='Start time (in UT hours) for data')
    p.add_argument('--tstop', type=float, default=None, help='Stop time (in UT hours) for data')
    p.add_argument('-n', '--ncores', type=int, default=32, help='Number of cores to use for parallel processing')
    return p

# --- Gaussian fitting ---
def gaussian_fit(img):
    xx, yy = np.unravel_index(np.argmax(img), img.shape)
    g_init = models.Gaussian2D(amplitude=img.max(), x_mean=yy, y_mean=xx)
    fit_g = fitting.LMLSQFitter(calc_uncertainties=True)
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = fit_g(g_init, x, y, img)
    return g, g(x, y)

def process_movie(filepath, times, fov, pix):
    try:
        # Suppress ehtim output
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                mov = eh.movie.load_hdf5(filepath)
                mov.reset_interp(bounds_error=False)
                frames = [mov.get_image(t).regrid_image(fov*eh.RADPERUAS, pix).imarr() for t in times]
        
        median_frame = np.median(frames, axis=0)
        dyn_frames = np.clip(frames - median_frame, 0, None)
        
        fits = [gaussian_fit(img)[0] for img in dyn_frames]
        
        xs = np.array([fov//2 - (fov/pix)*g.x_mean.value for g in fits])
        ys = np.array([fov//2 - (fov/pix)*g.y_mean.value for g in fits])
        sigma = np.array([fov/pix*(g.x_stddev.value+g.y_stddev.value)/2 for g in fits])
        distance = np.sqrt(xs**2 + ys**2)
        angle = (np.rad2deg(-np.angle(xs + 1j*ys) + np.pi/2) + 180) % 360 - 180
        fwhm = 2.355 * sigma
        flux = np.array([g.amplitude.value * 2 * np.pi * g.x_stddev.value * g.y_stddev.value for g in fits])
        
        return pd.DataFrame({
            "time": times, "x": xs, "y": ys,
            "distance": distance, "angle": angle,
            "fwhm": fwhm, "flux": flux,
        })
    except Exception as e:
        return None

def main():
    args = create_parser().parse_args()
    
    # Expand input globs
    input_files = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        if matched:
            input_files.extend(matched)
        else:
            input_files.append(pattern)
    input_files = sorted(input_files)
    
    if not input_files:
        print("No input files found.")
        return
    print(f"Found {len(input_files)} input file(s).")

    # Load observation and truth to determine times
    print(f"Loading observation: {args.data}")
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            obs = eh.obsdata.load_uvfits(args.data)
    
    obs_times = np.unique(obs.data['time'])
    data_min_t = obs_times.min()
    data_max_t = obs_times.max()
    print(f"Data time range: {data_min_t:.3f} - {data_max_t:.3f} h")

    if args.tstart is not None or args.tstop is not None:
        tstart = args.tstart if args.tstart is not None else data_min_t
        tstop = args.tstop if args.tstop is not None else data_max_t
        print(f"Time flagging data to use in range: {tstart:.3f} - {tstop:.3f} h")
        
        with open(os.devnull, 'w') as devnull:
             with redirect_stdout(devnull), redirect_stderr(devnull):
                 obs = obs.flag_UT_range(UT_start_hour=tstart, UT_stop_hour=tstop, output='flagged')
        
        obs_times = np.unique(obs.data['time'])
        if len(obs_times) == 0:
            print("No data remaining after time flagging.")
            return
        
        print(f"New data time range: {obs_times.min():.3f} - {obs_times.max():.3f} h")
    
    times = obs_times
    
    min_t_list = []
    max_t_list = []
    
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            # Scan Inputs
            for m_path in input_files:
                mv = eh.movie.load_hdf5(m_path)
                min_t_list.append(min(mv.times))
                max_t_list.append(max(mv.times))
            # Scan Truth
            truth_mov = eh.movie.load_hdf5(args.truthmv)
            min_t_list.append(min(truth_mov.times))
            max_t_list.append(max(truth_mov.times))

    if not min_t_list:
        print("No valid movies found.")
        return

    movie_min_t = max(min_t_list)
    movie_max_t = min(max_t_list)
    print(f"Movie time range: {movie_min_t:.3f} - {movie_max_t:.3f} h")

    if movie_min_t > times.min() or movie_max_t < times.max():
         print("Warning: Movie times do not span the whole duration of data. Extrapolation will be used.")

    print(f"Processing {len(times)} time steps.")
    
    fov, pix = 200, 128
    
    # --- Process Truth ---
    df_truth = process_movie(args.truthmv, times, fov, pix)
    if df_truth is None:
        print("Failed to process truth movie.")
        return

    # --- Process Input Files ---
    results = []
    with ProcessPoolExecutor(max_workers=args.ncores) as executor:
        futures = [executor.submit(process_movie, f, times, fov, pix) for f in input_files]
        for fut in tqdm(futures, total=len(input_files), desc="Processing"):
            res = fut.result()
            if res is not None:
                results.append(res)
    
    if not results:
        print("No valid results obtained.")
        return

    # --- Compute Mean/Std or Single Values ---
    quantities = ["x", "y", "distance", "angle", "fwhm", "flux"]
    is_bayesian = len(results) > 1
    
    data_dict = {"time": times}
    
    for q in quantities:
        vals = np.array([df[q].values for df in results])
        if is_bayesian:
            data_dict[q+"_mean"] = vals.mean(axis=0)
            data_dict[q+"_std"] = vals.std(axis=0)
        else:
            data_dict[q] = vals[0]

    df_results = pd.DataFrame(data_dict)
    
    # --- Thresholds ---
    thresholds = {
        "x": 5, "y": 5, "distance": 5, "angle": 20,
        "fwhm": 5, "flux": 0.25,  # flux is fractional
    }

    # --- Compute Pass Percentages ---
    pass_percent = {}
    for q in quantities:
        truth_vals = df_truth[q].values
        
        if is_bayesian:
            val = df_results[q+"_mean"]
            std = df_results[q+"_std"]
        else:
            val = df_results[q]
            std = np.zeros_like(val)

        if q == "flux":
            lower_thr = truth_vals * (1 - thresholds["flux"])
            upper_thr = truth_vals * (1 + thresholds["flux"])
        else:
            lower_thr = truth_vals - thresholds[q]
            upper_thr = truth_vals + thresholds[q]

        # check overlap between (val-std, val+std) and (lower_thr, upper_thr)
        lower_res = val - std
        upper_res = val + std
        overlap = (upper_res >= lower_thr) & (lower_res <= upper_thr)
        pass_percent[q] = 100 * overlap.sum() / len(overlap)

    # --- Combine Data ---
    rename_dict = {col: col + '_truth' for col in df_truth.columns if col != 'time'}
    df_truth_renamed = df_truth.rename(columns=rename_dict)
    df_combined = df_results.merge(df_truth_renamed, on="time")

    for q in quantities:
        df_combined[q+"_threshold"] = thresholds[q] if q != "flux" else thresholds["flux"] * 100
        df_combined[q+"_pass_percent"] = pass_percent[q]

    # --- Save CSV ---
    csv_path = f"{args.outpath}_hotspot.csv"
    df_combined.to_csv(csv_path, index=False)
    print(f"✅ Saved results to {csv_path}")

    # --- Plot ---
    colors = {"truth": "k", "recon": "tab:blue"}
    ylabels = [
        "x ($\mu$as)", "y ($\mu$as)",
        "Distance ($\mu$as)", "PA ($^{\circ}$ E of N)",
        "FWHM ($\mu$as)", "Flux (Jy)"
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for i, (q, ylabel) in enumerate(zip(quantities, ylabels)):
        ax = axes[i]
        truth_vals = df_truth[q].values
        
        if is_bayesian:
            val = df_results[q+"_mean"]
            std = df_results[q+"_std"]
        else:
            val = df_results[q]
            std = None
            
        ax.scatter(df_truth["time"], truth_vals, color="k", lw=2, label="truth", s=10, alpha=0.7)

        # Threshold shading
        if q == "flux":
            upper = truth_vals * (1 + thresholds["flux"])
            lower = truth_vals * (1 - thresholds["flux"])
        else:
            upper = truth_vals + thresholds[q]
            lower = truth_vals - thresholds[q]
        ax.fill_between(df_truth["time"], lower, upper, color="gray", alpha=0.3, label="threshold")

        # mean ± std or single value
        if is_bayesian:
            ax.errorbar(df_results["time"], val, yerr=std,
                        fmt='o', color=colors["recon"], capsize=3, alpha=0.7)
        else:
            ax.plot(df_results["time"], val, 'o', color=colors["recon"], alpha=0.7)

        ax.set_ylabel(ylabel)
        ax.grid(True, ls='--', alpha=0.4)

        pct = pass_percent[q]
        ax.set_title(f"{q}: {pct:.1f}% pass", fontsize=18)

    axes[-1].set_xlabel("Time (UT)")
    axes[-2].set_xlabel("Time (UT)")
    axes[3].set_yticks([-180, -90, 0, 90, 180])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3)
    
    # Use basename of first input file or something generic for title
    title_name = os.path.basename(args.outpath)
    fig.suptitle(f"{title_name}: Hotspot Feature Extraction", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = f"{args.outpath}_hotspot.png"
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")

if __name__ == "__main__":
    main()
