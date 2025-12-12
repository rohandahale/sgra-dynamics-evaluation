###################################################################################
# Author: Rohan Dahale, Date: 10 December 2025
# Discussions with: Marianna Foschi, Antonio Fuentes, Aviad Levis, Kotaro Moriyama
# Istropy functions from: Joeseph Farah
###################################################################################

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
import scipy
import ehtim as eh
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor
import functools
import glob
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
from scipy.fft import fft2, ifft2, fftshift

# Set plot style
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.family"] = "sans-serif"


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, required=True, help='UVFITS data file')
    p.add_argument('--truthmv', type=str, required=True, help='Truth HDF5 movie file')
    p.add_argument('--input', type=str, nargs='+', required=True, help='Glob pattern(s) or list of HDF5 model files (e.g., "path/*.h5")')
    p.add_argument('-o', '--outpath', type=str, default='./nxcorr', help='Output prefix (without extension)')
    p.add_argument('--tstart', type=float, default=None, help='Start time (in UT hours) for data')
    p.add_argument('--tstop', type=float, default=None, help='Stop time (in UT hours) for data')
    p.add_argument('-n', '--ncores', type=int, default=32, help='Number of cores to use for parallel processing')
    return p

def compute_ramesh_metric(us, vs, N=None):

    if N is None:
        N = len(us)

    mean_u2 = np.sum([u**2. for u in us]) / (2.*N)
    mean_v2 = np.sum([v**2. for v in vs]) / (2.*N)
    mean_uv = np.sum([us[i]*vs[i] for i in range(len(us))]) / (2.*N)

    numerator = np.sqrt( (mean_u2-mean_v2)**2. + 4*(mean_uv**2.) )
    denominator = mean_u2 + mean_v2

    return 1 - numerator / denominator

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def radial_homogeneity(u, v):

    uvdists = np.sqrt(u**2. + v**2.)

    uvdists = sorted(uvdists)
    uvdists = list(uvdists)
    uvdists.append(1e10)
    uvdists.append(0e10)
    uvdists = np.array(uvdists)
    score = jensen_shannon_distance(uvdists, [i*1.e10/len(uvdists) for i in range(len(uvdists))])
    
    return score



def isotropy_metric_normalized(u, v, i_max=None, r_max=None):

    if i_max is None or r_max is None: print("Please provide a value for both i_max and r_max.")

    ## here we're just following the formula in the paper
    iso = compute_ramesh_metric(u, v)
    rad_hom = radial_homogeneity(u, v)

    return (iso/i_max) * (1 - rad_hom/r_max)


def get_weights(obs, times):
    """
    Calculate weights (w_norm) for the observation times.
    """

    obs.add_scans()
    obslist = obs.split_obs()
    
    # We assume 'times' passed in ensures obslist is aligned or we just use obslist directly.
    # The calling function handles time flagging and consistency.
    obslist_t = obslist
    
    I_scores = []
    snr = {'I': [], 'Q': [], 'U': [], 'V': []}
    
    imax = 1
    rmax = 0.513
    
    for s_obs in obslist_t:
        # Create a copy for flagging (used for I_scores)
        s_obs_flagged = s_obs.copy()
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                if 'AA' in s_obs_flagged.data['t1'] or 'AA' in s_obs_flagged.data['t2']: 
                    s_obs_flagged = s_obs_flagged.flag_sites('AP')
                if 'JC' in s_obs_flagged.data['t1'] or 'JC' in s_obs_flagged.data['t2']: 
                    s_obs_flagged = s_obs_flagged.flag_sites('SM')
            
        if len(s_obs_flagged.data) == 0:
            I_scores.append(0)
            for k in snr: snr[k].append(0)
            continue
            
        unpackedobj = np.transpose(s_obs_flagged.unpack(['u', 'v'], debias=True, conj=True))
        u = unpackedobj['u']
        v = unpackedobj['v']
        I_scores.append(isotropy_metric_normalized(u, v, i_max=imax, r_max=rmax))
        
        # SNR uses unflagged original s_obs
        df = pd.DataFrame(s_obs.data)
        snr['I'].append(np.mean(np.abs(df['vis'])/df['sigma']))
        snr['Q'].append(np.mean(np.abs(df['qvis'])/df['qsigma']))
        snr['U'].append(np.mean(np.abs(df['uvis'])/df['usigma']))
        snr['V'].append(np.mean(np.abs(df['vvis'])/df['vsigma']))
        
    I_scores = np.array(I_scores)
    if np.max(I_scores) > 0:
        I_scores = I_scores / np.max(I_scores)
        
    for x in snr.keys():
        snr[x] = np.array(snr[x])
        if np.max(snr[x]) != np.min(snr[x]):
             snr[x] = np.min(I_scores) + ((snr[x] - np.min(snr[x])) / (np.max(snr[x]) - np.min(snr[x]))) * (np.max(I_scores) - np.min(I_scores))
        else:
             snr[x] = I_scores
             
    w_norm = {}
    for x in ['I', 'Q', 'U', 'V']:
        w = I_scores * snr[x]
        w_sum = np.sum(w)
        if w_sum > 0:
            w_norm[x] = np.array(w / w_sum)
        else:
            w_norm[x] = np.zeros_like(w)
            
    return obslist_t, w_norm

def rotate_evpa(im, angle):
    im2 = im.copy()
    angle = np.deg2rad(angle)
    chi = np.angle(im2.qvec + 1j * im2.uvec) / 2 + angle
    m = np.abs(im2.qvec + 1j * im2.uvec / im2.ivec)
    i = im2.ivec
    im2.qvec = i * m * np.cos(2 * chi)
    im2.uvec = i * m * np.sin(2 * chi)
    return im2

def pnxcorr(im_truth, im_recon, npix, fov, beam, shift=None, truth_chi_rot=20):
    # 1. Regrid images
    imt = im_truth.regrid_image(fov, npix)
    imr = im_recon.regrid_image(fov, npix)

    # 2. Construct Complex Polarization Vectors (P = Q + iU)
    P_truth = imt.qvec.reshape(npix, npix) + 1j * imt.uvec.reshape(npix, npix)
    P_recon = imr.qvec.reshape(npix, npix) + 1j * imr.uvec.reshape(npix, npix)

    # 3. Compute Normalization Factor
    norm_factor = np.sqrt(np.sum(np.abs(P_recon)**2)) * np.sqrt(np.sum(np.abs(P_truth)**2))
    
    if norm_factor == 0:
        return 0.0, 0.0

    # 4. FFT-based Cross-Correlation
    f_truth = fft2(P_truth)
    f_recon = fft2(P_recon)
    
    # Compute cross-correlation map
    cc_map = ifft2(f_recon * np.conj(f_truth))
    
    # Normalize the map
    nxcorr_map = cc_map / norm_factor

    # 5. Determine Alignment
    if shift is None:
        # SEARCH MODE: Find the shift that maximizes the correlation magnitude
        best_idx = np.unravel_index(np.argmax(np.abs(nxcorr_map)), nxcorr_map.shape)
    else:
        # NO SHIFT MODE: Use index (0,0) which corresponds to zero lag
        best_idx = (0, 0)

    # Extract the complex correlation at the determined index
    max_complex_corr = nxcorr_map[best_idx]

    # 6. Return the Real part (penalizes EVPA misalignment)
    evpa_corr = np.real(max_complex_corr)
    
    # Threshold calculation 
    chi_rot_rad = np.deg2rad(truth_chi_rot)
    phase_threshold = np.cos(2 * chi_rot_rad)

    return evpa_corr, phase_threshold

def enxcorr(im_truth, im_recon, npix, fov, beam, shift=None, truth_chi_rot=20, min_P_mask_frac=0.05):
    # Regrid images
    imt = im_truth.regrid_image(fov, npix)
    imr = im_recon.regrid_image(fov, npix)

    # Q/U arrays
    Q_truth = imt.qvec.reshape(npix, npix)
    U_truth = imt.uvec.reshape(npix, npix)
    Q_recon = imr.qvec.reshape(npix, npix)
    U_recon = imr.uvec.reshape(npix, npix)

    # Truth amplitude masking
    P_amp_truth = np.sqrt(Q_truth**2 + U_truth**2)
    P_peak = np.max(P_amp_truth)
    mask = P_amp_truth >= (P_peak * min_P_mask_frac)
    
    # EVPA decomposition
    zeta_truth = 0.5 * np.arctan2(U_truth, Q_truth)
    zeta_recon = 0.5 * np.arctan2(U_recon, Q_recon)
    Pdir_truth = np.exp(1j * 2 * zeta_truth) * mask
    Pdir_recon = np.exp(1j * 2 * zeta_recon) # mask is not used here for recon

    # FFT-based Cross-Correlation
    C_corr = ifft2(fft2(Pdir_recon) * np.conj(fft2(Pdir_truth)))

    norm_factor = np.sum(mask)
    if norm_factor > 0:
        C_corr /= norm_factor

    if shift is None:
        # SEARCH MODE
        absC = np.abs(C_corr)
        best_idx = np.unravel_index(np.argmax(absC), absC.shape)
    else:
        # NO SHIFT MODE (Zero lag at 0,0)
        best_idx = (0, 0)

    max_C = C_corr[best_idx]
    
    evpa_corr = np.real(max_C)
    chi_rot_rad = np.deg2rad(truth_chi_rot)
    phase_threshold = np.cos(2 * chi_rot_rad)

    return evpa_corr, phase_threshold


def get_nxcorr_cri_beam(im, beam, pol='I'):
    im_blur = im.blur_gauss(beam, frac=1.0, frac_pol=1.0)
    nxcorr_cri = im.compare_images(im_blur, pol=pol, metric=['nxcorr'])[0][0]
    return nxcorr_cri


def process_frame(args_tuple):
    imt, im, beam, npix, fov, pol, shift = args_tuple
    if pol == 'I':
        if shift is not None:
            nxcorr = imt.compare_images(im, pol=pol, metric=['nxcorr'], shift=shift)[0][0]
        else:
            nxcorr = imt.compare_images(im, pol=pol, metric=['nxcorr'])[0][0]
        threshold = get_nxcorr_cri_beam(imt, beam, pol='I')
    elif pol == 'Pmag':
        im2 = im.copy()
        imt2 = imt.copy()
        im2.ivec = np.sqrt(im2.qvec**2 + im2.uvec**2)
        imt2.ivec = np.sqrt(imt2.qvec**2 + imt2.uvec**2)
        if shift is not None:
            nxcorr = imt2.compare_images(im2, pol='I', metric=['nxcorr'], shift=shift)[0][0]
        else:
            nxcorr = imt2.compare_images(im2, pol='I', metric=['nxcorr'])[0][0]
        threshold = get_nxcorr_cri_beam(imt2, beam, pol='I')
    elif pol == 'Pvec':
        if shift is not None:
            nxcorr, threshold = pnxcorr(imt, im, npix, fov, beam, shift=shift)
        else:
            nxcorr, threshold = pnxcorr(imt, im, npix, fov, beam)
    elif pol == 'X':
        if shift is not None:
            nxcorr, threshold = enxcorr(imt, im, npix, fov, beam, shift=shift)
        else:
            nxcorr, threshold = enxcorr(imt, im, npix, fov, beam)

    else:
        raise ValueError(f"Invalid polarization: {pol}")
    return nxcorr, threshold

def process_movie_nxcorr(m_path, mvt, times, obslist_t, npix, fov, pol, mode, w_norm):
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            mv = eh.movie.load_hdf5(m_path)
            mv.reset_interp(bounds_error=False)
            im_list = []
            imt_list = []
            beams = []
            for i, t in enumerate(times):
                im = mv.get_image(t).regrid_image(fov, npix)
                imt = mvt.get_image(t).regrid_image(fov, npix)
                beam = obslist_t[i].fit_beam(weighting='uniform')
                if mode == 'dynamic':
                    shift = imt.align_images([im])[1]
                    im = im.shift(shift[0])
                im_list.append(im)
                imt_list.append(imt)
                beams.append(beam)

            nxcorr_values = []
            raw_thresholds = []
            
            if mode == 'static':
                def get_stack(images, p):
                    return np.array([img.imarr(pol=p) for img in images])
                median_im = im_list[0].copy()
                median_imt = imt_list[0].copy()
                for p_char in ['I', 'Q', 'U']:
                    med = np.median(get_stack(im_list, p_char), axis=0)
                    med_t = np.median(get_stack(imt_list, p_char), axis=0)
                    if p_char == 'I': 
                        median_im.ivec = med.flatten()
                        median_imt.ivec = med_t.flatten()
                    elif p_char == 'Q':
                        median_im.qvec = med.flatten()
                        median_imt.qvec = med_t.flatten()
                    elif p_char == 'U':
                        median_im.uvec = med.flatten()
                        median_imt.uvec = med_t.flatten()
                shift = median_imt.align_images([median_im])[1]
                median_im = median_im.shift(shift[0])
                res_nxcorr, _ = process_frame((median_imt, median_im, beams[0], npix, fov, pol, None))
                for i in range(len(times)):
                    _, res_thres = process_frame((median_imt, median_im, beams[i], npix, fov, pol, None))
                    nxcorr_values.append(res_nxcorr)
                    raw_thresholds.append(res_thres)
            elif mode == 'dynamic':
                def get_stack(images, p):
                    return np.array([img.imarr(pol=p) for img in images])
                median_im = im_list[0].copy()
                median_imt = imt_list[0].copy()
                for p_char in ['I', 'Q', 'U']:
                    med = np.median(get_stack(im_list, p_char), axis=0)
                    med_t = np.median(get_stack(imt_list, p_char), axis=0)
                    if p_char == 'I': 
                        median_im.ivec = med.flatten()
                        median_imt.ivec = med_t.flatten()
                    elif p_char == 'Q':
                        median_im.qvec = med.flatten()
                        median_imt.qvec = med_t.flatten()
                    elif p_char == 'U':
                        median_im.uvec = med.flatten()
                        median_imt.uvec = med_t.flatten()
                for i in range(len(times)):
                    im_res = im_list[i].copy()
                    imt_res = imt_list[i].copy()
                    for p_char in ['I', 'Q', 'U']:
                        val = im_list[i].imarr(pol=p_char) - median_im.imarr(pol=p_char)
                        val_t = imt_list[i].imarr(pol=p_char) - median_imt.imarr(pol=p_char)
                        if p_char == 'I':
                            im_res.ivec = np.clip(val.flatten(), 1e-7, None)
                            imt_res.ivec = np.clip(val_t.flatten(), 1e-7, None)
                        elif p_char == 'Q':
                            im_res.qvec = val.flatten()
                            imt_res.qvec = val_t.flatten()
                        elif p_char == 'U':
                            im_res.uvec = val.flatten()
                            imt_res.uvec = val_t.flatten()
                    res = process_frame((imt_res, im_res, beams[i], npix, fov, pol, [0,0]))
                    nxcorr_values.append(res[0])
                    raw_thresholds.append(res[1])
            else:
                for i in range(len(times)):
                    res = process_frame((imt_list[i], im_list[i], beams[i], npix, fov, pol, None))
                    nxcorr_values.append(res[0])
                    raw_thresholds.append(res[1])

            if pol == 'I':
                w_ratio = w_norm['I'] / np.max(w_norm['I'])
            elif pol in ['Pmag', 'X', 'Pvec']:
                w_QU = (w_norm['Q'] + w_norm['U']) / 2
                w_ratio = w_QU / np.max(w_QU)
            else:
                w_ratio = np.ones(len(times))

            # Apply weighting
            weighted_thresholds = w_ratio * np.array(raw_thresholds)
            
            # For static mode, take mean of weighted thresholds and replicate
            if mode == 'static':
                mean_weighted_threshold = np.mean(weighted_thresholds)
                weighted_thresholds = np.ones(len(times)) * mean_weighted_threshold
            
            pass_rate = None
            if mode != 'static':
                diff = np.array(nxcorr_values) - weighted_thresholds
                pass_count = np.sum(diff > 0)
                pass_rate = (pass_count / len(nxcorr_values)) * 100
                
            return {
                'nxcorr': nxcorr_values,
                'threshold': weighted_thresholds,
                'pass_rate': pass_rate
            }

def save_and_plot(times, all_metrics_data, mode, outpath, is_bayesian):
    """
    Save consolidated CSV and create 4-panel plot (I, Pmag, X, Pvec).
    all_metrics_data: dict {pol: metrics_data_list}
    """
    results = {'time': times}
    
    # Process each polarization
    pols_to_plot = ['I', 'Pmag', 'X', 'Pvec']
    
    for pol in pols_to_plot:
        if pol not in all_metrics_data:
            continue
            
        metrics_data = all_metrics_data[pol]
        all_nxcorr = np.array([m['nxcorr'] for m in metrics_data])
        all_thres = np.array([m['threshold'] for m in metrics_data])
        
        if is_bayesian:
            results[f'nxcorr_{pol}_mean'] = np.mean(all_nxcorr, axis=0)
            results[f'nxcorr_{pol}_std'] = np.std(all_nxcorr, axis=0)
            # Threshold is const, use mean
            results[f'nxcorr_{pol}_thres'] = np.mean(all_thres, axis=0)
            
            if mode != 'static':
                # Pass if (mean + std) > threshold
                # User request: "any part of errorbar above threshold is a pass"
                pass_condition = (results[f'nxcorr_{pol}_mean'] + results[f'nxcorr_{pol}_std']) > results[f'nxcorr_{pol}_thres']
                pass_rate = (np.sum(pass_condition) / len(times)) * 100
                results[f'pass_rate_{pol}'] = np.full(len(times), pass_rate)
        else:
            results[f'nxcorr_{pol}'] = all_nxcorr[0]
            results[f'nxcorr_{pol}_thres'] = all_thres[0]
            if mode != 'static':
                results[f'pass_rate_{pol}'] = np.full(len(times), metrics_data[0]['pass_rate'])

    # Save CSV
    csv_path = f"{outpath}_{mode}_nxcorr.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 5), sharex=True)
    
    color = 'tab:blue'
    
    for i, pol in enumerate(pols_to_plot):
        ax = axes[i]
        ax.set_ylabel(f'nxcorr ({pol})')
        ax.set_xlabel('Time (UTC)')
        
        if pol not in all_metrics_data:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            continue
            
        if is_bayesian:
            ax.errorbar(times, results[f'nxcorr_{pol}_mean'], 
                       yerr=results[f'nxcorr_{pol}_std'], 
                       fmt='-o', label='Mean', color=color, capsize=3, alpha=0.8)
            # Threshold as simple line
            ax.plot(times, results[f'nxcorr_{pol}_thres'], 
                       '--', label='Threshold', color='black', alpha=0.6)
            
            if mode != 'static':
                pass_str = f"Pass: {results[f'pass_rate_{pol}'][0]:.1f}%"
                ax.plot([], [], ' ', label=pass_str)
        else:
            ax.plot(times, results[f'nxcorr_{pol}'], '-o', color=color, label='nxcorr')
            ax.plot(times, results[f'nxcorr_{pol}_thres'], '--', color='black', label='Threshold')
            if mode != 'static':
                pass_str = f"Pass: {results[f'pass_rate_{pol}'][0]:.1f}%"
                ax.plot([], [], ' ', label=pass_str)
        
        ax.legend()

    plt.tight_layout()
    png_path = f"{outpath}_{mode}_nxcorr.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {png_path}")
    plt.close()

def main():
    args = create_parser().parse_args()
    input_files = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        if matched:
            input_files.extend(matched)
        else:
            input_files.append(pattern)
    if not input_files:
        print("No input files found.")
        return
    print(f"Found {len(input_files)} input file(s).")
    
    print(f"Loading observation: {args.data}")
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            obs = eh.obsdata.load_uvfits(args.data)
            obs = obs.avg_coherent(60)
            
    # Calculate data time range
    obs.add_scans()
    obslist = obs.split_obs() 
    obs_times = np.array([o.data['time'][0] for o in obslist])
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
                 obs.add_scans()
                 obslist = obs.split_obs()
        
        if not obslist:
            print("No data remaining after time flagging.")
            return

        obs_times = np.array([o.data['time'][0] for o in obslist])
        print(f"New data time range: {obs_times.min():.3f} - {obs_times.max():.3f} h")

    times = obs_times

    min_t_list = []
    max_t_list = []
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for m_path in input_files:
                mv = eh.movie.load_hdf5(m_path)
                min_t_list.append(min(mv.times))
                max_t_list.append(max(mv.times))
            mvt = eh.movie.load_hdf5(args.truthmv)
            mvt.reset_interp(bounds_error=False)
            min_t_list.append(min(mvt.times))
            max_t_list.append(max(mvt.times))
    
    movie_min_t = max(min_t_list)
    movie_max_t = min(max_t_list)
    print(f"Movie time range: {movie_min_t:.3f} - {movie_max_t:.3f} h")

    if movie_min_t > times.min() or movie_max_t < times.max():
         print("Warning: Movie times do not span the whole duration of data. Extrapolation will be used.")

    print(f"Processing {len(times)} time steps.")
    
    obslist_t, w_norm = get_weights(obs, times)
    
    npix = 200
    fov = 200 * eh.RADPERUAS
    
    modes = ['total', 'static', 'dynamic']
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Processing Mode: {mode}")
        print(f"{'='*60}")
        
        all_metrics_data = {}
        
        for pol in ['I', 'Pmag', 'X', 'Pvec']:
            print(f"  Processing polarization: {pol}")
            metrics_data = []
            process_func = functools.partial(process_movie_nxcorr, 
                                            mvt=mvt, times=times, 
                                            obslist_t=obslist_t, npix=npix, 
                                            fov=fov, pol=pol, mode=mode, w_norm=w_norm)
            max_workers = args.ncores
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results_gen = executor.map(process_func, input_files)
                for res in tqdm(results_gen, total=len(input_files), desc=f"{pol} {mode}"):
                    if res is not None:
                        metrics_data.append(res)
            
            if metrics_data:
                all_metrics_data[pol] = metrics_data
        
        if not all_metrics_data:
            continue
            
        # Determine if bayesian based on first pol
        first_pol = list(all_metrics_data.keys())[0]
        is_bayesian = len(all_metrics_data[first_pol]) > 1
        
        save_and_plot(times, all_metrics_data, mode, args.outpath, is_bayesian)

if __name__ == "__main__":
    main()