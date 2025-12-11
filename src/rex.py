######################################################################
# Author: Rohan Dahale, Date: 10 Dec 2025
# Based on: Kotaro Moriyama's rex.py
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
from scipy import interpolate, stats
from copy import copy, deepcopy
import itertools
from astropy.constants import k_B, c
import astropy.units as u
from ehtim.const_def import *

# Set plot style
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.family"] = "sans-serif"

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, required=True, help='UVFITS data file')
    p.add_argument('--truthmv', type=str, default=None, help='Truth HDF5 movie file (optional)')
    p.add_argument('--input', type=str, nargs='+', required=True, help='Glob pattern(s) or list of HDF5 model files')
    p.add_argument('-o', '--outpath', type=str, default='./rex', help='Output prefix (without extension)')
    p.add_argument('-n', '--ncores', type=int, default=32, help='Number of cores to use for parallel processing')
    return p

######################################################################
# REx functions
######################################################################

def calculate_true_d_error(D, W, D_err, W_err):
    ln2 = np.log(2)
    ratio = W / D
    ratio_sq = ratio**2
    common_denominator = (1 - (1 / (4 * ln2)) * ratio_sq)**2
    partial_d = (1 - (3 / (4 * ln2)) * ratio_sq) / common_denominator
    partial_w = ((1 / (2 * ln2)) * ratio) / common_denominator
    d_err_term_sq = np.square(partial_d * D_err)
    w_err_term_sq = np.square(partial_w * W_err)
    true_d_err = np.sqrt(d_err_term_sq + w_err_term_sq)
    return true_d_err

def quad_interp_radius(r_max, dr, val_list):
    v_L = val_list[0]
    v_max = val_list[1]
    v_R = val_list[2]
    rpk = r_max + dr*(v_L - v_R) / (2 * (v_L + v_R - 2*v_max))
    vpk = 8*v_max*(v_L + v_R) - (v_L - v_R)**2 - 16*v_max**2
    vpk /= (8*(v_L + v_R - 2*v_max))
    return (rpk, vpk)

def calc_width(tmpIr,radial,rpeak):
    spline = interpolate.UnivariateSpline(radial, tmpIr-0.5*tmpIr.max(), s=0)
    roots = spline.roots()
    if len(roots) == 0:
        return(radial[0], radial[-1])
    rmin = radial[0]
    rmax = radial[-1]
    for root in np.sort(roots):
        if root < rpeak:
            rmin = root
        else:
            rmax = root
            break
    return (rmin, rmax)

def fit_ring(image,Nr=50,Npa=25,rmin_search = 10,rmax_search = 100,fov_search = 0.1,Nserch =20):
    image_blur = image.blur_circ(2.0*RADPERUAS,fwhm_pol=0)
    xc,yc = eh.features.rex.findCenter(image, rmin_search=rmin_search, rmax_search=rmax_search,
                         nrays_search=Npa, nrs_search=Nr,
                         fov_search=fov_search, n_search=Nserch)
    return xc,yc

def extract_hole(image,Nxc,Nyc, r=30):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2>=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)
    return outimage

def extract_outer(image,Nxc,Nyc, r=30):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2<=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)
    return outimage

def extract_ring(image, Nxc,Nyc,rin=30,rout=50):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - rin**2<=0)] = 0
    masked[np.where(x**2 + y**2 - rout**2>=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)
    return outimage

def extract_ring_quantites(image,xc=None,yc=None, rcutoff=5):
    Npa=360
    Nr=100

    if xc==None or yc==None:
        xc,yc = fit_ring(image)
    
    x= np.arange(image.xdim)*image.psize/RADPERUAS
    # y must be strictly increasing for RectBivariateSpline
    # Original code had y flipped (decreasing). 
    # If z[0] corresponds to y_max, and we make y increasing (y_min to y_max),
    # we must flip z along axis 0 so z[0] corresponds to y_min.
    y= np.arange(image.ydim)*image.psize/RADPERUAS
    z = image.imarr()
    z = np.flipud(z)
    
    f_image = interpolate.RectBivariateSpline(y, x, z)

    radial_imarr = np.zeros([Nr,Npa])

    pa = np.linspace(0,360,Npa)
    pa_rad = np.deg2rad(pa)
    radial = np.linspace(0,50,Nr)
    dr = radial[-1]-radial[-2]

    Rmesh, PAradmesh = np.meshgrid(radial, pa_rad)
    x_grid = Rmesh*np.sin(PAradmesh) + xc
    y_grid = Rmesh*np.cos(PAradmesh) + yc
    
    # Vectorized interpolation
    # RectBivariateSpline expects y, x. 
    # We need to evaluate at specific points. ev(y, x)
    # Flatten grids
    z_interp = f_image.ev(y_grid.flatten(), x_grid.flatten())
    radial_imarr = z_interp.reshape(Npa, Nr).T
    radial_imarr = np.fliplr(radial_imarr)

    peakpos = np.unravel_index(np.argmax(radial_imarr), shape=radial_imarr.shape)

    Rpeak=[]
    Rmin=[]
    Rmax=[]
    ridx_r50= np.argmin(np.abs(radial - 50))
    I_floor = radial_imarr[ridx_r50,:].mean()
    for ipa in range(len(pa)):
        tmpIr = copy(radial_imarr[:,ipa])-I_floor
        tmpIr[np.where(radial < rcutoff)]=0
        ridx_pk = np.argmax(tmpIr)
        rpeak = radial[ridx_pk]
        if ridx_pk > 0 and ridx_pk < Nr-1:
            val_list= tmpIr[ridx_pk-1:ridx_pk+2]
            rpeak = quad_interp_radius(rpeak, dr, val_list)[0]
        Rpeak.append(rpeak)
        rmin,rmax = calc_width(tmpIr,radial,rpeak)
        Rmin.append(rmin)
        Rmax.append(rmax)
    
    paprofile = pd.DataFrame()
    paprofile["PA"] = pa
    paprofile["rpeak"] = Rpeak
    paprofile["rhalf_max"]=Rmax
    paprofile["rhalf_min"]=Rmin

    D = np.mean(paprofile["rpeak"]) * 2
    Derr = paprofile["rpeak"].std() * 2
    W = np.mean(paprofile["rhalf_max"] - paprofile["rhalf_min"])
    Werr =  (paprofile["rhalf_max"] - paprofile["rhalf_min"]).std()

    rin  = D/2.-W/2.
    rout  = D/2.+W/2.
    if rin <= 0.:
        rin  = 0.

    exptheta =np.exp(1j*pa_rad)

    pa_ori_r=[]
    amp_r = []
    ridx1 = np.argmin(np.abs(radial - rin))
    ridx2 = np.argmin(np.abs(radial - rout))
    for r in range(ridx1, ridx2+1, 1):
        amp =  (radial_imarr[r,:]*exptheta).sum()/(radial_imarr[r,:]).sum()
        amp_r.append(amp)
        pa_ori = np.angle(amp, deg=True)
        pa_ori_r.append(pa_ori)
    pa_ori_r=np.array(pa_ori_r)
    amp_r = np.array(amp_r)
    PAori = stats.circmean(pa_ori_r,high=360,low=0)
    PAerr = stats.circstd(pa_ori_r,high=360,low=0)
    A = np.mean(np.abs(amp_r))
    Aerr = np.std(np.abs(amp_r))

    ridx_r5= np.argmin(np.abs(radial - 5))
    ridx_pk = np.argmin(np.abs(radial - D/2))
    fc = radial_imarr[0:ridx_r5,:].mean()/radial_imarr[ridx_pk,:].mean()

    fwhm_maj,fwhm_min,theta = image.fit_gauss()
    fwhm_maj /= RADPERUAS
    fwhm_min /= RADPERUAS

    Nxc = int(xc/image.psize*RADPERUAS)
    Nyc = int(yc/image.psize*RADPERUAS)
    hole = extract_hole(image,Nxc,Nyc,r=rin)
    ring = extract_ring(image,Nxc,Nyc,rin=rin, rout=rout)
    outer = extract_outer(image,Nxc,Nyc,r=rout)
    hole_flux = hole.total_flux()
    outer_flux = outer.total_flux()
    ring_flux = ring.total_flux()

    Shole  = np.pi*rin**2
    Souter = (2.*rout)**2.-np.pi*rout**2
    Sring = np.pi*rout**2-np.pi*rin**2

    Shole = Shole*RADPERUAS**2
    Souter = Souter*RADPERUAS**2
    Sring = Sring*RADPERUAS**2

    freq = image.rf*u.Hz
    hole_dflux  = hole_flux/Shole*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    outer_dflux = outer_flux/Souter*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    ring_dflux = ring_flux/Sring*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    
    true_D=np.array(D/(1-(1/(4*np.log(2)))*(W/D)**2))
    true_Derr = calculate_true_d_error(D, W, Derr, Werr)

    outputs = dict(
        time = image.time,
        papeak=pa[peakpos[1]],
        xc=xc,
        yc=yc,
        PAori = PAori,
        PAerr = PAerr,
        A = A,
        Aerr = Aerr,
        fc = fc,
        D = D,
        Derr = Derr,
        W = W,
        Werr = Werr,
        true_D = true_D,
        true_Derr = true_Derr,
        fwhm_maj=fwhm_maj,
        fwhm_min=fwhm_min,
        hole_flux = hole_flux,
        outer_flux = outer_flux,
        ring_flux = ring_flux,
        totalflux = image.total_flux(),
        hole_dflux = hole_dflux,
        outer_dflux = outer_dflux,
        ring_dflux = ring_dflux
    )
    return outputs

def make_polar_imarr(imarr, dx, xc=None, yc=None, rmax=50, Nr=50, Npa=180, kind="linear", image=None):
    nx,ny = imarr.shape
    dy=dx
    x= np.arange(nx)*dx/RADPERUAS
    y= np.arange(ny)*dy/RADPERUAS
    if xc==None or yc==None:
        xc,yc = fit_ring(image)

    z = imarr
    # f_image = interpolate.interp2d(x,y,z,kind=kind) # Deprecated
    f_image = interpolate.RectBivariateSpline(y, x, z)

    radial_imarr = np.zeros([Nr,Npa])
    pa = np.linspace(0,360,Npa)
    pa_rad = np.deg2rad(pa)
    radius = np.linspace(0,rmax,Nr)
    dr = radius[-1]-radius[-2]

    Rmesh, PAradmesh = np.meshgrid(radius, pa_rad)
    x_grid, y_grid = Rmesh*np.sin(PAradmesh) + xc, Rmesh*np.cos(PAradmesh) + yc
    
    # Vectorized
    z_interp = f_image.ev(y_grid.flatten(), x_grid.flatten())
    radial_imarr = z_interp.reshape(Npa, Nr).T
    radial_imarr = np.fliplr(radial_imarr)

    return radial_imarr,radius, pa

def extract_pol_quantites(im,xc=None, yc=None, blur_size=-1):
    Itot, Qtot, Utot = sum(im.imvec), sum(im.qvec), sum(im.uvec)
    if len(im.vvec)==0:
        im.vvec = np.zeros_like(im.imvec)
    Vtot = sum(im.vvec)
    mnet=np.sqrt(Qtot*Qtot + Utot*Utot)/Itot

    if blur_size<0:
        mavg = sum(np.sqrt(im.qvec**2 + im.uvec**2))/Itot
    else:
        im_blur = im.blur_circ(blur_size*eh.RADPERUAS, fwhm_pol=blur_size*eh.RADPERUAS)
        mavg = sum(np.sqrt(im_blur.qvec**2 + im_blur.uvec**2))/np.sum(im_blur.imvec)

    evpa =  (180./np.pi)*0.5*np.angle(Qtot+1j*Utot)
    vnet = np.abs(Vtot)/Itot

    P = im.qvec+ 1j*im.uvec
    P_radial, radius, pa = make_polar_imarr(P.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    I_radial, dummy, dummy = make_polar_imarr(im.imvec.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    V_radial, dummy, dummy = make_polar_imarr(im.vvec.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    
    Pring, Vring, Vring2, Iring = 0, 0, 0, 0
    
    # Vectorized integration
    # pa is 1D, radius is 1D
    # P_radial is (Nr, Npa)
    
    # Create grids for broadcasting
    radius_grid = radius[:, np.newaxis]
    pa_grid = pa[np.newaxis, :]
    
    exp_minus_2j = np.exp(-2*1j*np.deg2rad(pa_grid))
    exp_minus_1j = np.exp(-1*1j*np.deg2rad(pa_grid))
    
    Pring = np.sum(P_radial * exp_minus_2j * radius_grid)
    Vring2 = np.sum(V_radial * exp_minus_2j * radius_grid)
    Vring = np.sum(V_radial * exp_minus_1j * radius_grid)
    Iring = np.sum(I_radial * radius_grid)

    beta2 = Pring/Iring
    beta2_abs, beta2_angle = np.abs(beta2), np.rad2deg(np.angle(beta2))

    beta2_v = Vring2/Iring
    beta2_v_abs, beta2_v_angle = np.abs(beta2_v), np.rad2deg(np.angle(beta2_v))
    beta_v = Vring/Iring
    beta_v_abs, beta_v_angle = np.abs(beta_v), np.rad2deg(np.angle(beta_v))

    outputs = dict(
        time_utc = im.time,
        mnet = mnet,
        mavg = mavg,
        evpa = evpa,
        beta2_abs = beta2_abs,
        beta2_angle = beta2_angle,
        vnet = vnet,
        beta_v_abs = beta_v_abs,
        beta_v_angle = beta_v_angle,
        beta2_v_abs = beta2_v_abs,
        beta2_v_angle = beta2_v_angle
        )
    return outputs

######################################################################
# Main
######################################################################

def process_movie(m_path, times):
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            mv = eh.movie.load_hdf5(m_path)
            
            # Average frame for center finding
            mv_ave = mv.avg_frame()
            xc, yc = fit_ring(mv_ave)
            
            results = []
            
            for t in times:
                im = mv.get_image(t)
                
                # Ring quantities
                ring_out = extract_ring_quantites(im, xc=xc, yc=yc)
                
                # Pol quantities
                pol_out = {}
                if len(im.qvec) > 0 and len(im.uvec) > 0:
                    pol_out = extract_pol_quantites(im, xc=xc, yc=yc)
                
                # Merge dictionaries
                combined = {**ring_out, **pol_out}
                results.append(combined)
                
    return results

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
    
    # Load observation for times
    print(f"Loading observation: {args.data}")
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            obs = eh.obsdata.load_uvfits(args.data)
            obs.add_scans()
            obslist = obs.split_obs()
            
    obs_times = np.array([o.data['time'][0] for o in obslist])
    
    # Scan movies for time range
    min_t_list = []
    max_t_list = []
    print("Scanning movies for time range...")
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for m_path in input_files:
                mv = eh.movie.load_hdf5(m_path)
                min_t_list.append(min(mv.times))
                max_t_list.append(max(mv.times))
            
            # Also check truth movie if provided
            if args.truthmv:
                mv_truth = eh.movie.load_hdf5(args.truthmv)
                min_t_list.append(min(mv_truth.times))
                max_t_list.append(max(mv_truth.times))
                
    if not min_t_list:
        print("No valid movies found.")
        return

    min_t = max(min_t_list)
    max_t = min(max_t_list)
    
    valid_indices = np.where((obs_times >= min_t) & (obs_times <= max_t))[0]
    times = obs_times[valid_indices]
    
    print(f"Processing {len(times)} time steps from {min_t} to {max_t}.")
    
    # Process Truth Movie
    truth_results = None
    if args.truthmv:
        print("Processing truth movie...")
        truth_results = process_movie(args.truthmv, times)
        # truth_results is a list of dicts
    
    # Parallel Processing for Input Movies
    max_workers = args.ncores
    print(f"Starting parallel processing with {max_workers} workers...")
    
    movie_results = []
    process_func = functools.partial(process_movie, times=times)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_gen = executor.map(process_func, input_files)
        for res in tqdm(results_gen, total=len(input_files), desc="Processing movies"):
            if res is not None:
                movie_results.append(res)
                
    if not movie_results:
        print("No results generated.")
        return
    
    # Get all keys from first result
    keys = list(movie_results[0][0].keys())
    
    # Prepare data for DataFrame
    data = {'time': times}
    
    # Add truth data if available
    if truth_results:
        for key in keys:
            if key == 'time' or key == 'time_utc': continue
            truth_values = np.array([t.get(key, np.nan) for t in truth_results])
            data[f'{key}_truth'] = truth_values
    
    is_bayesian = len(movie_results) > 1
    
    for key in keys:
        if key == 'time' or key == 'time_utc': continue
        
        # Gather all values for this key across all movies and times
        # Shape: (n_movies, n_times)
        values = np.array([[m[t_idx].get(key, np.nan) for t_idx in range(len(times))] for m in movie_results])
        
        if is_bayesian:            
            data[f'{key}_mean'] = np.nanmean(values, axis=0)
            data[f'{key}_std'] = np.nanstd(values, axis=0)
            
            err_key = None
            if key + 'err' in keys:
                err_key = key + 'err'
            elif key == 'PAori': err_key = 'PAerr'
            
            if err_key:
                err_values = np.array([[m[t_idx].get(err_key, np.nan) for t_idx in range(len(times))] for m in movie_results])
                # Propagate error: sqrt(std(values)^2 + mean(err_values^2))
                mean_sq_err = np.nanmean(err_values**2, axis=0)
                total_std = np.sqrt(data[f'{key}_std']**2 + mean_sq_err)
                data[f'{key}_std'] = total_std # Update std to include propagated error
                
        else:
            data[key] = values[0]
            
    df = pd.DataFrame(data)
    
    # Calculate Pass Percentages if truth is available
    pass_percentages = {}
    threshold_arrays = {} # Store threshold arrays for plotting
    
    if truth_results:
        # Define thresholds
        # Absolute thresholds
        abs_thresholds = {
            'D': 5.0,
            'W': 5.0,
            'papeak': 26.0,
            'beta2_angle': 26.0
        }
        
        # Relative thresholds (10%)
        rel_thresholds = ['A', 'true_D'] 
        # Add polarization magnitudes if present
        pol_mags = ['mnet', 'mavg', 'beta2_abs', 'vnet']
        for pm in pol_mags:
            if f'{pm}_mean' in df.columns or pm in df.columns:
                rel_thresholds.append(pm)
                
        if 'fc_mean' in df.columns or 'fc' in df.columns:
            rel_thresholds.append('fc')

        # PA Threshold (Scaled)
        A0 = 0.7184071604180173
        pa_threshold0 = 26
        recon_A = df['A_mean'] if is_bayesian else df['A']
        recon_A_safe = np.where(recon_A == 0, 1e-6, recon_A)
        pa_threshold_arr = pa_threshold0 * A0 / recon_A_safe
        threshold_arrays['PAori'] = pa_threshold_arr
        
        # Calculate Pass Percentages
        metrics_to_check = ['D', 'W', 'A', 'papeak', 'PAori', 'beta2_angle', 'true_D']
        if 'fc' in rel_thresholds: metrics_to_check.append('fc')
        for pm in pol_mags:
             if pm in rel_thresholds:
                 metrics_to_check.append(pm)
        
        for metric in metrics_to_check:
            # Check if metric exists in df
            recon_col = f'{metric}_mean' if is_bayesian else metric
            truth_col = f'{metric}_truth'
            std_col = f'{metric}_std'
            err_col = f'{metric}err' if metric + 'err' in df.columns else ( 'PAerr' if metric == 'PAori' else None)
            
            if recon_col not in df.columns or truth_col not in df.columns:
                continue
                
            recon_val = df[recon_col]
            truth_val = df[truth_col]
            
            # Determine threshold array
            if metric == 'PAori':
                thres = pa_threshold_arr
            elif metric in abs_thresholds:
                thres = np.full_like(truth_val, abs_thresholds[metric])
            elif metric in rel_thresholds:
                # 10% of truth value
                thres = 0.10 * np.abs(truth_val)
            else:
                # Fallback
                thres = np.zeros_like(truth_val)
                
            threshold_arrays[metric] = thres
            
            # Calculate diff
            if metric in ['PAori', 'papeak', 'beta2_angle']:
                # Angular difference
                diff = np.abs(recon_val - truth_val)
                diff = np.minimum(diff, 360 - diff)
            else:
                diff = np.abs(recon_val - truth_val)
                
            # Check condition
            # Is truth within (val +/- err) +/- threshold?
            # i.e. |val - truth| <= err + threshold
            uncertainty = 0
            if is_bayesian:
                uncertainty = df[std_col]
            else:
                # Handle true_D error specially
                if metric == 'true_D':
                    if 'true_Derr' in df.columns:
                        uncertainty = df['true_Derr']
                elif err_col and err_col in df.columns:
                    uncertainty = df[err_col]
                
            pass_condition = (diff - uncertainty) <= thres
                
            pass_pct = np.sum(pass_condition) / len(recon_val) * 100
            pass_percentages[metric] = pass_pct
            df[f'pass_percent_{metric}'] = pass_pct

    csv_path = args.outpath + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Plotting
    # Ring Quantities
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(32,8), sharex=True)
    
    ax[0,0].set_ylabel('Diameter $({\mu as})$')
    ax[0,1].set_ylabel('FWHM $({\mu as})$')
    ax[0,2].set_ylabel('Position angle ($^\circ$)')
    ax[1,0].set_ylabel('True Diameter $({\mu as})$')
    ax[1,1].set_ylabel('Asymmetry')
    ax[1,2].set_ylabel('Peak PA ($^\circ$)')
    
    for a in ax.flatten():
        a.set_xlabel('Time (UTC)')

    # Helper for plotting
    def plot_metric(ax, metric_name, label_prefix=''):
        # Plot Truth if available
        if truth_results and f'{metric_name}_truth' in df.columns:
            ax.plot(times, df[f'{metric_name}_truth'], '-', color='black', label='Truth', alpha=0.7)
            
            # Add shading for threshold
            if metric_name in threshold_arrays:
                thres = threshold_arrays[metric_name]
                truth_val = df[f'{metric_name}_truth']
                ax.fill_between(times, truth_val - thres, truth_val + thres, color='black', alpha=0.15)
        
        label = 'Recon'
        if metric_name in pass_percentages:
            label += f" (Pass: {pass_percentages[metric_name]:.1f}%)"
            
        if is_bayesian:
            ax.errorbar(times, df[f'{metric_name}_mean'], yerr=df[f'{metric_name}_std'], fmt='-o', label=label, capsize=3, alpha=0.6, color='tab:blue')
        else:
            # Check for error column
            err_col = f'{metric_name}err'
            if metric_name == 'PAori': err_col = 'PAerr'
            
            if err_col in df.columns:
                 ax.errorbar(times, df[metric_name], yerr=df[err_col], fmt='-o', label=label, capsize=3, alpha=0.6, color='tab:blue')
            else:
                ax.plot(times, df[metric_name], '-o', color='tab:blue', label=label)
            
    plot_metric(ax[0,0], 'D')
    plot_metric(ax[0,1], 'W')
    plot_metric(ax[0,2], 'PAori')
    plot_metric(ax[1,0], 'true_D')
    plot_metric(ax[1,1], 'A')
    plot_metric(ax[1,2], 'papeak')

    for a in ax.flatten():
        a.legend()
        
    plt.tight_layout()
    plt.savefig(args.outpath + '.png', bbox_inches='tight', dpi=300)
    print(f"Saved plot to {args.outpath}.png")
    
    # Pol Quantities (if available)
    if 'mnet_mean' in df.columns or 'mnet' in df.columns:
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(32,8), sharex=True)
        ax[0,0].set_ylabel('$|m|_{net}$')
        ax[0,1].set_ylabel(r'$\langle |m|\rangle$')
        ax[0,2].set_ylabel('$\chi\ (^\circ)$')
        ax[1,0].set_ylabel(r'$\beta_2$')
        ax[1,1].set_ylabel(r'$\angle \beta_2\ (^\circ)$')
        ax[1,2].set_ylabel('$|v|_{net}$')
        
        for a in ax.flatten():
            a.set_xlabel('Time (UTC)')
            
        plot_metric(ax[0,0], 'mnet')
        plot_metric(ax[0,1], 'mavg')
        plot_metric(ax[0,2], 'evpa')
        plot_metric(ax[1,0], 'beta2_abs')
        plot_metric(ax[1,1], 'beta2_angle')
        plot_metric(ax[1,2], 'vnet')
        
        # Add shading for Beta2 Angle threshold if truth available
        if truth_results and 'beta2_angle' in threshold_arrays:
            thres = threshold_arrays['beta2_angle']
            truth_val = df['beta2_angle_truth']
            ax[1,1].fill_between(times, truth_val - thres, truth_val + thres, color='black', alpha=0.15)
        
        for a in ax.flatten():
            a.legend()
            
        plt.tight_layout()
        plt.savefig(args.outpath + '_pol.png', bbox_inches='tight', dpi=300)
        print(f"Saved plot to {args.outpath}_pol.png")

if __name__ == "__main__":
    main()
