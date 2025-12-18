######################################################################
# Author: Rohan Dahale, Date: 08 December 2025
######################################################################

import os

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import ehtim as eh
import numpy as np
import functools
from concurrent.futures import ProcessPoolExecutor

import argparse
import glob
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, required=True,
                   help='path to uvfits data file')
    p.add_argument('--input', type=str, nargs='+', required=True,
                   help='input HDF5 files (can use shell glob expansion)')
    p.add_argument('-o', '--outname', type=str, required=True,
                   help='base output name (without extension)')
    p.add_argument('--tstart', type=float, default=None, help='Start time (in UT hours) for data')
    p.add_argument('--tstop', type=float, default=None, help='Stop time (in UT hours) for data')
    p.add_argument('-n', '--ncores', type=int, default=16,
                   help='number of cores for parallel processing (default: 16)')
    return p

# List of parsed arguments
args = create_parser().parse_args()

######################################################################
# Set parameters
npix   = 200
fov    = 200 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

# Parse output path
outname_dir = os.path.dirname(args.outname)

# If no directory specified, use current directory
if not outname_dir:
    outname_dir = '.'

# Create output directories relative to the outname directory
regrid_dir = os.path.join(outname_dir, 'regrid_hdf5')
static_dir = os.path.join(outname_dir, 'static_part')
dynamic_dir = os.path.join(outname_dir, 'dynamic_part')
uniform_dir = os.path.join(outname_dir, 'uniform_hdf5')

os.makedirs(regrid_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs(dynamic_dir, exist_ok=True)
os.makedirs(uniform_dir, exist_ok=True)

# Get input files (already expanded by shell)
input_files = sorted(args.input)
if not input_files:
    print(f"No input files provided")
    exit(1)

print(f"Found {len(input_files)} input files")

######################################################################
# Load observation and determine valid time range
######################################################################
print(f"Loading observation: {args.data}")
with open(os.devnull, 'w') as devnull:
    with redirect_stdout(devnull), redirect_stderr(devnull):
        obs = eh.obsdata.load_uvfits(args.data)
        obs = obs.add_fractional_noise(0.01)
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
        exit(1)

    obs_times = np.array([o.data['time'][0] for o in obslist])
    print(f"New data time range: {obs_times.min():.3f} - {obs_times.max():.3f} h")

times = obs_times
min_t = obs_times.min()
max_t = obs_times.max()

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
    exit(1)

movie_min_t = max(min_t_list)
movie_max_t = min(max_t_list)
print(f"Movie time range: {movie_min_t:.3f} - {movie_max_t:.3f} h")

if movie_min_t > times.min() or movie_max_t < times.max():
     print("Warning: Movie times do not span the whole duration of data. Extrapolation will be used.")

print(f"Number of valid time frames: {len(times)}")

######################################################################
# Function to process a single file
######################################################################
def process_single_file(m_path, times, min_t, max_t, npix, fov, blur, 
                       regrid_dir, static_dir, dynamic_dir, uniform_dir):
    """Process a single HDF5 file with regrid, static, dynamic, and uniform operations"""
    
    basename = os.path.basename(m_path)
    basename_noext = os.path.splitext(basename)[0]
    
    # Define output paths
    regrid_out = os.path.join(regrid_dir, f'{basename}')
    static_out = os.path.join(static_dir, f'{basename_noext}.fits')
    dynamic_out = os.path.join(dynamic_dir, f'{basename}')
    uniform_out = os.path.join(uniform_dir, f'{basename}')
    
    # Load movie
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            mov = eh.movie.load_hdf5(m_path)
            mov.reset_interp(bounds_error=False)
    
    ######################################################################
    # 1. Regrid to 200x200 pixels and 200x200uas FOV
    ######################################################################
    if not os.path.exists(regrid_out):
        imlist = []
        for t in times:
            im = mov.get_image(t)
            im = im.blur_circ(fwhm_i=blur, fwhm_pol=blur).regrid_image(fov, npix)
            imlist.append(im)
        
        new_movie = eh.movie.merge_im_list(imlist)
        new_movie.reset_interp(bounds_error=False)
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                new_movie.save_hdf5(regrid_out)
    
    ######################################################################
    # 2. Extract and save static part (median)
    ######################################################################
    if not os.path.exists(static_out):
        imlist = []
        imlistIarr = []
        imlistQarr = []
        imlistUarr = []
        imlistVarr = []
        
        for t in times:
            im = mov.get_image(t)
            im = im.blur_circ(fwhm_i=blur, fwhm_pol=blur).regrid_image(fov, npix)
            imlist.append(im)
            imlistIarr.append(im.imarr(pol='I'))
            imlistQarr.append(im.imarr(pol='Q'))
            imlistUarr.append(im.imarr(pol='U'))
            imlistVarr.append(im.imarr(pol='V'))
        
        medianI = np.median(imlistIarr, axis=0)
        medianQ = np.median(imlistQarr, axis=0)
        medianU = np.median(imlistUarr, axis=0)
        medianV = np.median(imlistVarr, axis=0)
        
        if len(imlist[0].ivec) != 0:
            imlist[0].ivec = medianI.flatten()
        if len(imlist[0].qvec) != 0:
            imlist[0].qvec = medianQ.flatten()
        if len(imlist[0].uvec) != 0:
            imlist[0].uvec = medianU.flatten()
        if len(imlist[0].vvec) != 0:
            imlist[0].vvec = medianV.flatten()
        
        imlist[0].save_fits(static_out)
    
    ######################################################################
    # 3. Extract and save dynamic part (subtract median)
    ######################################################################
    if not os.path.exists(dynamic_out):
        imlist = []
        imlistIarr = []
        imlistQarr = []
        imlistUarr = []
        imlistVarr = []
        
        for t in times:
            im = mov.get_image(t)
            im = im.blur_circ(fwhm_i=blur, fwhm_pol=blur).regrid_image(fov, npix)
            imlist.append(im)
            imlistIarr.append(im.imarr(pol='I'))
            imlistQarr.append(im.imarr(pol='Q'))
            imlistUarr.append(im.imarr(pol='U'))
            imlistVarr.append(im.imarr(pol='V'))
        
        medianI = np.median(imlistIarr, axis=0)
        medianQ = np.median(imlistQarr, axis=0)
        medianU = np.median(imlistUarr, axis=0)
        medianV = np.median(imlistVarr, axis=0)
        
        for im in imlist:
            if len(im.ivec) != 0:
                imat = im.imarr(pol='I') - medianI
                im.ivec = imat.flatten()
            if len(im.qvec) != 0:
                qmat = im.imarr(pol='Q') - medianQ
                im.qvec = qmat.flatten()
            if len(im.uvec) != 0:
                umat = im.imarr(pol='U') - medianU
                im.uvec = umat.flatten()
            if len(im.vvec) != 0:
                vmat = im.imarr(pol='V') - medianV
                im.vvec = vmat.flatten()
        
        dyn_movie = eh.movie.merge_im_list(imlist)
        dyn_movie.reset_interp(bounds_error=False)
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                dyn_movie.save_hdf5(dynamic_out)
    
    ######################################################################
    # 4. Create time-uniform HDF5 (frames every two minutes)
    ######################################################################
    if not os.path.exists(uniform_out):
        # Time stamps every two minutes
        step_hr = 2*1.0/60.0
        uniform_times = np.arange(min_t, max_t + 1e-5, step_hr)
        ntimes = len(uniform_times)
        
        frame_list = []
        for t in uniform_times:
            frame = mov.get_image(t).regrid_image(fov, npix)
            frame_list.append(frame)
        
        uniform_movie = eh.movie.merge_im_list(frame_list)
        uniform_movie.reset_interp(bounds_error=False)
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                uniform_movie.save_hdf5(uniform_out)
    
    return basename

######################################################################
# Process files in parallel
######################################################################
print(f"\nProcessing {len(input_files)} files using {args.ncores} cores...")

# Use functools.partial to bind constant arguments
process_func = functools.partial(process_single_file, 
                               times=times, min_t=min_t, max_t=max_t, 
                               npix=npix, fov=fov, blur=blur,
                               regrid_dir=regrid_dir, static_dir=static_dir, 
                               dynamic_dir=dynamic_dir, uniform_dir=uniform_dir)

# Process in parallel using ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=args.ncores) as executor:
    results = list(tqdm(
        executor.map(process_func, input_files),
        total=len(input_files),
        desc="Processing files"
    ))

print("\n" + "="*60)
print("Processing complete!")
print("="*60)

