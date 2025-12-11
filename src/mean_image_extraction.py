import os

# Set environment variables to single-threaded before importing numpy/scipy/ehtim
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ehtim.const_def import *
from scipy import interpolate, stats
import astropy.units as u
from astropy.constants import k_B, c
import numpy as np
import pandas as pd
import ehtim as eh
from copy import copy, deepcopy
import argparse
import glob
from tqdm import tqdm
import itertools
import multiprocessing

######################################################################
# REx functions
######################################################################

def calculate_true_d_error(D, W, D_err, W_err):
    """
    Calculates the propagated error for the quantity true_D.
    """
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

def extract_ring_quantites(image, xc=None, yc=None, rcutoff=5):
    Npa = 360
    Nr = 100

    if xc is None or yc is None:
        xc, yc = fit_ring(image)
    
    # Gridding and interpolation
    x = np.arange(image.xdim) * image.psize / RADPERUAS
    y = np.flip(np.arange(image.ydim) * image.psize / RADPERUAS)
    z = image.imarr()
    f_image = interpolate.interp2d(x, y, z, kind="cubic")

    # Create a mesh grid in polar coordinates
    radial_imarr = np.zeros([Nr, Npa])

    pa = np.linspace(0, 360, Npa)
    pa_rad = np.deg2rad(pa)
    radial = np.linspace(0, 50, Nr)
    dr = radial[-1] - radial[-2]

    Rmesh, PAradmesh = np.meshgrid(radial, pa_rad)
    x_grid = Rmesh * np.sin(PAradmesh) + xc
    y_grid = Rmesh * np.cos(PAradmesh) + yc
    
    for r in range(Nr):
        z_vals = [f_image(x_grid[i][r], y_grid[i][r]) for i in range(len(pa))]
        radial_imarr[r, :] = np.array(z_vals)[:, 0]
    radial_imarr = np.fliplr(radial_imarr)

    # Calculate the r_pk at each PA and average
    peakpos = np.unravel_index(np.argmax(radial_imarr), shape=radial_imarr.shape)

    Rpeak = []
    Rmin = []
    Rmax = []
    ridx_r50 = np.argmin(np.abs(radial - 50))
    I_floor = radial_imarr[ridx_r50, :].mean()
    
    for ipa in range(len(pa)):
        tmpIr = copy(radial_imarr[:, ipa]) - I_floor
        tmpIr[np.where(radial < rcutoff)] = 0
        ridx_pk = np.argmax(tmpIr)
        rpeak = radial[ridx_pk]
        if ridx_pk > 0 and ridx_pk < Nr - 1:
            val_list = tmpIr[ridx_pk - 1:ridx_pk + 2]
            rpeak = quad_interp_radius(rpeak, dr, val_list)[0]
        Rpeak.append(rpeak)
        rmin, rmax = calc_width(tmpIr, radial, rpeak)
        Rmin.append(rmin)
        Rmax.append(rmax)
        
    paprofile = pd.DataFrame()
    paprofile["PA"] = pa
    paprofile["rpeak"] = Rpeak
    paprofile["rhalf_max"] = Rmax
    paprofile["rhalf_min"] = Rmin

    D = np.mean(paprofile["rpeak"]) * 2
    Derr = paprofile["rpeak"].std() * 2
    W = np.mean(paprofile["rhalf_max"] - paprofile["rhalf_min"])
    Werr = (paprofile["rhalf_max"] - paprofile["rhalf_min"]).std()

    # Calculate orientation angle, contrast, and asymmetry
    rin = D / 2. - W / 2.
    rout = D / 2. + W / 2.
    if rin <= 0.:
        rin = 0.

    exptheta = np.exp(1j * pa_rad)

    pa_ori_r = []
    amp_r = []
    ridx1 = np.argmin(np.abs(radial - rin))
    ridx2 = np.argmin(np.abs(radial - rout))
    for r in range(ridx1, ridx2 + 1, 1):
        amp = (radial_imarr[r, :] * exptheta).sum() / (radial_imarr[r, :]).sum()
        amp_r.append(amp)
        pa_ori = np.angle(amp, deg=True)
        pa_ori_r.append(pa_ori)
    
    pa_ori_r = np.array(pa_ori_r)
    amp_r = np.array(amp_r)
    PAori = stats.circmean(pa_ori_r, high=360, low=0)
    PAerr = stats.circstd(pa_ori_r, high=360, low=0)
    A = np.mean(np.abs(amp_r))
    Aerr = np.std(np.abs(amp_r))

    ridx_r5 = np.argmin(np.abs(radial - 5))
    ridx_pk = np.argmin(np.abs(radial - D / 2))
    fc = radial_imarr[0:ridx_r5, :].mean() / radial_imarr[ridx_pk, :].mean()

    # Source size from 2nd moment
    fwhm_maj, fwhm_min, theta = image.fit_gauss()
    fwhm_maj /= RADPERUAS
    fwhm_min /= RADPERUAS

    # Calculate flux ratio
    Nxc = int(xc / image.psize * RADPERUAS)
    Nyc = int(yc / image.psize * RADPERUAS)
    hole = extract_hole(image, Nxc, Nyc, r=rin)
    ring = extract_ring(image, Nxc, Nyc, rin=rin, rout=rout)
    outer = extract_outer(image, Nxc, Nyc, r=rout)
    hole_flux = hole.total_flux()
    outer_flux = outer.total_flux()
    ring_flux = ring.total_flux()

    Shole = np.pi * rin**2
    Souter = (2. * rout)**2. - np.pi * rout**2
    Sring = np.pi * rout**2 - np.pi * rin**2

    # Convert uas^2 to rad^2
    Shole = Shole * RADPERUAS**2
    Souter = Souter * RADPERUAS**2
    Sring = Sring * RADPERUAS**2

    # Unit K brightness temperature
    freq = image.rf * u.Hz
    hole_dflux = hole_flux / Shole * (c**2 / 2 / k_B / freq**2).to(u.K / u.Jansky).value
    outer_dflux = outer_flux / Souter * (c**2 / 2 / k_B / freq**2).to(u.K / u.Jansky).value
    ring_dflux = ring_flux / Sring * (c**2 / 2 / k_B / freq**2).to(u.K / u.Jansky).value
    
    true_D = np.array(D / (1 - (1 / (4 * np.log(2))) * (W / D)**2))
    true_Derr = calculate_true_d_error(D, W, Derr, Werr)

    # Output dictionary
    outputs = dict(
        rpeak=radial[peakpos[0]],
        papeak=pa[peakpos[1]],
        xc=xc,
        yc=yc,
        PAori=PAori,
        PAerr=PAerr,
        A=A,
        Aerr=Aerr,
        fc=fc,
        D=D,
        Derr=Derr,
        W=W,
        Werr=Werr,
        true_D=true_D,
        true_Derr=true_Derr,
        fwhm_maj=fwhm_maj,
        fwhm_min=fwhm_min,
        hole_flux=hole_flux,
        outer_flux=outer_flux,
        ring_flux=ring_flux,
        totalflux=image.total_flux(),
        hole_dflux=hole_dflux,
        outer_dflux=outer_dflux,
        ring_dflux=ring_dflux
    )
    return outputs

def extract_hole(image, Nxc, Nyc, r=30):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim) - Nxc + 1) * outimage.psize / RADPERUAS
    y = (np.arange(outimage.ydim) - Nyc + 1) * outimage.psize / RADPERUAS
    x, y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2 >= 0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim * outimage.xdim)
    return outimage

def extract_outer(image, Nxc, Nyc, r=30):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim) - Nxc + 1) * outimage.psize / RADPERUAS
    y = (np.arange(outimage.ydim) - Nyc + 1) * outimage.psize / RADPERUAS
    x, y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2 <= 0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim * outimage.xdim)
    return outimage

def extract_ring(image, Nxc, Nyc, rin=30, rout=50):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim) - Nxc + 1) * outimage.psize / RADPERUAS
    y = (np.arange(outimage.ydim) - Nyc + 1) * outimage.psize / RADPERUAS
    x, y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - rin**2 <= 0)] = 0
    masked[np.where(x**2 + y**2 - rout**2 >= 0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim * outimage.xdim)
    return outimage

def quad_interp_radius(r_max, dr, val_list):
    v_L = val_list[0]
    v_max = val_list[1]
    v_R = val_list[2]
    rpk = r_max + dr * (v_L - v_R) / (2 * (v_L + v_R - 2 * v_max))
    vpk = 8 * v_max * (v_L + v_R) - (v_L - v_R)**2 - 16 * v_max**2
    vpk /= (8 * (v_L + v_R - 2 * v_max))
    return (rpk, vpk)

def calc_width(tmpIr, radial, rpeak):
    spline = interpolate.UnivariateSpline(radial, tmpIr - 0.5 * tmpIr.max(), s=0)
    roots = spline.roots()

    if len(roots) == 0:
        return (radial[0], radial[-1])

    rmin = radial[0]
    rmax = radial[-1]
    for root in np.sort(roots):
        if root < rpeak:
            rmin = root
        else:
            rmax = root
            break

    return (rmin, rmax)

def fit_ring(image, Nr=50, Npa=25, rmin_search=10, rmax_search=100, fov_search=0.1, Nserch=20):
    xc, yc = eh.features.rex.findCenter(image, rmin_search=rmin_search, rmax_search=rmax_search,
                                        nrays_search=Npa, nrs_search=Nr,
                                        fov_search=fov_search, n_search=Nserch)
    return xc, yc

######################################################################
# Polarization functions
######################################################################

def make_polar_imarr(imarr, dx, xc=None, yc=None, rmax=50, Nr=50, Npa=180, kind="linear", image=None):
    nx, ny = imarr.shape
    x = np.arange(nx) * dx / RADPERUAS
    y = np.arange(ny) * dx / RADPERUAS
    
    if xc is None or yc is None:
        xc, yc = fit_ring(image)

    z = imarr
    f_image = interpolate.interp2d(x, y, z, kind=kind)

    radial_imarr = np.zeros([Nr, Npa])
    pa = np.linspace(0, 360, Npa)
    pa_rad = np.deg2rad(pa)
    radius = np.linspace(0, rmax, Nr)

    Rmesh, PAradmesh = np.meshgrid(radius, pa_rad)
    x_grid, y_grid = Rmesh * np.sin(PAradmesh) + xc, Rmesh * np.cos(PAradmesh) + yc
    
    for ir in range(Nr):
        z_vals = [f_image(x_grid[ipa][ir], y_grid[ipa][ir]) for ipa in range(Npa)]
        radial_imarr[ir, :] = z_vals[:]
    radial_imarr = np.fliplr(radial_imarr)

    return radial_imarr, radius, pa

def extract_pol_quantites(im, xc=None, yc=None, blur_size=-1):
    Itot, Qtot, Utot = sum(im.imvec), sum(im.qvec), sum(im.uvec)
    if len(im.vvec) == 0:
        im.vvec = np.zeros_like(im.imvec)
    Vtot = sum(im.vvec)
    
    mnet = np.sqrt(Qtot * Qtot + Utot * Utot) / Itot

    if blur_size < 0:
        mavg = sum(np.sqrt(im.qvec**2 + im.uvec**2)) / Itot
    else:
        im_blur = im.blur_circ(blur_size * eh.RADPERUAS, fwhm_pol=blur_size * eh.RADPERUAS)
        mavg = sum(np.sqrt(im_blur.qvec**2 + im_blur.uvec**2)) / np.sum(im_blur.imvec)

    evpa = (180. / np.pi) * 0.5 * np.angle(Qtot + 1j * Utot)
    vnet = np.abs(Vtot) / Itot

    P = im.qvec + 1j * im.uvec
    P_radial, radius, pa = make_polar_imarr(P.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    I_radial, _, _ = make_polar_imarr(im.imvec.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    V_radial, _, _ = make_polar_imarr(im.vvec.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    
    Pring, Vring, Vring2, Iring = 0, 0, 0, 0
    for ir, ipa in itertools.product(range(len(radius)), range(len(pa))):
        factor = radius[ir]
        Pring += P_radial[ir, ipa] * np.exp(-2 * 1j * np.deg2rad(pa[ipa])) * factor
        Vring2 += V_radial[ir, ipa] * np.exp(-2 * 1j * np.deg2rad(pa[ipa])) * factor
        Vring += V_radial[ir, ipa] * np.exp(-1 * 1j * np.deg2rad(pa[ipa])) * factor
        Iring += I_radial[ir, ipa] * factor
        
    beta2 = Pring / Iring
    beta2_abs, beta2_angle = np.abs(beta2), np.rad2deg(np.angle(beta2))

    beta2_v = Vring2 / Iring
    beta2_v_abs, beta2_v_angle = np.abs(beta2_v), np.rad2deg(np.angle(beta2_v))
    beta_v = Vring / Iring
    beta_v_abs, beta_v_angle = np.abs(beta_v), np.rad2deg(np.angle(beta_v))

    outputs = dict(
        mnet=mnet,
        mavg=mavg,
        evpa=evpa,
        beta2_abs=beta2_abs,
        beta2_angle=beta2_angle,
        vnet=vnet,
        beta_v_abs=beta_v_abs,
        beta_v_angle=beta_v_angle,
        beta2_v_abs=beta2_v_abs,
        beta2_v_angle=beta2_v_angle
    )
    return outputs

######################################################################
# Main execution
######################################################################

def process_single_file(f):
    try:
        im = eh.image.load_fits(f)
        
        # Extract ring quantities
        ring_outputs = extract_ring_quantites(im)
        
        # Extract polarization quantities if present
        has_pol = len(im.qvec) > 0 and len(im.uvec) > 0
        if has_pol:
            xc = ring_outputs['xc']
            yc = ring_outputs['yc']
            pol_outputs = extract_pol_quantites(im, xc=xc, yc=yc)
            ring_outputs.update(pol_outputs)
        
        return ring_outputs
        
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract mean image quantities from FITS files.")
    parser.add_argument('--fits', type=str, required=True, help='Path to FITS file or directory of FITS files')
    parser.add_argument('-o', '--outpath', type=str, default='./output.csv', help='Output CSV file path')
    parser.add_argument('--ncores', type=int, default=32, help='Number of cores for parallel processing')
    args = parser.parse_args()

    fits_path = args.fits
    outpath = args.outpath
    ncores = args.ncores

    if os.path.isdir(fits_path):
        files = sorted(glob.glob(os.path.join(fits_path, "*.fits")))
        print(f"Found {len(files)} FITS files in {fits_path}")
    else:
        files = [fits_path]
        print(f"Processing single FITS file: {fits_path}")

    all_results = []

    if len(files) > 1:
        print(f"Processing with {ncores} cores...")
        with multiprocessing.Pool(ncores) as pool:
            # Use tqdm to show progress
            results = list(tqdm(pool.imap(process_single_file, files), total=len(files)))
            
        # Filter out None results (failed processings)
        all_results = [r for r in results if r is not None]
    else:
        # Single file, no need for pool overhead
        res = process_single_file(files[0])
        if res:
            all_results.append(res)

    if not all_results:
        print("No results extracted.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    # All samples are saved as rows.
    df.to_csv(outpath, index=False)
    print(f"Saved results to {outpath}")

if __name__ == "__main__":
    main()