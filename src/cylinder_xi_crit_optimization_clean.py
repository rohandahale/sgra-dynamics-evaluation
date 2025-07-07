"""
Pattern Speed across Xi_crit values

Original Author: Nick Conroy, Date: 25 April 2025. 
Based on cylinder_mcmc.py, author Nick Conroy, edited Rohan Dahale, date: 08 April 2025.

Changes from cylinder_mcmc.py:
-added new save file, pattern_speed_vs_xicrit_savefile, which contains Omega_p for each xi_crit in array
-changed run_mcmc function to run_xi_crit_survey. Delete survey of (x, y, r), now survey xi_crit values. 
-Set xi_crit_stdunits = True and added to relevant functions throughout
-deleted some excess: e.g. determine_xi_crit_factor function , plot_histogram function, etc.


"""

"""
Pattern Speed across Xi_crit values with Integrated Ring Fitting

Original Author: Nick Conroy, Date: 25 April 2025.
Based on cylinder_mcmc.py, author Nick Conroy, edited Rohan Dahale, date: 08 April 2025.
Merged with RingFitter from cylinder_mcmc.py (Script 1) on 01 May 2025.
Main execution block updated to run RingFitter first on 01 May 2025.

This script first fits ring parameters (center, radius, uncertainty) from an
input hdf5 video file using the RingFitter class and saves them. Then, it
calculates the pattern speed for a range of xi_crit values using the
CylinderAnalysis class and the freshly fitted ring parameters.

Requires input HDF5 file. Generates ring parameters CSV file.
"""

import argparse
import os
import sys
from typing import Tuple, Optional, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import pandas as pd
import scipy.ndimage as ndimage
from scipy.ndimage import label
from scipy.stats import truncnorm
import ehtim as eh # RingFitter dependency
from scipy import interpolate, stats # RingFitter dependency
from copy import deepcopy # RingFitter dependency

# Attempt to import utilities, handle if not found
try:
    from utilities import *
    colors, titles, labels, mfcs, mss = common()
except ImportError:
    print("Warning: 'utilities' module not found. Using default plotting parameters.")
    # Define dummy values or handle plotting differently if utilities are essential
    colors, titles, labels, mfcs, mss = (None, None, None, None, None) # Example dummy

# Constants
RADPERUAS = 4.848136811094136e-12
GM_C3 = 20.46049 / 3600  # hours
M_PER_RAD = 41139576306.70914

# %%############################################################################
# Ring Fitting
# ##############################################################################

class RingFitter:
    """Class for fitting and extracting ring parameters from EHT images."""

    def __init__(self, fov=200*eh.RADPERUAS, npix=200):
        """Initialize the RingFitter."""
        self.fov = fov
        self.npix = npix

    def fit_from_file(self, input_file, output_file="ring.csv", **kwargs):
        """Process a movie file to extract ring parameters and save to CSV."""
        fov = kwargs.pop('fov', self.fov); npix = kwargs.pop('npix', self.npix)
        try:
            mv = eh.movie.load_hdf5(input_file)
            im = mv.avg_frame().regrid_image(fov, npix)
        except Exception as e:
            print(f"Error loading or processing HDF5 file '{input_file}': {e}")
            raise # Re-raise the exception to be caught in __main__
        return self.fit_from_image(im, output_file, **kwargs)

    def fit_from_image(self, image, output_file=None, **kwargs):
        """Extract ring parameters from an image and optionally save to CSV."""
        center_x=kwargs.pop('center_x',None); center_y=kwargs.pop('center_y',None)
        search_radius_min=kwargs.pop('search_radius_min',10); search_radius_max=kwargs.pop('search_radius_max',100)
        nrays_search=kwargs.pop('nrays_search',25); nrs_search=kwargs.pop('nrs_search',50)
        fov_search=kwargs.pop('fov_search',0.1); n_search=kwargs.pop('n_search',20)
        image_blur_fwhm=kwargs.pop('image_blur_fwhm',2.0*eh.RADPERUAS); image_threshold=kwargs.pop('image_threshold',0.05)

        if center_x is None or center_y is None:
            try:
                center_x, center_y = self._find_center(
                    image, search_radius_min=search_radius_min, search_radius_max=search_radius_max,
                    nrays_search=nrays_search, nrs_search=nrs_search, fov_search=fov_search, n_search=n_search,
                    blur_fwhm=image_blur_fwhm, threshold=image_threshold)
            except Exception as e:
                print(f"Error finding ring center using REX: {e}")
                print("Please check ehtim installation and image properties.")
                raise

        try:
            params = self._extract_ring_parameters(image, center_x, center_y, **kwargs)
        except Exception as e:
             print(f"Error extracting ring parameters: {e}")
             raise

        fov_uas = image.fovx()/eh.RADPERUAS; npix = image.xdim
        # Convert pixel center (params['xc'], params['yc']) relative to bottom-left
        # to uas offset (x0, y0) relative to image center.
        x0 = (params["xc"] - (npix/2 - 0.5)) * image.psize / eh.RADPERUAS
        y0 = (params["yc"] - (npix/2 - 0.5)) * image.psize / eh.RADPERUAS

        df = pd.DataFrame({"x0":[x0], "y0":[y0], "r":[params["D"]/2], "r_err":[params["Derr"]/2]})

        if output_file:
            try:
                df.to_csv(output_file, index=False)
                print(f"Ring parameters saved to: {output_file}")
            except Exception as e:
                print(f"Error saving ring parameters to {output_file}: {e}")
                # Continue analysis even if saving fails? Or raise error? Let's raise.
                raise
        return df

    def _find_center(self, image, search_radius_min=10, search_radius_max=100,
                    nrays_search=25, nrs_search=50, fov_search=0.1, n_search=20,
                    blur_fwhm=2.0*eh.RADPERUAS, threshold=0.05):
        """Find the center of a ring in an image using REX. (uas inputs)"""
        image_blur = image.blur_circ(blur_fwhm)
        image_mod = image_blur.threshold(cutoff=threshold * image_blur.imarr().max())
        xc, yc = eh.features.rex.findCenter(
            image_mod,
            rmin_search=search_radius_min*eh.RADPERUAS/image.psize, # uas to pixels
            rmax_search=search_radius_max*eh.RADPERUAS/image.psize, # uas to pixels
            nrays_search=nrays_search, nrs_search=nrs_search,
            fov_search=fov_search, n_search=n_search)
        return xc, yc # Returns pixel coordinates

    def _extract_ring_parameters(self, image, center_x_pix, center_y_pix,
                                min_radius=5, max_radius=50,
                                n_angles=360, n_radial=100,
                                return_full_data=False):
        """Extract detailed ring parameters from an image using polar sampling. (uas inputs)"""
        psize_uas = image.psize/eh.RADPERUAS
        angles=np.linspace(0,360,n_angles,endpoint=False); angles_rad=np.deg2rad(angles)
        # Use min_radius/max_radius passed from kwargs for the radial points
        radial_points_uas = np.linspace(min_radius, max_radius, n_radial)
        dr_uas = radial_points_uas[1] - radial_points_uas[0]

        x_pix=np.arange(image.xdim); y_pix=np.arange(image.ydim); z=image.imarr()
        image_interp = interpolate.RectBivariateSpline(y_pix, x_pix, z, kx=1, ky=1)

        radial_image=np.zeros([n_radial,n_angles])
        r_mesh_uas,angle_mesh_rad=np.meshgrid(radial_points_uas,angles_rad)
        x_pix_mesh=center_x_pix+(r_mesh_uas/psize_uas)*np.cos(angle_mesh_rad)
        y_pix_mesh=center_y_pix+(r_mesh_uas/psize_uas)*np.sin(angle_mesh_rad)
        x_pix_mesh=np.clip(x_pix_mesh,0,image.xdim-1); y_pix_mesh=np.clip(y_pix_mesh,0,image.ydim-1)

        radial_image_transposed=image_interp(y_pix_mesh,x_pix_mesh,grid=False)
        radial_image=radial_image_transposed.T

        r_peak_values_uas=[]; r_min_values_uas=[]; r_max_values_uas=[]
        for angle_idx in range(n_angles):
            intensity_profile=radial_image[:,angle_idx]
            intensity_floor=np.min(intensity_profile); intensity_profile_norm=intensity_profile-intensity_floor
            peak_idx=np.argmax(intensity_profile_norm); r_peak_uas=radial_points_uas[peak_idx]
            if 0<peak_idx<n_radial-1:
                nearby_values=intensity_profile[peak_idx-1:peak_idx+2]
                r_peak_uas,_ = self._quad_interp_radius(r_peak_uas, dr_uas, nearby_values)
            r_min_uas,r_max_uas = self._calc_width(intensity_profile_norm,radial_points_uas,r_peak_uas)
            r_peak_values_uas.append(r_peak_uas); r_min_values_uas.append(r_min_uas); r_max_values_uas.append(r_max_uas)

        r_peak_values_uas=np.array(r_peak_values_uas)
        ring_diameter_uas=np.mean(r_peak_values_uas)*2; diameter_error_uas=np.std(r_peak_values_uas)*2
        result={"xc":center_x_pix,"yc":center_y_pix,"D":ring_diameter_uas,"Derr":diameter_error_uas}
        # Add full data if requested (omitted here for brevity, same as before)
        return result

    def _quad_interp_radius(self, r_max, dr, val_list):
        """Quadratic interpolation to find peak radius. (uas inputs)"""
        v_L,v_max,v_R = val_list; denominator = 2*(v_L+v_R-2*v_max)
        if np.isclose(denominator,0): rpk=r_max; vpk=v_max
        else:
            offset = dr*(v_L-v_R)/denominator; rpk=r_max+offset
            rpk = np.clip(rpk,r_max-dr,r_max+dr); vpk = v_max-0.25*(v_L-v_R)*(offset/dr)
        return (rpk, vpk)

    def _calc_width(self, intensity_norm, radial_points_uas, peak_radius_uas):
        """Calculate ring width from half max points. (uas inputs)"""
        peak_intensity = np.max(intensity_norm)
        if np.isclose(peak_intensity, 0): return radial_points_uas[0],radial_points_uas[-1]
        half_max_value = 0.5*peak_intensity
        try:
            spline = interpolate.UnivariateSpline(radial_points_uas, intensity_norm - half_max_value, s=0, k=3)
            roots = spline.roots()
        except Exception as e: roots = []
        if len(roots) == 0: return radial_points_uas[0],radial_points_uas[-1]
        roots_below = roots[roots < peak_radius_uas]; roots_above = roots[roots >= peak_radius_uas]
        r_min = np.max(roots_below) if len(roots_below)>0 else radial_points_uas[0]
        r_max = np.min(roots_above) if len(roots_above)>0 else radial_points_uas[-1]
        return r_min, r_max

def fit_ring_from_file(input_file, output_file="ring.csv", **kwargs):
    """Convenience function for RingFitter. (uas inputs/kwargs)"""
    # Extract specific kwargs or use defaults
    fov = kwargs.pop('fov', 200 * eh.RADPERUAS)
    npix = kwargs.pop('npix', 200)
    min_radius_fit = kwargs.pop('min_radius', 10.0) # uas for extraction
    max_radius_fit = kwargs.pop('max_radius', 100.0) # uas for extraction
    # Use extraction radii for REX search by default, unless overridden
    search_radius_min_rex = kwargs.pop('search_radius_min', min_radius_fit) # uas for REX
    search_radius_max_rex = kwargs.pop('search_radius_max', max_radius_fit) # uas for REX

    fitter = RingFitter(fov=fov, npix=npix)
    # Pass radii correctly to fit_from_file -> fit_from_image
    return fitter.fit_from_file(
        input_file, output_file,
        min_radius=min_radius_fit,         # Passed to _extract_ring_parameters
        max_radius=max_radius_fit,         # Passed to _extract_ring_parameters
        search_radius_min=search_radius_min_rex, # Passed to _find_center
        search_radius_max=search_radius_max_rex, # Passed to _find_center
        **kwargs # Pass any other kwargs like blur, threshold etc.
    )

# %%############################################################################
# Cylinder Analysis Functionality
# ##############################################################################

class CylinderAnalysis:
    """Class for performing pattern speed analysis on cylinder plots."""
    # __init__ and other methods remain the same as in the previous correct version
    # ... (methods: __init__, add_colorbar, bilinear_interp, mean_subtract,
    # extract_ring_profile, plot_mean_image_with_ring, sample_cylinder,
    # compute_autocorrelation, find_pattern_speed, plot_autocorrelation,
    # calculate_pattern_speed, run_xi_crit_survey, plot_ps_vs_xicrit) ...

    def __init__(self, file_path: str, ring_file_path: str, output_dir: str, xi_crit_stdunits: bool = True):
        """Initialize the CylinderAnalysis class with input and output paths."""
        self.file_path = file_path; self.ring_file_path = ring_file_path
        self.xi_crit_stdunits = xi_crit_stdunits; self.output_dir = output_dir

        # Load data (HDF5 and Ring CSV)
        self.hfp = h5py.File(file_path, 'r')
        try:
            self.df = pd.read_csv(ring_file_path)
            if self.df.empty: raise ValueError("Ring parameter file is empty.")
            required=['x0','y0','r','r_err']; missing=[c for c in required if c not in self.df.columns]
            if missing: raise ValueError(f"Ring file missing columns: {missing}")
        except FileNotFoundError: print(f"Error: Ring file not found: {ring_file_path}"); sys.exit(1)
        except Exception as e: print(f"Error loading ring file {ring_file_path}: {e}"); sys.exit(1)

        # Dimensions, scales, FOV, time etc.
        self.N,self.M=self.hfp['I'].shape[1:]; self.dx=self.dy=float(self.hfp['header'].attrs['psize'])/RADPERUAS
        self.times=np.array(self.hfp['times']); self.nftot=len(self.times)
        if self.nftot>1: self.dt=round(np.diff(self.times).mean()/GM_C3,1)
        else: self.dt=0
        self.FOV_uas=self.dx*self.N; self.FOV_rad=self.FOV_uas*RADPERUAS; self.FOV_M=self.FOV_rad*M_PER_RAD
        self.Tmax = (self.times[-1]-self.times[0])/GM_C3 if self.nftot>1 else 0

        # Image data, smoothing parameters
        self.Iall=np.transpose(self.hfp['I'],[1,2,0])
        self.ntheta=180; self.theta=np.linspace(0.,2.*np.pi,self.ntheta,endpoint=False)
        self.dtheta=360/self.ntheta; self.pa=self.theta
        self.mean_im=np.mean(self.Iall,axis=2)
        fwhm_uas=20.; fwhm_pix=fwhm_uas/self.dx; self.sig=fwhm_pix/(2.*np.sqrt(2.*np.log(2.)))
        self.sIall=ndimage.gaussian_filter(self.Iall,sigma=(self.sig,self.sig,0))
        self.mean_sim=np.mean(self.sIall,axis=2)

        # Output save file name
        unit_suffix = 'stdunits' if self.xi_crit_stdunits else 'xiunits'
        self.pattern_speed_vs_xicrit_savefile = os.path.join(self.output_dir, f'pattern_speed_vs_xicrit_{unit_suffix}.npy')

    @staticmethod
    def add_colorbar(mappable):
        """Adds a colorbar to a matplotlib mappable object."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable; import matplotlib.pyplot as plt
        ax=mappable.axes; fig=ax.figure; divider=make_axes_locatable(ax)
        cax=divider.append_axes("right",size="5%",pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    @staticmethod
    def bilinear_interp(dat, x, y):
        """Performs bilinear interpolation at pixel coordinates (x, y)."""
        ny,nx=dat.shape; x=np.clip(x,0,nx-1.001); y=np.clip(y,0,ny-1.001)
        i=int(y); j=int(x); di=y-i; dj=x-j
        z00=dat[i,j]; z01=dat[i,min(j+1,nx-1)]; z10=dat[min(i+1,ny-1),j]; z11=dat[min(i+1,ny-1),min(j+1,nx-1)]
        return z00*(1.-di)*(1.-dj) + z01*(1.-di)*dj + z10*di*(1.-dj) + z11*di*dj

    @staticmethod
    def mean_subtract(m):
        """Subtracts row and column means from a 2D array."""
        nt,nth=m.shape;
        for i in range(nth): m[:,i]-=np.mean(m[:,i])
        for i in range(nt): m[i,:]-=np.mean(m[i,:])

    def extract_ring_profile(self, x_shift_uas: float, y_shift_uas: float, r_shift_uas: float) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Extracts ring profile coordinates (uas) and pixel indices applying shifts."""
        r0=self.df['r'].iloc[0]; x0=self.df['x0'].iloc[0]; y0=self.df['y0'].iloc[0]
        x0s=x0+x_shift_uas; y0s=y0+y_shift_uas; rs=max(r0+r_shift_uas,0)
        x_ring_uas=x0s+rs*np.cos(self.pa); y_ring_uas=y0s+rs*np.sin(self.pa)
        cxp=self.N/2.-0.5; cyp=self.M/2.-0.5
        x_pix=cxp+x_ring_uas/self.dx; y_pix=cyp+y_ring_uas/self.dy
        return x_ring_uas, y_ring_uas, (y_pix, x_pix)

    def plot_mean_image_with_ring(self, index: int = 0):
        """Plots the mean smoothed image with the ring and uncertainty band."""
        if self.df is None or self.df.empty:
            print("Error: Ring parameters dataframe (self.df) is not loaded or empty.")
            return
        try:
            # Use .iloc[0] safely
            r_m = self.df['r'].iloc[0]
            x0_m = self.df['x0'].iloc[0]
            y0_m = self.df['y0'].iloc[0]
            r_e = self.df['r_err'].iloc[0]
        except (KeyError, IndexError) as e:
            print(f"Error accessing ring parameters from dataframe: {e}")
            return # Stop if parameters are missing/invalid

        r_i = max(0, r_m - r_e) # Inner radius
        r_o = r_m + r_e       # Outer radius

        # --- CORRECTED COORDINATE CALCULATIONS ---
        x_center = x0_m + r_m * np.cos(self.pa)
        y_center = y0_m + r_m * np.sin(self.pa)
        x_outer  = x0_m + r_o * np.cos(self.pa)
        y_outer  = y0_m + r_o * np.sin(self.pa)
        x_inner  = x0_m + r_i * np.cos(self.pa)
        y_inner  = y0_m + r_i * np.sin(self.pa)
        # --- END CORRECTION ---

        fig, ax = plt.subplots(figsize=(5, 5))
        ext = [-self.FOV_uas / 2, self.FOV_uas / 2, -self.FOV_uas / 2, self.FOV_uas / 2]
        im = ax.imshow(self.mean_im, cmap='afmhot', aspect='equal', interpolation='bilinear', origin='lower', extent=ext)
        ax.set_title('Mean Image with Fitted Ring')
        ax.set_xlim(ext[0:2]); ax.set_ylim(ext[2:4])
        ax.set_xlabel(r'$x\, [\mu{\rm as}]$'); ax.set_ylabel(r'$y\, [\mu{\rm as}]$')

        # --- Use corrected variable names ---
        ax.plot(x_center, y_center, color='skyblue', ls='--', lw=1, label=f'Mean R ({r_m:.2f})')
        fill_x = np.concatenate([x_outer, x_inner[::-1]])
        fill_y = np.concatenate([y_outer, y_inner[::-1]])
        # --- End variable name update ---

        ax.fill(fill_x, fill_y, color='skyblue', alpha=0.4, label=f'Error ($\pm${r_e:.2f})')
        self.add_colorbar(im)
        ax.legend(fontsize='small')
        fname = os.path.join(self.output_dir, f'Figure1_Mean_smoothed_ring_{index:03d}.png')
        try:
            plt.savefig(fname, bbox_inches='tight', dpi=150)
            print(f"Saved ring plot: {fname}")
        except Exception as e:
            print(f"Error saving ring plot '{fname}': {e}")
        plt.close(fig)

    def sample_cylinder(self, y_pix: np.ndarray, x_pix: np.ndarray) -> np.ndarray:
        """Samples the 3D smoothed image stack along ring pixel coordinates."""
        qs = np.zeros((self.nftot, self.ntheta))
        for k in range(self.nftot):
            fd=self.sIall[:,:,k]; qs[k,:]=[self.bilinear_interp(fd,x_pix[l],y_pix[l]) for l in range(self.ntheta)]
        return qs

    def compute_autocorrelation(self, qs: np.ndarray) -> np.ndarray:
        """Computes the centered, normalized 2D autocorrelation of the cylinder."""
        if qs.size==0 or np.all(np.isclose(qs,0)): return np.zeros_like(qs)
        qsn=qs.copy(); self.mean_subtract(qsn);
        if np.all(np.isclose(qsn,0)): return np.zeros_like(qs)
        qk=fft.fft2(qsn); Pk=np.abs(qk)**2; acf=np.real(fft.ifft2(Pk))
        s0,s1=acf.shape[0]//2,acf.shape[1]//2; racf=np.roll(acf,(s0,s1),axis=(0,1))
        pv=racf[s0,s1]; racf/=(pv if pv>1e-10 else 1.0); # Avoid division by zero
        if pv <= 1e-10: racf = np.zeros_like(racf) # Ensure zero ACF if peak is zero
        return racf

    def find_pattern_speed(self, racf: np.ndarray, xi_crit: float, xi_crit_stdunits: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        """Finds pattern speed using moment analysis on thresholded ACF region."""
        nt,nphi=racf.shape; ct,cp=nt//2,nphi//2; rc=racf.copy(); xa=xi_crit
        if xi_crit_stdunits: rs=np.std(racf); xa = xi_crit*rs if rs>1e-9 else xi_crit
        xa = min(xa,0.999); tm=(racf>=xa).astype(int); labs,nf=label(tm)
        if nf==0 or labs[ct,cp]==0:
            print(f"W: No central region xi={xi_crit:.2f}({xa:.3f}). Using 5x5."); Q=np.zeros_like(racf,bool)
            r0,r1=max(0,ct-2),min(nt,ct+3); c0,c1=max(0,cp-2),min(nphi,cp+3); Q[r0:r1,c0:c1]=True; rc[~Q]=0.
        else: cl=labs[ct,cp]; Q=(labs==cl); rc[~Q]=0.; nc=np.count_nonzero(np.sum(Q,axis=0)); #if nc<5: print(f"W: Region width {nc}<5")
        ts=np.arange(nt)-ct; phis=np.arange(nphi)-cp; dtp=self.dt; dpp=self.dtheta
        mt2,mp2,mtp,m0=(0.,)*4; ti,pi=np.where(Q)
        for i,j in zip(ti,pi): v=rc[i,j]; tl=ts[i]; pl=phis[j]; m0+=v; mt2+=v*tl*tl; mp2+=v*pl*pl; mtp+=v*tl*pl
        ps=0.;
        if m0<1e-9: print("W: Moment0~0");
        elif mt2<1e-9: print("W: Moment_t2~0");
        elif dtp==0: print("W: dt=0");
        else: psp=mtp/mt2; ps=psp*dpp/dtp
        if np.isnan(ps): ps=0.
        return ps, rc, Q

    def plot_autocorrelation(self, racf: np.ndarray, racf_cut: np.ndarray, Q: np.ndarray, ps: float, xc: float, is_std: bool):
        """Plots the ACF, masked region Q, and pattern speed line."""
        nt,nphi=racf.shape; tm=(nt/2.)*self.dt; pm=(nphi/2.)*self.dtheta; ext=[-tm,tm,-pm,pm]; xl,yl=ext[0:2],ext[2:4]
        fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(racf.T,cmap='afmhot',aspect='auto',origin='lower',extent=ext,interpolation='bilinear')
        tc=np.linspace(-tm,tm,nt); pc=np.linspace(-pm,pm,nphi); ax.contour(tc,pc,Q.T,levels=[0.5],colors='purple',linewidths=1.0)
        ax.set_xlabel(r'$\Delta t\,[GM/c^3]$'); ax.set_ylabel(r'$\Delta\phi\,[\mathrm{deg}]$')
        xus=r'\\sigma_{ACF}' if is_std else ''; title=f'Autocorrelation($\chi_c={xc:.2f}$'+'$\\sigma_{ACF}$)'
        ax.set_title(title); ax.set_xlim(xl); ax.set_ylim(yl)
        if self.dt>0: tl=np.array(xl); pl=ps*tl; ax.plot(tl,pl,'green',ls='--',alpha=0.8,lw=1.5,label=f'Slope={ps:.2f}deg/$(GM/c^3)$'); ax.legend(fontsize='small')
        self.add_colorbar(im); xs=f"xicrit{xc:.2f}{'_std'if is_std else''}"
        fn=os.path.join(self.output_dir,f'Figure2_Autocorrelation_{xs}.png'); plt.savefig(fn,bbox_inches='tight',dpi=150); plt.close(fig)

    def calculate_pattern_speed(self, x_shift_uas=0., y_shift_uas=0., r_shift_uas=0., xi_crit=0.15, xi_crit_stdunits=True, plot_acf=False) -> float:
        """Calculates pattern speed for given shifts and xi_crit. Optionally plots ACF."""
        _,_,(yp,xp)=self.extract_ring_profile(x_shift_uas,y_shift_uas,r_shift_uas); qs=self.sample_cylinder(yp,xp)
        racf=self.compute_autocorrelation(qs); ps,rc,Q=self.find_pattern_speed(racf,xi_crit,xi_crit_stdunits)
        if plot_acf: self.plot_autocorrelation(racf,rc,Q,ps,xi_crit,xi_crit_stdunits)
        return ps

    def run_xi_crit_survey(self, n_plots=5, save_ring_plot=True):
        """Calculates pattern speed over a range of xi_crit values (zero shift)."""
        if save_ring_plot: self.plot_mean_image_with_ring(index=0)
        if self.xi_crit_stdunits: xcs=np.arange(0.2,4.5,0.05); ustr="std dev units"
        else: xcs=np.arange(0.05,1.0,0.05); ustr="absolute units"
        print(f"Surveying xi_crit in {ustr} from {xcs[0]:.2f} to {xcs[-1]:.2f}")
        ns=len(xcs); pss=np.zeros(ns); pis=np.linspace(0,ns-1,n_plots,dtype=int) if n_plots>0 and ns>0 else []
        print(f"Calculating pattern speed for {ns} xi_crit values...")
        for i,xi in enumerate(xcs):
            sp=(i in pis); pss[i]=self.calculate_pattern_speed(xi_crit=xi,xi_crit_stdunits=self.xi_crit_stdunits,plot_acf=sp)
            if (i+1)%(max(1,ns//10))==0: print(f"... processed {i+1}/{ns}")
        dts=np.array([xcs,pss]);
        try: np.save(self.pattern_speed_vs_xicrit_savefile,dts); print(f"\nResults saved: {self.pattern_speed_vs_xicrit_savefile}")
        except Exception as e: print(f"\nError saving results: {e}")
        self.plot_ps_vs_xicrit(xcs,pss)

    def plot_ps_vs_xicrit(self, xcs: np.ndarray, pss: np.ndarray):
        """Plots pattern speed vs. xi_crit values."""
        fig,ax=plt.subplots(figsize=(7,5)); ax.plot(xcs,pss,'.-',ms=4)
        ax.set_ylabel(r'$\Omega_p\,[\mathrm{deg}/(GM/c^3)]$')
        if self.xi_crit_stdunits: ax.set_xlabel(r'$\chi_c\,[\sigma_{ACF}]$'); title=r'PS vs. ACF Threshold($\sigma$)'
        else: ax.set_xlabel(r'$\chi_c$'); title=r'PS vs. ACF Threshold(Abs)'
        ax.set_title(title); ax.grid(True,ls=':',alpha=0.6)
        usfx='stdunits' if self.xi_crit_stdunits else 'xiunits'
        fn=os.path.join(self.output_dir,f'Figure3_PatternSpeed_vs_XiCrit_{usfx}.png')
        plt.savefig(fn,bbox_inches='tight',dpi=150); print(f"Saved PS vs Xi_crit plot: {fn}"); plt.close(fig)


# %%############################################################################
# Argument Parsing and Main Execution
# ##############################################################################

def parse_arguments():
    """
    Parse command line arguments (Runs RingFitter first).

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Fit ring parameters and calculate pattern speed across xi_crit values.')

    parser.add_argument('file_path', type=str, help='Path to the input HDF5 data file (movie)')
    # No ring_file_path argument needed here
    parser.add_argument('output_dir', type=str, help='Base directory to save output subfolder')
    parser.add_argument('--n_acf_plots', type=int, default=5,
                        help='Number of example autocorrelation plots to generate (default: 5)')

    # Optional arguments for RingFitter (defaults match user's hardcoded values)
    parser.add_argument('--fov', type=float, default=200.0, help='Ring Fitting: Field of view in uas (default: 200)')
    parser.add_argument('--npix', type=int, default=200, help='Ring Fitting: Number of pixels (default: 200)')
    parser.add_argument('--min_radius', type=float, default=10.0, help='Ring Fitting: Min radius in uas (default: 10)')
    parser.add_argument('--max_radius', type=float, default=100.0, help='Ring Fitting: Max radius in uas (default: 100)')
    # Can add --search_radius_min/max if needed to be different from min/max_radius

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Create a specific output subdirectory based on the input file name
    base_output_dir = args.output_dir
    # Sanitize input filename to create a valid directory name
    input_basename = os.path.basename(args.file_path)
    input_name_part = os.path.splitext(input_basename)[0]
    # Replace potentially problematic characters for directory names
    safe_subdir_name = "".join([c if c.isalnum() else "_" for c in input_name_part]) + '_cylinder_output'
    specific_output_dir = base_output_dir #os.path.join(base_output_dir, safe_subdir_name)

    # Ensure the specific output directory exists
    if not os.path.exists(specific_output_dir):
        try:
            os.makedirs(specific_output_dir)
            print(f"Created output directory: {specific_output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{specific_output_dir}': {e}")
            sys.exit(1)
    else:
         print(f"Using existing output directory: {specific_output_dir}")


    # Define the path for the ring parameters CSV file WITHIN the specific output dir
    ring_output_file = os.path.join(specific_output_dir, "ring_parameters.csv")

    print("--- Running Ring Fitting ---")
    # Run ring fitting using parameters from args or defaults
    try:
         # Use parameters from args namespace
         results_df = fit_ring_from_file(
             args.file_path,
             ring_output_file,
             fov=args.fov * eh.RADPERUAS, # Convert uas to radians
             npix=args.npix,
             min_radius=args.min_radius, # uas
             max_radius=args.max_radius, # uas
             # search_radius_min/max will default to min/max_radius inside the function
         )
         print("Ring fitting completed successfully.")
         # Optional: print(results_df)
    except ImportError:
         print("Error: 'ehtim' library not found or failed to import.")
         print("Please ensure ehtim is installed correctly to run ring fitting.")
         sys.exit(1)
    except FileNotFoundError:
         print(f"Error: Input HDF5 file not found at '{args.file_path}'")
         sys.exit(1)
    except Exception as e:
         print(f"An error occurred during ring fitting: {e}")
         # import traceback # Uncomment for detailed debug info
         # traceback.print_exc() # Uncomment for detailed debug info
         print("Exiting due to ring fitting error.")
         sys.exit(1)


    print("\n--- Running Cylinder Analysis (Xi_crit Survey) ---")
    # Initialize the analysis class using the newly GENERATED ring file path
    try:
        cylinder = CylinderAnalysis(
            file_path=args.file_path,
            ring_file_path=ring_output_file, # Use the generated file
            output_dir=specific_output_dir, # Use the specific dir
            xi_crit_stdunits=True
        )

        # Run the xi_crit survey
        cylinder.run_xi_crit_survey(n_plots=args.n_acf_plots, save_ring_plot=True)

        print("\nAnalysis finished successfully.")
        print(f"Output files saved to: {cylinder.output_dir}") # refers to specific_output_dir

    except FileNotFoundError:
         # This might happen if ring fitting failed silently or file was deleted
         print(f"Error: Ring parameter file expected at '{ring_output_file}' was not found.")
         print("This could indicate an issue during the ring fitting stage.")
         sys.exit(1)
    except Exception as e:
         print(f"An error occurred during cylinder analysis: {e}")
         # import traceback # Uncomment for detailed debug info
         # traceback.print_exc() # Uncomment for detailed debug info
         sys.exit(1)