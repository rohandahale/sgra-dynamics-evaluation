"""
Pattern Speed Uncertainty Calculation Using Monte Carlo Methods

Original Author: Nick Conroy, Date: 28 March 2025
Edited: Rohan Dahale, Date: 08 April 2025

This script calculates pattern speed and its uncertainty from input hdf5 video using
Monte Carlo sampling.
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
import ehtim as eh
from scipy import interpolate, stats
from copy import deepcopy

from utilities import *
colors, titles, labels, mfcs, mss = common()

# Constants
RADPERUAS = 4.848136811094136e-12
GM_C3 = 20.46049 / 3600  # hours
M_PER_RAD = 41139576306.70914

class RingFitter:
    """Class for fitting and extracting ring parameters from EHT images."""
    
    def __init__(self, fov=200*eh.RADPERUAS, npix=200):
        """Initialize the RingFitter.
        
        Args:
            fov: Field of view in radians
            npix: Number of pixels in each dimension
        """
        self.fov = fov
        self.npix = npix
    
    def fit_from_file(self, input_file, output_file="ring.csv", **kwargs):
        """Process a movie file to extract ring parameters and save to CSV.
        
        Args:
            input_file: Path to the hdf5 movie file
            output_file: Path to save the CSV output
            **kwargs: Any parameters to pass to underlying methods
            
        Returns:
            DataFrame containing ring parameters
        """
        # Extract fov and npix from kwargs if provided
        fov = kwargs.pop('fov', self.fov)
        npix = kwargs.pop('npix', self.npix)
        
        # Load the movie and get average frame
        mv = eh.movie.load_hdf5(input_file)
        im = mv.avg_frame().regrid_image(fov, npix)
        
        # Extract ring parameters
        return self.fit_from_image(im, output_file, **kwargs)
    
    def fit_from_image(self, image, output_file=None, **kwargs):
        """Extract ring parameters from an image and optionally save to CSV.
        
        Args:
            image: ehtim Image object
            output_file: Optional path to save CSV output
            **kwargs: Any parameters to pass to underlying methods
            
        Returns:
            DataFrame containing ring parameters
        """
        # Extract parameters for finding center
        center_x = kwargs.pop('center_x', None)
        center_y = kwargs.pop('center_y', None)
        search_radius_min = kwargs.pop('search_radius_min', 10)
        search_radius_max = kwargs.pop('search_radius_max', 100)
        nrays_search = kwargs.pop('nrays_search', 25)
        nrs_search = kwargs.pop('nrs_search', 50)
        fov_search = kwargs.pop('fov_search', 0.1)
        n_search = kwargs.pop('n_search', 20)
        image_blur_fwhm = kwargs.pop('image_blur_fwhm', 2.0*eh.RADPERUAS)
        image_threshold = kwargs.pop('image_threshold', 0.05)
        
        # Find the center of the ring if not provided
        if center_x is None or center_y is None:
            center_x, center_y = self._find_center(
                image,
                search_radius_min=search_radius_min,
                search_radius_max=search_radius_max,
                nrays_search=nrays_search,
                nrs_search=nrs_search,
                fov_search=fov_search,
                n_search=n_search,
                blur_fwhm=image_blur_fwhm,
                threshold=image_threshold
            )
        
        # Extract all ring parameters with remaining kwargs
        params = self._extract_ring_parameters(image, center_x, center_y, **kwargs)
        
        # Calculate center coordinates relative to image center
        fov = kwargs.pop('fov', self.fov)
        x0 = center_x - fov/2/eh.RADPERUAS
        y0 = center_y - fov/2/eh.RADPERUAS
        
        # Create output dataframe
        df = pd.DataFrame({
            "x0": [x0],
            "y0": [y0],
            "r": [params["D"]/2],
            "r_err": [params["Derr"]/2]
        })
        
        # Add any additional parameters the user might want
        for key, value in params.items():
            if key not in ["xc", "yc", "D", "Derr"]:
                df[key] = [value]
        
        # Save to CSV if output file is specified
        if output_file:
            df.to_csv(output_file, index=False)
        
        return df
    
    def _find_center(self, image, search_radius_min=10, search_radius_max=100,
                    nrays_search=25, nrs_search=50, fov_search=0.1, n_search=20,
                    blur_fwhm=2.0*eh.RADPERUAS, threshold=0.05):
        """Find the center of a ring in an image.
        
        Args:
            image: ehtim Image object
            search_radius_min: Minimum search radius
            search_radius_max: Maximum search radius
            nrays_search: Number of angles to search
            nrs_search: Number of radii to search
            fov_search: Field of view for search as fraction of image
            n_search: Number of search iterations
            blur_fwhm: FWHM for Gaussian blur pre-processing
            threshold: Threshold for intensity cutoff
            
        Returns:
            Tuple of (x_center, y_center)
        """
        image_blur = image.blur_circ(blur_fwhm)
        image_mod = image_blur.threshold(cutoff=threshold)
        xc, yc = eh.features.rex.findCenter(
            image_mod, 
            rmin_search=search_radius_min, 
            rmax_search=search_radius_max,
            nrays_search=nrays_search, 
            nrs_search=nrs_search,
            fov_search=fov_search, 
            n_search=n_search
        )
        return xc, yc
    
    def _extract_ring_parameters(self, image, center_x, center_y, 
                                min_radius=5, max_radius=50, 
                                n_angles=360, n_radial=100,
                                return_full_data=False):
        """Extract detailed ring parameters from an image.
        
        Args:
            image: ehtim Image object
            center_x: x-coordinate of ring center
            center_y: y-coordinate of ring center
            min_radius: Minimum radius to consider
            max_radius: Maximum radius to consider
            n_angles: Number of angles to sample
            n_radial: Number of radial points to sample
            return_full_data: Whether to return the full radial image data
            
        Returns:
            Dictionary of ring parameters
        """
        # Setup radial and angular grid
        angles = np.linspace(0, 360, n_angles)
        angles_rad = np.deg2rad(angles)
        radial_points = np.linspace(0, max_radius, n_radial)
        dr = radial_points[1] - radial_points[0]
        
        # Create interpolation function for image using RectBivariateSpline instead of interp2d
        x = np.arange(image.xdim) * image.psize / eh.RADPERUAS
        #y = np.flip(np.arange(image.ydim) * image.psize / eh.RADPERUAS)
        y = np.arange(image.ydim) * image.psize / eh.RADPERUAS
        z = image.imarr()
        image_interp = interpolate.RectBivariateSpline(y, x, z)
        
        # Sample image in polar coordinates
        radial_image = np.zeros([n_radial, n_angles])
        r_mesh, angle_mesh = np.meshgrid(radial_points, angles_rad)
        x_mesh = r_mesh * np.sin(angle_mesh) + center_x
        y_mesh = r_mesh * np.cos(angle_mesh) + center_y
        
        for r in range(n_radial):
            # Use the vectorized evaluation method of RectBivariateSpline
            vals = [image_interp(y_mesh[i][r], x_mesh[i][r], grid=False) for i in range(len(angles))]
            radial_image[r, :] = np.array(vals)
        radial_image = np.fliplr(radial_image)
        
        # Calculate ring parameters at each angle
        r_peak_values = []
        r_min_values = []
        r_max_values = []
        
        # Use far radius as intensity floor reference
        far_radius_idx = np.argmin(np.abs(radial_points - max_radius))
        intensity_floor = radial_image[far_radius_idx, :].mean()
        
        # Calculate peak radius and width at each angle
        for angle_idx in range(len(angles)):
            intensity_profile = radial_image[:, angle_idx] - intensity_floor
            intensity_profile[np.where(radial_points < min_radius)] = 0
            
            peak_idx = np.argmax(intensity_profile)
            r_peak = radial_points[peak_idx]
            
            # Refine peak location with quadratic interpolation
            if peak_idx > 0 and peak_idx < n_radial - 1:
                nearby_values = intensity_profile[peak_idx-1:peak_idx+2]
                r_peak = self._quad_interp_radius(r_peak, dr, nearby_values)[0]
            
            # Calculate ring width
            r_min, r_max = self._calc_width(intensity_profile, radial_points, r_peak)
            
            # Store results
            r_peak_values.append(r_peak)
            r_min_values.append(r_min)
            r_max_values.append(r_max)
        
        # Calculate average ring diameter and width
        ring_diameter = np.mean(r_peak_values) * 2
        diameter_error = np.std(r_peak_values) * 2
        
        result = {
            "xc": center_x,
            "yc": center_y,
            "D": ring_diameter,
            "Derr": diameter_error
        }
        
        # Include full data if requested
        if return_full_data:
            result.update({
                "radial_image": radial_image,
                "radial_points": radial_points,
                "angles": angles,
                "r_peak_values": r_peak_values,
                "r_min_values": r_min_values,
                "r_max_values": r_max_values
            })
            
        return result
    
    def _quad_interp_radius(self, r_max, dr, val_list):
        """Quadratic interpolation to find peak radius.
        
        Args:
            r_max: Initial radius estimate
            dr: Radial spacing
            val_list: Intensity values at r-dr, r, r+dr
            
        Returns:
            Tuple of (refined_radius, peak_value)
        """
        v_L, v_max, v_R = val_list
        rpk = r_max + dr*(v_L - v_R) / (2 * (v_L + v_R - 2*v_max))
        vpk = 8*v_max*(v_L + v_R) - (v_L - v_R)**2 - 16*v_max**2
        vpk /= (8*(v_L + v_R - 2*v_max))
        return (rpk, vpk)
    
    def _calc_width(self, intensity, radial_points, peak_radius):
        """Calculate ring width from half max points.
        
        Args:
            intensity: Array of intensity values
            radial_points: Array of radial coordinates
            peak_radius: Radius of the peak
            
        Returns:
            Tuple of (r_min, r_max) at half-maximum
        """
        spline = interpolate.UnivariateSpline(radial_points, intensity-0.5*intensity.max(), s=0)
        roots = spline.roots()
        
        if len(roots) == 0:
            return radial_points[0], radial_points[-1]

        r_min = radial_points[0]
        r_max = radial_points[-1]
        for root in np.sort(roots):
            if root < peak_radius:
                r_min = root
            else:
                r_max = root
                break
        return r_min, r_max


def fit_ring_from_file(input_file, output_file="ring.csv", **kwargs):
    """Convenience function for simple one-line usage.
    
    Args:
        input_file: Path to hdf5 movie file
        output_file: Path to CSV output file
        **kwargs: Any customization parameters
        
    Returns:
        DataFrame with ring parameters
    """
    fitter = RingFitter(
        fov=kwargs.pop('fov', 200*eh.RADPERUAS),
        npix=kwargs.pop('npix', 200)
    )
    return fitter.fit_from_file(input_file, output_file, **kwargs)


class CylinderAnalysis:
    """Class for performing pattern speed analysis on cylinder plots."""

    def __init__(self, file_path: str, ring_file_path: str, output_dir: str):
        """
        Initialize the CylinderAnalysis class with input and output paths.
        
        Args:
            file_path: Path to the main HDF5 data file
            ring_file_path: Path to the CSV file containing ring parameters
            output_dir: Directory to save output files
        """
        self.file_path = file_path
        self.ring_file_path = ring_file_path
        
        # Set up output directory
        self.output_dir = output_dir
        if not self.output_dir.endswith('_cylinder_output/'):
            self.output_dir += '_cylinder_output/'
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Load data
        self.hfp = h5py.File(file_path, 'r')
        self.df = pd.read_csv(ring_file_path)
        
        # Parse dimensions and scales
        self.N, self.M = self.hfp['I'].shape[1:]  # x,y-size of array
        self.dx = self.dy = float(self.hfp['header'].attrs['psize']) / RADPERUAS  # x,y-size of pixel
        self.times = np.array(self.hfp['times'])
        self.nftot = len(self.times)
        self.dt = round(np.diff(self.times).mean() / GM_C3, 1)
        
        # Calculate field of view
        self.FOV_uas = self.dx * self.N
        self.FOV_rad = self.FOV_uas * RADPERUAS
        self.FOV_M = self.FOV_rad * M_PER_RAD
        
        # Tmax in GM/c^3 units
        self.Tmax = self.times[-1] / GM_C3
        
        # Process image data
        self.Iall = np.transpose(self.hfp['I'], [1, 2, 0])
        
        # Circle parameters for sampling
        self.ntheta = 180
        self.theta = np.linspace(0., 2.*np.pi, self.ntheta, endpoint=False)
        self.dtheta = 360 / self.ntheta
        self.pa = self.theta + 0.5*np.pi  # position angle
        
        # Calculate mean images
        self.mean_im = np.mean(self.Iall, axis=2)
        
        # Get smoothed images
        self.sig = 20/(self.dx * (2*np.sqrt(2*np.log(2))))  # 20 uas FWHM resolution
        self.sIall = ndimage.gaussian_filter(self.Iall, sigma=(self.sig, self.sig, 0))
        self.mean_sim = np.mean(self.sIall, axis=2)
        
        # Output file paths
        self.pattern_speed_samples_savefile = os.path.join(self.output_dir, 'pattern_speed_samples.npy')
        self.pattern_speed_uncertainty_savefile = os.path.join(self.output_dir, 'pattern_speed_uncertainty.npy')
        
        # Determine xi_crit factor based on input file type
        self.bestbet_xi_crit_stdfactor = self._determine_xi_crit_factor()

    def _determine_xi_crit_factor(self) -> float:
        """
        Determine the appropriate xi_crit standard factor based on the input file type.
        
        Returns:
            The standard factor to use for xi_crit
        """
        if 'truth' in self.file_path:
            if 'hs' in self.file_path:
                return 2.0
            else:
                return 3.0
        elif 'modeling_mean' in self.file_path:
            return 0.6 #3.7 #4.4
        elif 'resolve_mean' in self.file_path:
            return 0.2 #0.5
        elif 'doghit' in self.file_path:
            return 1.2 #1.3
        elif 'ehtim' in self.file_path:
            return 0.6 #2.2 #0.8
        elif 'kine' in self.file_path:
            return 0.6 #0.3
        elif 'ngmem' in self.file_path:
            return 0.4 #0.2
        else:
            # Default value if no matching file type
            return 0.6

    @staticmethod
    def add_colorbar(mappable):
        """
        Add a colorbar to a mappable object (like an image plot).
        
        Args:
            mappable: The matplotlib mappable object
            
        Returns:
            A matplotlib colorbar object
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    @staticmethod
    def bilinear_interp(dat, x, y):
        """
        Perform bilinear interpolation for the given data at position (x,y).
        
        Args:
            dat: 2D data array
            x: x-coordinate
            y: y-coordinate
            
        Returns:
            Interpolated value
        """
        i = int(round(x-0.5))
        j = int(round(y-0.5))
        
        # Bilinear interpolation
        di = x-i
        dj = y-j
        z00 = dat[i, j]
        z01 = dat[i, j+1]
        z10 = dat[i+1, j]
        z11 = dat[i+1, j+1]
        
        return z00*(1.-di)*(1.-dj) + z01*(1.-di)*dj + z10*di*(1.-dj) + z11*di*dj

    @staticmethod
    def mean_subtract(m):
        """
        Subtract the mean from each row and column of a 2D array.
        
        Args:
            m: 2D array to process
        """
        ntheta = m.shape[1]
        nftot = m.shape[0]
        
        for i in range(ntheta):
            m[:, i] = (m[:, i] - m[:, i].mean())
        for i in range(nftot):
            m[i, :] = (m[i, :] - m[i, :].mean())

    def extract_ring_profile(self, x_shift_factor: float, y_shift_factor: float, 
                            r_shift_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the ring profile from the smoothed image with the given shifts.
        
        Args:
            x_shift_factor: Shift in x coordinate (in microarcseconds)
            y_shift_factor: Shift in y coordinate (in microarcseconds)
            r_shift_factor: Shift in radius (in microarcseconds)
            
        Returns:
            Tuple containing (x coordinates, y coordinates, circular indices)
        """
        # Get mean ring parameters from dataframe
        r = np.mean(self.df['r'])
        x0 = np.mean(self.df['x0'])
        y0 = np.mean(self.df['y0'])

        # Convert ring to pixels
        r_pix = r * (self.N / self.FOV_uas)
        x0 = x0 * (self.N / self.FOV_uas)
        y0 = y0 * (self.N / self.FOV_uas)

        # Apply shifts
        x0 = x0 + x_shift_factor * (self.N / self.FOV_uas)
        y0 = y0 + y_shift_factor * (self.N / self.FOV_uas)

        # Calculate circular indices
        ishift = -y0 + self.M / 2
        jshift = -x0 + self.N / 2
        icirc = ishift + r_pix * np.sin(self.pa)
        jcirc = jshift + r_pix * np.cos(self.pa)
        
        # Convert to physical coordinates
        x = (icirc - self.N / 2) * self.dx
        y = (jcirc - self.M / 2) * self.dy
        
        return x, y, (icirc, jcirc)

    def plot_mean_image_with_ring(self, index: int = 0):
        """
        Plot the mean image with the ring overlay and radius error.

        Args:
            index: Index for the figure name (used in the output filename).
        """
        # Get mean ring parameters from the dataframe
        # Ensure the dataframe `self.df` is loaded correctly in __init__
        if self.df is None or self.df.empty:
             print("Error: Ring parameters dataframe (self.df) is not loaded or empty.")
             return # Or handle the error appropriately

        try:
            r_mean = np.mean(self.df['r'])
            x0_mean = np.mean(self.df['x0'])
            y0_mean = np.mean(self.df['y0'])
            rerr_mean = np.mean(self.df['r_err'])
        except KeyError as e:
            print(f"Error: Missing expected column in ring dataframe: {e}")
            return # Or handle the error appropriately


        # Define inner and outer radii for the uncertainty band
        r_inner = r_mean - rerr_mean
        r_outer = r_mean + rerr_mean

        # Ensure inner radius is non-negative
        r_inner = max(0, r_inner)

        # Calculate coordinates for the mean ring, and the inner/outer boundaries
        # using the position angles defined in __init__ (self.pa)
        x_center = x0_mean + r_mean * np.cos(self.pa)
        y_center = y0_mean + r_mean * np.sin(self.pa)
        x_outer = x0_mean + r_outer * np.cos(self.pa)
        y_outer = y0_mean + r_outer * np.sin(self.pa)
        x_inner = x0_mean + r_inner * np.cos(self.pa)
        y_inner = y0_mean + r_inner * np.sin(self.pa)

        # --- Plotting ---
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)

        # Display the mean image
        im = ax.imshow(self.mean_im, cmap='afmhot', aspect='auto', interpolation='bilinear',
                       origin='lower', extent=[-self.FOV_uas/2, self.FOV_uas/2,
                                               -self.FOV_uas/2, self.FOV_uas/2])
        plt.title('Mean Image with Ring and Uncertainty')
        plt.axis('scaled') # Ensure aspect ratio is equal
        plt.xlim(-self.FOV_uas/2, self.FOV_uas/2)
        plt.ylim(-self.FOV_uas/2, self.FOV_uas/2)
        plt.xlabel(r'$x\, [\mu{\rm as}]$')
        plt.ylabel(r'$y\, [\mu{\rm as}]$')

        # Plot the mean ring (optional, changed color for visibility)
        ax.plot(x_center, y_center, color='blue', ls='--', lw=1, label=f'Mean Radius ({r_mean:.2f} $\mu$as)')

        # Create coordinates for the filled polygon representing the uncertainty
        # Concatenate outer boundary points and reversed inner boundary points
        fill_x = np.concatenate([x_outer, x_inner[::-1]])
        fill_y = np.concatenate([y_outer, y_inner[::-1]])

        # Plot the filled uncertainty ring
        ax.fill(fill_x, fill_y, color='blue', alpha=0.4, label=f'Radius Error ($\pm${rerr_mean:.2f} $\mu$as)')

        # Add a colorbar for the image
        self.add_colorbar(im) # Use the existing add_colorbar method

        # Add legend
        ax.legend(fontsize='small')

        # Save the figure
        output_filename = os.path.join(self.output_dir, f'Figure1_Mean_unsmoothed_ring_{index:03d}.png')
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Saved ring plot to: {output_filename}") # Optional: confirmation message
        plt.close(fig)


    def sample_cylinder(self, icirc: np.ndarray, jcirc: np.ndarray) -> np.ndarray:
        """
        Sample the 3D image along the ring to create a cylinder.
        
        Args:
            icirc: i (row) indices of the ring
            jcirc: j (column) indices of the ring
            
        Returns:
            The sampled cylinder (time x angle)
        """
        qs = np.zeros((self.nftot, self.ntheta))
        for k in range(self.nftot):
            for l in range(self.ntheta):
                qs[k, l] = self.bilinear_interp(self.sIall[:, :, k], icirc[l], jcirc[l])
        return qs

    def compute_autocorrelation(self, qs: np.ndarray) -> np.ndarray:
        """
        Compute the autocorrelation of the cylinder plot.
        
        Args:
            qs: The cylinder data (time x angle)
            
        Returns:
            The normalized autocorrelation function
        """
        # Create normalized cylinder plot
        qsn = np.copy(qs)
        self.mean_subtract(qsn)
        
        # Compute autocorrelation via FFT
        qk = fft.fft2(qsn)
        Pk = np.absolute(qk)**2
        acf = np.real(fft.ifft2(Pk))
        
        # Shift the peak to the center
        shifti = int(acf.shape[0]/2.)
        shiftj = int(acf.shape[1]/2.)
        racf = np.roll(acf, (shifti, shiftj), axis=(0, 1))
        
        # Normalize
        racf /= np.max(racf)
        
        return racf

    def find_pattern_speed(self, racf: np.ndarray, xi_crit: float, 
                          xi_crit_stdunits: bool = False) -> Tuple[float, np.ndarray]:
        """
        Find the pattern speed from the autocorrelation function.
        
        Args:
            racf: The normalized autocorrelation
            xi_crit: Correlation threshold value
            xi_crit_stdunits: If True, xi_crit is in standard deviation units
            
        Returns:
            Tuple of (pattern speed, filtered autocorrelation)
        """
        # Apply threshold in standard deviation units if requested
        if xi_crit_stdunits:
            racf_std = np.std(racf)
            xi_crit = xi_crit * racf_std
            
        racf_cut = np.copy(racf)
        
        # Find the connected region containing the central peak
        labels, num_features = label((racf > xi_crit).astype(int))
        Q = labels == labels[racf.shape[0] // 2, racf.shape[1] // 2]
        non_zero_columns = np.count_nonzero(np.sum(Q, axis=0))
        
        # Handle edge cases with too high xi_crit or not enough data
        if (xi_crit >= 1) or (non_zero_columns < 5):
            non_zero_columns = 5
            central_row_index = racf.shape[0] // 2
            central_column_index = racf.shape[1] // 2
            
            # Calculate new edge_value
            xi_crit = np.max(racf[:, int(central_column_index + non_zero_columns // 2)])
            
            # Create a new mask
            threshold_mask = (racf >= xi_crit).astype(int) & \
                (np.abs(np.arange(racf.shape[1]) - central_column_index) <= non_zero_columns // 2)
            
            labels, num_features = label(threshold_mask)
            central_peak_label = labels[central_row_index, central_column_index]
            Q = labels == central_peak_label

        # Set up coordinate grid
        ts = np.linspace(-len(racf)/2, len(racf)/2, len(racf), endpoint=False)
        phis = np.linspace(-len(racf[0])/2, len(racf[0])/2, len(racf[0]), endpoint=False)
        delta_t = ts[1] - ts[0]
        delta_phi = phis[1] - phis[0]

        # Calculate moments
        moment_t = 0
        moment_phi = 0
        moment_t_phi = 0
        moment = 0
        
        for i in range(len(ts)):
            for j in range(len(phis)):
                if not Q[i, j]:
                    racf_cut[i, j] = 0.0
                    continue
                    
                moment += racf_cut[i, j]
                moment_t += racf_cut[i, j] * ts[i] * ts[i]
                moment_phi += racf_cut[i, j] * phis[j] * phis[j]
                moment_t_phi += racf_cut[i, j] * ts[i] * phis[j]

        # Normalize moments
        moment *= delta_t * delta_phi
        moment_t *= delta_t * delta_phi / moment
        moment_phi *= delta_t * delta_phi / moment
        moment_t_phi *= delta_t * delta_phi / moment

        # Calculate pattern speed
        pattern_speed = moment_t_phi / moment_t
        pattern_speed = pattern_speed * self.dtheta / self.dt
        
        # Handle nan cases (symmetrical vertical peak)
        if np.isnan(pattern_speed):
            pattern_speed = 0
            
        return pattern_speed, racf_cut

    def plot_autocorrelation(self, racf: np.ndarray, racf_cut: np.ndarray, 
                            Q: np.ndarray, pattern_speed: float):
        """
        Plot the autocorrelation function with pattern speed.
        
        Args:
            racf: The normalized autocorrelation
            racf_cut: The filtered autocorrelation
            Q: The mask used for filtering
            pattern_speed: The calculated pattern speed
        """
        # Calculate plot extent
        extent = [-(0.5*self.nftot + 0.5)*self.dt, (0.5*self.nftot - 0.5)*self.dt,
                 -(0.5*self.ntheta + 0.5)*self.dtheta, (0.5*self.ntheta - 0.5)*self.dtheta]
        xlim = [-(0.5*self.nftot + 0.5)*self.dt, (0.5*self.nftot - 0.5)*self.dt]
        ylim = [-(0.5*self.ntheta + 0.5)*self.dtheta, (0.5*self.ntheta - 0.5)*self.dtheta]
        
        # For odd number of frames, shift by half a pixel
        if (self.nftot % 2) != 0:
            extent = extent + np.array([+self.dt/2, +self.dt/2, 0, 0])
            
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        im = ax.imshow(racf.T, cmap='afmhot', aspect='auto', origin='lower',
                      extent=extent, interpolation='bilinear')
        
        # Save array data for later use
        np.save(os.path.join(self.output_dir, 'autocorrelation.npy'), racf)
        np.save(os.path.join(self.output_dir, 'autocorrelation_xlim.npy'), 
               [-len(racf[:, 0])*self.dt/2., len(racf[:, 0])*self.dt/2.])
        np.save(os.path.join(self.output_dir, 'autocorrelation_ylim.npy'), 
               [-len(racf[0, :])*self.dtheta/2, len(racf[0, :])*self.dtheta/2.])
        
        # Plot contour of the mask
        plt.contour(Q.T, extent=extent, origin='lower', levels=[0.5], c='purple')
        
        # Add labels and title
        ax.set_xlabel(r'$\Delta t [G M/c^3]$')
        ax.set_ylabel(r'$\Delta \mathrm{PA} [{\rm deg}]$')
        ax.set_title('Autocorrelation')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Plot the pattern speed line
        x_vals = np.arange(0, 5*self.nftot, 1)
        y_vals = pattern_speed * x_vals
        ax.plot(x_vals, y_vals, 'g--', alpha=0.6, 
               label=f'Measured Slope {pattern_speed:0.2f} [deg / GMc^-3]')
        
        self.add_colorbar(im)
        ax.legend()
        plt.savefig(os.path.join(self.output_dir, 'Figure2_Autocorrelation.png'), 
                   bbox_inches='tight')
        plt.close(fig)

    def find_bestbet_xi_crit(self, x_shift_factor: float=0, y_shift_factor: float=0, 
                           r_shift_factor: float=0, xi_crit: float=1.0, 
                           xi_crit_stdunits: bool=True, plot: bool=False) -> float:
        """
        Find the best xi_crit value for the given parameters.
        
        Args:
            x_shift_factor: Shift in x coordinate
            y_shift_factor: Shift in y coordinate
            r_shift_factor: Shift in radius
            xi_crit: Initial xi_crit value
            xi_crit_stdunits: If True, xi_crit is in standard deviation units
            plot: Whether to plot the autocorrelation
            
        Returns:
            The best xi_crit value
        """
        # Extract ring and compute cylinder
        x, y, (icirc, jcirc) = self.extract_ring_profile(x_shift_factor, y_shift_factor, r_shift_factor)
        
        # Plot mean image with ring if requested
        if plot:
            self.plot_mean_image_with_ring()
            
        # Sample cylinder and compute autocorrelation
        qs = self.sample_cylinder(icirc, jcirc)
        racf = self.compute_autocorrelation(qs)
        
        # Convert xi_crit to absolute units if needed
        if xi_crit_stdunits:
            racf_std = np.std(racf)
            xi_crit_abs = xi_crit * racf_std
        else:
            xi_crit_abs = xi_crit
            
        # Plot autocorrelation if requested
        if plot:
            # Find the connected region containing the central peak
            labels, _ = label((racf > xi_crit_abs).astype(int))
            Q = labels == labels[racf.shape[0] // 2, racf.shape[1] // 2]
            
            # Calculate pattern speed
            pattern_speed, racf_cut = self.find_pattern_speed(racf, xi_crit_abs)
            
            # Plot the autocorrelation
            self.plot_autocorrelation(racf, racf_cut, Q, pattern_speed)
            
        return xi_crit_abs

    def calculate_pattern_speed(self, x_shift_factor: float=0, y_shift_factor: float=0,
                               r_shift_factor: float=0, xi_crit: float=0.15,
                               xi_crit_stdunits: bool=False, plot: bool=False) -> float:
        """
        Calculate pattern speed for the given parameters.
        
        Args:
            x_shift_factor: Shift in x coordinate
            y_shift_factor: Shift in y coordinate
            r_shift_factor: Shift in radius
            xi_crit: Correlation threshold value
            xi_crit_stdunits: If True, xi_crit is in standard deviation units
            plot: Whether to plot intermediate results
            
        Returns:
            The calculated pattern speed
        """
        # Extract ring and compute cylinder
        x, y, (icirc, jcirc) = self.extract_ring_profile(x_shift_factor, y_shift_factor, r_shift_factor)
        
        # Plot mean image with ring if requested
        if plot:
            self.plot_mean_image_with_ring()
            
        # Sample cylinder and compute autocorrelation
        qs = self.sample_cylinder(icirc, jcirc)
        racf = self.compute_autocorrelation(qs)
        
        # Calculate pattern speed
        pattern_speed, _ = self.find_pattern_speed(racf, xi_crit, xi_crit_stdunits)
        
        return pattern_speed

    def run_monte_carlo(self, n_samples: int=1000, save_plot: bool=True) -> dict:
        """
        Run Monte Carlo simulation to estimate pattern speed uncertainty.
        
        Args:
            n_samples: Number of Monte Carlo samples
            save_plot: Whether to save plots
            
        Returns:
            Dictionary with pattern speed statistics
        """
        # Find best xi_crit value
        bestbet_xicrit = self.find_bestbet_xi_crit(
            x_shift_factor=0, y_shift_factor=0, r_shift_factor=0,
            xi_crit=self.bestbet_xi_crit_stdfactor, xi_crit_stdunits=True,
            plot=save_plot
        )
        
        # Generate parameter samples
        sigma = 0.7  # sigma fit from xi_crit RMSE curves
        a, b = (0 - bestbet_xicrit) / sigma, (1 - bestbet_xicrit) / sigma
        xi_crit_samples = truncnorm.rvs(a, b, loc=bestbet_xicrit, scale=sigma, size=n_samples)
        
        # Position and radius uncertainty
        rerr = np.mean(self.df['r_err'])
        x_factor_samples = np.random.normal(0, rerr, n_samples)
        y_factor_samples = np.random.normal(0, rerr, n_samples)
        r_factor_samples = np.random.normal(0, rerr, n_samples)
        
        # Calculate pattern speed for each sample
        pattern_speed_samples = np.array([
            self.calculate_pattern_speed(x, y, r, xi) 
            for x, y, r, xi in zip(x_factor_samples, y_factor_samples, 
                                  r_factor_samples, xi_crit_samples)
        ])
        
        # Calculate statistics
        bestbet_ps = self.calculate_pattern_speed(0, 0, 0, bestbet_xicrit)
        mean_ps = np.mean(pattern_speed_samples)
        std_ps = np.std(pattern_speed_samples)
        percentiles = np.percentile(pattern_speed_samples, [15.865, 50, 84.135])  # 1-sigma range
        
        # Calculate median statistics
        median = percentiles[1]
        median_plus_sigma = percentiles[2] - median
        median_minus_sigma = median - percentiles[0]
        
        # Find mode (peak of histogram)
        counts, bin_edges = np.histogram(pattern_speed_samples, bins=50)
        modal_bin_index = np.argmax(counts)
        modal_bin_center = 0.5 * (bin_edges[modal_bin_index] + bin_edges[modal_bin_index + 1])
        mode_value = modal_bin_center
        mode_plus_sigma = percentiles[2] - mode_value
        mode_minus_sigma = mode_value - percentiles[0]
        
        # Calculate bestbet uncertainties
        #plus_sigma = percentiles[2] - bestbet_ps
        #minus_sigma = bestbet_ps - percentiles[0]
        
        plus_sigma  = np.sqrt(np.mean((pattern_speed_samples[pattern_speed_samples > bestbet_ps] - bestbet_ps)**2))
        minus_sigma = np.sqrt(np.mean((pattern_speed_samples[pattern_speed_samples < bestbet_ps] - bestbet_ps)**2))
        
        # Save results
        pattern_speed_uncertainty_array = np.array([
            bestbet_ps, plus_sigma, minus_sigma, 
            median, median_plus_sigma, median_minus_sigma,
            mode_value, mode_plus_sigma, mode_minus_sigma
        ])
        
        np.save(self.pattern_speed_samples_savefile, pattern_speed_samples)
        np.save(self.pattern_speed_uncertainty_savefile, pattern_speed_uncertainty_array)
        
        # Plot histogram if requested
        if save_plot:
            self.plot_histogram(pattern_speed_samples, bestbet_ps, plus_sigma, minus_sigma,
                              median, median_plus_sigma, median_minus_sigma,
                              mode_value, mode_plus_sigma, mode_minus_sigma)
        
        # Return statistics dictionary
        return {
            'bestbet_ps': bestbet_ps,
            'plus_sigma': plus_sigma,
            'minus_sigma': minus_sigma,
            'median': median,
            'median_plus_sigma': median_plus_sigma,
            'median_minus_sigma': median_minus_sigma,
            'mode': mode_value,
            'mode_plus_sigma': mode_plus_sigma,
            'mode_minus_sigma': mode_minus_sigma,
            'mean': mean_ps,
            'std': std_ps
        }
        
    def plot_histogram(self, pattern_speed_samples: np.ndarray, 
                      bestbet_ps: float, plus_sigma: float, minus_sigma: float,
                      median: float, median_plus_sigma: float, median_minus_sigma: float,
                      mode_value: float, mode_plus_sigma: float, mode_minus_sigma: float):
        """
        Plot a histogram of pattern speed samples with statistics.
        
        Args:
            pattern_speed_samples: Array of pattern speed samples
            bestbet_ps: Best pattern speed value
            plus_sigma: Upper uncertainty on bestbet value
            minus_sigma: Lower uncertainty on bestbet value
            median: Median pattern speed
            median_plus_sigma: Upper uncertainty on median
            median_minus_sigma: Lower uncertainty on median
            mode_value: Mode pattern speed
            mode_plus_sigma: Upper uncertainty on mode
            mode_minus_sigma: Lower uncertainty on mode
        """
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        ax.hist(pattern_speed_samples, bins=50, histtype='step')
        
        # Plot vertical lines for different statistics
        ax.axvline(x=bestbet_ps, color='r', alpha=0.5, linestyle='--', 
                  label=fr'Bestbet $\Omega_p$ = {bestbet_ps:.2f}$^{{+{plus_sigma:.2f}}}_{{{-minus_sigma:.2f}}}$ deg per $GMc^{{-3}}$')
        #ax.axvline(x=median, color='g', alpha=0.5, linestyle='--', 
        #          label=fr'Median $\Omega_p$ = {median:.2f}$^{{+{median_plus_sigma:.2f}}}_{{{-median_minus_sigma:.2f}}}$ deg per $GMc^{{-3}}$')
        #ax.axvline(x=mode_value, color='k', alpha=0.5, linestyle='--', 
        #          label=fr'Mode $\Omega_p$ = {mode_value:.2f}$^{{+{mode_plus_sigma:.2f}}}_{{{-mode_minus_sigma:.2f}}}$ deg per $GMc^{{-3}}$')
        
        ax.set_xlabel(r'$\Omega_p$ [deg per $GMc^{-3}$]')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=10, loc='best')
        plt.savefig(os.path.join(self.output_dir, 'Figure3_Pattern_Speed_Histogram.png'), 
                   bbox_inches='tight')
        plt.close(fig)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Calculate pattern speed and uncertainty using Monte Carlo methods')
    
    parser.add_argument('file_path', type=str, help='Path to the HDF5 data file')
    parser.add_argument('output_dir', type=str, default='', help='Directory to save output files')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of Monte Carlo samples')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    output_dir = args.output_dir + '_cylinder_output/'

    if output_dir.endswith('_cylinder_output/'):
        pass
    else:
        output_dir = output_dir + '_cylinder_output/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file  =output_dir+ "ring_parameters.csv"
    # Run ring fitting with custom parameters
    results = fit_ring_from_file(
        args.file_path, 
        output_file,
        fov=200 * eh.RADPERUAS,
        npix=200,
        min_radius=10,
        max_radius=100
    )
    
    # Initialize the analysis class
    cylinder = CylinderAnalysis(
        file_path=args.file_path,
        ring_file_path=output_file,
        output_dir=output_dir
    )
    
    # Run the Monte Carlo analysis
    results = cylinder.run_monte_carlo(n_samples=args.n_samples, save_plot=True)
    
    # Print results to console
    print("\nPattern Speed Analysis Results:")
    print("-------------------------------")
    print(f"Best Estimate: {results['bestbet_ps']:.2f}+{results['plus_sigma']:.2f}-{results['minus_sigma']:.2f} deg per GMc^-3")
    print(f"Median: {results['median']:.2f}+{results['median_plus_sigma']:.2f}-{results['median_minus_sigma']:.2f} deg per GMc^-3")
    print(f"Mode: {results['mode']:.2f}+{results['mode_plus_sigma']:.2f}-{results['mode_minus_sigma']:.2f} deg per GMc^-3")
    print("\nOutput files saved to:", cylinder.output_dir)