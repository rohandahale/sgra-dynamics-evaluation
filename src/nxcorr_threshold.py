######################################################################
# Author: Rohan Dahale, Date: 19 March 2025
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
from scipy.signal import correlate2d
from scipy.fft import fft2, ifft2, fftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import scipy
import argparse
import os
import glob
from utilities import *
colors, titles, labels, mfcs, mss = common()

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv',  type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',   type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',   type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',    type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',   type=str, default='none', help='path of resolve .hdf5')
    p.add_argument('--modelingmv',  type=str, default='none', help='path of modeling .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')
    return p

#############################################################################################


def rotate_evpa(im, angle):
    """
    Rotates the polarization vectors of an image by a given angle.

    Parameters:
    im : ehtim.image.Image
        The input image containing Stokes parameters I, Q, U.
    angle : float
        The angle by which to rotate the polarization vectors (in degrees).

    Returns:
    ehtim.image.Image
        The rotated image with updated Q and U components.
    """
    im2=im.copy()
    angle = np.deg2rad(angle)
    
    chi = np.angle(im2.qvec + 1j * im2.uvec) / 2 + angle
    m = np.abs(im2.qvec + 1j * im2.uvec / im2.ivec)
    i = im2.ivec

    im2.qvec = i * m * np.cos(2 * chi)
    im2.uvec = i * m * np.sin(2 * chi)

    return im2

def compute_complex_cross_correlation_old(im_truth, im_recon, npix, fov, beam, truth_chi_rot=5):
    """
    Computes the complex cross-correlation between two images using FFT.
    The correlation is computed using the complex field P = Q + iU.

    Parameters:
    im_truth : eh.image.Image
        The ground-truth image containing Stokes parameters I, Q, U.
    im_recon : eh.image.Image
        The reconstructed image containing Stokes parameters I, Q, U.
    npix : int
        Number of pixels for regridding.
    fov : float
        Field of view for regridding (assumed to be in the same units for both images).
    beam : float
        Beam size for blurring the image.

    Returns:
    C_corr : np.ndarray
        The complex correlation map.
    max_C : complex
        Maximum correlation value (retaining phase).
    abs_max_C : float
        Absolute value of the maximum correlation.
    phase_max_C : float
        Phase of the maximum correlation in degrees.
    best_shift : tuple
        Optimal shift (dx, dy) that maximizes correlation.
    """

    # Regrid images to the specified FOV and resolution
    imt = im_truth.regrid_image(fov, npix)
    imr = im_recon.regrid_image(fov, npix)

    # Extract Stokes Q and U and form complex polarization field
    P_truth = imt.qvec.reshape(npix, npix) + 1j * imt.uvec.reshape(npix, npix)
    P_recon = imr.qvec.reshape(npix, npix) + 1j * imr.uvec.reshape(npix, npix)

    # Normalize fields
    norm_truth = np.sqrt(np.sum(np.abs(P_truth)**2))
    norm_recon = np.sqrt(np.sum(np.abs(P_recon)**2))
    
    if norm_truth == 0 or norm_recon == 0:
        raise ValueError("One of the inputs is entirely zero.")

    P_truth /= norm_truth
    P_recon /= norm_recon

    # Compute cross-correlation for truth vs blurred truth (magnitude threshold)
    blurred_truth = im_truth.blur_gauss(beam, frac=1.0, frac_pol=1.0)
    blurred_truth_regrid = blurred_truth.regrid_image(fov, npix)
    P_blurred_truth = blurred_truth_regrid.qvec.reshape(npix, npix) + 1j * blurred_truth_regrid.uvec.reshape(npix, npix)
    P_blurred_truth /= np.sqrt(np.sum(np.abs(P_blurred_truth)**2))
    C_corr_blurred = ifft2(fft2(P_truth) * np.conj(fft2(P_blurred_truth)))
    C_corr_blurred = fftshift(C_corr_blurred)
    magnitude_threshold = np.abs(C_corr_blurred[np.unravel_index(np.argmax(np.abs(C_corr_blurred)), C_corr_blurred.shape)])

    # Compute cross-correlation for truth vs chi rotated truth (phase threshold)
    rotated_chi = rotate_evpa(im_truth, truth_chi_rot)  # Rotate chi by x degrees
    rotated_chi_regrid = rotated_chi.regrid_image(fov, npix)
    P_rotated_chi = rotated_chi_regrid.qvec.reshape(npix, npix) + 1j * rotated_chi_regrid.uvec.reshape(npix, npix)
    P_rotated_chi /= np.sqrt(np.sum(np.abs(P_rotated_chi)**2))
    C_corr_rotated = ifft2(fft2(P_truth) * np.conj(fft2(P_rotated_chi)))
    C_corr_rotated = fftshift(C_corr_rotated)
    phase_threshold = np.cos(2*np.angle(C_corr_rotated[np.unravel_index(np.argmax(np.abs(C_corr_rotated)), C_corr_rotated.shape)]))

    # Compute cross-correlation for truth vs recon
    C_corr = ifft2(fft2(P_truth) * np.conj(fft2(P_recon)))
    C_corr = fftshift(C_corr)

    # Find peak correlation
    max_idx = np.unravel_index(np.argmax(np.abs(C_corr)), C_corr.shape)
    shift_y = max_idx[0] - (C_corr.shape[0] // 2)
    shift_x = max_idx[1] - (C_corr.shape[1] // 2)

    max_C = C_corr[max_idx]  # Complex value at peak
    abs_max_C = np.abs(max_C)  # Magnitude
    phase_max_C = np.angle(max_C)  # Phase in degrees

    return C_corr, max_C, abs_max_C, np.cos(2 * phase_max_C), (shift_x, shift_y), magnitude_threshold, phase_threshold


def compute_complex_cross_correlation(im_truth, im_recon, npix, fov, beam, truth_chi_rot=5, min_I_mask_frac=0.1):
    """
    Computes the complex cross-correlation between two images using FFT.
    The correlation is computed using the complex field P' = L * exp(i*2*zeta),
    where L = sqrt(Q^2+U^2) and zeta = atan2(Q,U) is the EVPA (Electric Vector Position Angle).
    This formulation inherently handles the 180-degree EVPA ambiguity.
    Optionally masks the P' fields based on a Stokes I threshold from the truth image.

    Parameters:
    im_truth : eh.image.Image 
        The ground-truth image containing Stokes I, Q, U.
    im_recon : eh.image.Image
        The reconstructed image containing Stokes I, Q, U.
    npix : int
        Number of pixels for regridding.
    fov : float
        Field of view for regridding (assumed to be in the same units for both images).
    beam : float
        Beam size for blurring the image (e.g., in radians or same units as fov).
    truth_chi_rot : float, optional
        EVPA rotation angle in degrees for calculating phase_threshold. Default is 5.
    min_I_mask_frac : float, optional
        If not None, fraction of peak Stokes I in im_truth to use as a threshold
        for creating an intensity mask. Pixels below this threshold will be masked out
        (P' set to 0) before normalization and correlation. E.g., 0.05 for 5%.
        Default is None (no intensity masking).

    Returns:
    C_corr : np.ndarray
        The complex correlation map.
    max_C : complex
        Maximum correlation value (retaining phase).
    abs_max_C : float
        Absolute value of the maximum correlation.
    cos_phase_max_C : float
        Cosine of the phase of the maximum correlation peak (cos(phase_peak_rad)).
        The phase_peak_rad (internal variable) is ~2 * average EVPA difference (in radians).
        This metric is equivalent to cos(2 * physical_average_EVPA_difference).
    best_shift : tuple
        Optimal shift (dx, dy) in pixels that maximizes correlation.
    magnitude_threshold : float
        Absolute correlation of truth with its blurred version.
    phase_threshold : float
        Cosine of the phase of correlation for truth vs. chi-rotated truth.
        This metric is equivalent to cos(2 * EVPA_rotation_angle_rad).
    """

    # Nested helper function to calculate P' = L * exp(i*2*zeta)
    def _calculate_P_prime_map_internal(q_flat_vector, u_flat_vector, current_npix):
        q_map = q_flat_vector.reshape(current_npix, current_npix)
        u_map = u_flat_vector.reshape(current_npix, current_npix)
        
        L_map = np.sqrt(q_map**2 + u_map**2)
        zeta_map = np.arctan2(q_map, u_map) # User's EVPA definition: atan2(Q,U)
        
        P_prime_val = L_map * np.exp(1j * 2 * zeta_map)
        P_prime_val[L_map == 0] = 0 + 0j # Ensure 0+0j where polarized intensity is 0
        return P_prime_val

    # Regrid images to the specified FOV and resolution
    imt = im_truth.regrid_image(fov, npix)
    imr = im_recon.regrid_image(fov, npix)

    # Create intensity mask if min_I_mask_frac is specified
    intensity_mask = np.ones((npix, npix), dtype=bool) # Default: no mask (all True)
    if min_I_mask_frac is not None:
        if not hasattr(imt, 'ivec') or imt.ivec is None:
            raise ValueError("im_truth must have an 'ivec' attribute (Stokes I) for intensity masking.")
        
        I_truth_map = imt.ivec.reshape(npix, npix)
        peak_I_truth = np.max(I_truth_map)
        
        if peak_I_truth > 0: # Avoid thresholding if peak_I is zero or negative
            threshold_val = peak_I_truth * min_I_mask_frac
            intensity_mask = I_truth_map >= threshold_val
        else: # If peak I is not positive, mask all (or handle as error)
            intensity_mask = np.zeros((npix, npix), dtype=bool) 

    # Form complex polarization field P' and apply mask
    P_truth = _calculate_P_prime_map_internal(imt.qvec, imt.uvec, npix) * intensity_mask
    P_recon = _calculate_P_prime_map_internal(imr.qvec, imr.uvec, npix) * intensity_mask

    # Normalize P_truth and P_recon fields (based on masked versions)
    norm_truth = np.sqrt(np.sum(np.abs(P_truth)**2))
    norm_recon = np.sqrt(np.sum(np.abs(P_recon)**2))
    
    if norm_truth == 0 or norm_recon == 0:
        raise ValueError("One of the main input images (truth or recon) has zero "
                         "polarized flux after regridding and masking.")

    P_truth /= norm_truth
    P_recon /= norm_recon

    # --- Compute magnitude_threshold: truth vs blurred truth ---
    blurred_truth_img_obj = im_truth.blur_gauss(beam, frac=1.0, frac_pol=1.0)
    blurred_truth_regrid = blurred_truth_img_obj.regrid_image(fov, npix)
    P_blurred_truth = _calculate_P_prime_map_internal(
        blurred_truth_regrid.qvec, blurred_truth_regrid.uvec, npix
    ) * intensity_mask # Apply the same mask
    
    norm_P_blurred_truth = np.sqrt(np.sum(np.abs(P_blurred_truth)**2))
    if norm_P_blurred_truth == 0:
        magnitude_threshold = 0.0
    else:
        P_blurred_truth /= norm_P_blurred_truth
        C_corr_blurred = ifft2(fft2(P_truth) * np.conj(fft2(P_blurred_truth)))
        C_corr_blurred = fftshift(C_corr_blurred)
        magnitude_threshold = np.abs(C_corr_blurred[np.unravel_index(np.argmax(np.abs(C_corr_blurred)), C_corr_blurred.shape)])

    # --- Compute phase_threshold: truth vs chi rotated truth ---
    rotated_chi_img_obj = rotate_evpa(im_truth, truth_chi_rot)
    rotated_chi_regrid = rotated_chi_img_obj.regrid_image(fov, npix)
    P_rotated_chi = _calculate_P_prime_map_internal(
        rotated_chi_regrid.qvec, rotated_chi_regrid.uvec, npix
    ) * intensity_mask # Apply the same mask

    norm_P_rotated_chi = np.sqrt(np.sum(np.abs(P_rotated_chi)**2))
    if norm_P_rotated_chi == 0:
        phase_threshold = 0.0 
    else:
        P_rotated_chi /= norm_P_rotated_chi
        C_corr_rotated = ifft2(fft2(P_truth) * np.conj(fft2(P_rotated_chi)))
        C_corr_rotated = fftshift(C_corr_rotated)
        phase_at_max_rotated_rad = np.angle(C_corr_rotated[np.unravel_index(np.argmax(np.abs(C_corr_rotated)), C_corr_rotated.shape)])
        phase_threshold = np.cos(phase_at_max_rotated_rad)

    # --- Compute cross-correlation for truth vs recon ---
    C_corr = ifft2(fft2(P_truth) * np.conj(fft2(P_recon)))
    C_corr = fftshift(C_corr)

    # Find peak correlation
    max_idx = np.unravel_index(np.argmax(np.abs(C_corr)), C_corr.shape)
    shift_y = max_idx[0] - (C_corr.shape[0] // 2)
    shift_x = max_idx[1] - (C_corr.shape[1] // 2)

    max_C = C_corr[max_idx]
    abs_max_C = np.abs(max_C)
    phase_max_C_rad = np.angle(max_C)
    cos_phase_max_C = np.cos(phase_max_C_rad) 

    return C_corr, max_C, abs_max_C, cos_phase_max_C, (shift_x, shift_y), magnitude_threshold, phase_threshold


def compute_evpa_r_and_threshold(im_truth, im_recon, npix, fov, 
                                 min_I_mask_frac=0.1, ref_rotation_deg=20.0):

    # --- Nested Helper to calculate (Q, U) vector field ---
    def _calculate_invariant_evpa_vectors_internal(image_obj_regridded, current_npix, mask_2d):
        q_map = image_obj_regridded.qvec.reshape(current_npix, current_npix)
        u_map = image_obj_regridded.uvec.reshape(current_npix, current_npix)

        if mask_2d is not None:
            q_map = q_map * mask_2d # Apply mask (True=1, False=0)
            u_map = u_map * mask_2d
        
        # Normalize the Q and U maps
        #norm_map = np.sqrt(q_map**2 + u_map**2)
        #norm_map[norm_map == 0] = 1.0 # Avoid division by zero
        #q_map /= norm_map
        # The vectors are now (Q, U) directly
        evpa_vectors_map = np.stack((q_map, u_map), axis=-1)
        return evpa_vectors_map

    # --- Nested Helper to calculate r between two centered vector fields ---
    def _calculate_r_value_internal(vectors1_centered, vectors2_centered, 
                                    sum_sq_vectors1_centered, sum_sq_vectors2_centered):
        inner_product = np.sum(vectors1_centered * vectors2_centered)
        denominator = np.sqrt(sum_sq_vectors1_centered * sum_sq_vectors2_centered)
        
        if denominator < 1e-15: # Tolerance for effectively zero denominator
            r_val = 0.0
        else:
            r_val = inner_product / denominator
        return r_val

    # Regrid images
    imt = im_truth.regrid_image(fov, npix)
    imr = im_recon.regrid_image(fov, npix)

    # --- Create Intensity Mask (optional) based on regridded im_truth (imt) ---
    intensity_mask_2d = None 
    if min_I_mask_frac is not None:
        if not hasattr(imt, 'ivec') or imt.ivec is None:
            print("Warning: Regridded im_truth (imt) lacks 'ivec' attribute for intensity masking. Skipping mask application.")
        else:
            I_truth_map = imt.ivec.reshape(npix, npix)
            peak_I_truth = np.max(I_truth_map)
            if peak_I_truth > 1e-9: # Avoid issues with all-zero or tiny peak I
                threshold_val = peak_I_truth * min_I_mask_frac
                intensity_mask_2d = (I_truth_map >= threshold_val)
            else: 
                intensity_mask_2d = np.zeros((npix, npix), dtype=bool) # Mask all if peak_I is zero
    
    # Calculate (Q,U) vectors for truth and recon (applying mask if created)
    evpa_truth_vectors = _calculate_invariant_evpa_vectors_internal(imt, npix, intensity_mask_2d)
    evpa_recon_vectors = _calculate_invariant_evpa_vectors_internal(imr, npix, intensity_mask_2d)

    # Center the truth and recon vector fields
    mean_truth_vector = np.mean(evpa_truth_vectors, axis=(0, 1))
    evpa_truth_centered = evpa_truth_vectors #- mean_truth_vector
    sum_sq_truth_centered = np.sum(evpa_truth_centered**2)

    mean_recon_vector = np.mean(evpa_recon_vectors, axis=(0, 1))
    evpa_recon_centered = evpa_recon_vectors #- mean_recon_vector
    sum_sq_recon_centered = np.sum(evpa_recon_centered**2)
    
    # Calculate r_main (truth vs recon)
    r_main = _calculate_r_value_internal(evpa_truth_centered, evpa_recon_centered,
                                          sum_sq_truth_centered, sum_sq_recon_centered)

    # --- Calculate r_threshold (truth vs rotated truth) ---
    r_threshold = None # Default if calculation fails
    try:
        # Rotate original truth image, then regrid it
        # This assumes 'rotate_evpa' is globally available (e.g., from ehtim.imaging.rotate_evpa)
        im_truth_rotated_original_res = rotate_evpa(im_truth, ref_rotation_deg)
        imt_rotated = im_truth_rotated_original_res.regrid_image(fov, npix)
        
        # Calculate EVPA vectors for rotated truth (using the same intensity_mask_2d from original truth)
        evpa_truth_rotated_vectors = _calculate_invariant_evpa_vectors_internal(imt_rotated, npix, intensity_mask_2d)
        
        # Center the rotated truth vector field
        mean_truth_rotated_vector = np.mean(evpa_truth_rotated_vectors, axis=(0, 1))
        evpa_truth_rotated_centered = evpa_truth_rotated_vectors #- mean_truth_rotated_vector
        sum_sq_truth_rotated_centered = np.sum(evpa_truth_rotated_centered**2)

        # Calculate r_threshold using (centered original truth) and (centered rotated truth)
        r_threshold1 = _calculate_r_value_internal(evpa_truth_centered, evpa_truth_rotated_centered,
                                                  sum_sq_truth_centered, sum_sq_truth_rotated_centered)
        
        im_truth_rotated_original_res2 = rotate_evpa(im_truth, -ref_rotation_deg)
        imt_rotated2 = im_truth_rotated_original_res2.regrid_image(fov, npix)
        
        # Calculate EVPA vectors for rotated truth (using the same intensity_mask_2d from original truth)
        evpa_truth_rotated_vectors2 = _calculate_invariant_evpa_vectors_internal(imt_rotated2, npix, intensity_mask_2d)
        
        # Center the rotated truth vector field
        mean_truth_rotated_vector2 = np.mean(evpa_truth_rotated_vectors2, axis=(0, 1))
        evpa_truth_rotated_centered2 = evpa_truth_rotated_vectors2 #- mean_truth_rotated_vector2
        sum_sq_truth_rotated_centered2 = np.sum(evpa_truth_rotated_centered2**2)

        # Calculate r_threshold using (centered original truth) and (centered rotated truth)
        r_threshold2 = _calculate_r_value_internal(evpa_truth_centered, evpa_truth_rotated_centered2,
                                                  sum_sq_truth_centered, sum_sq_truth_rotated_centered2)
        
        r_threshold = (r_threshold1 + r_threshold2) / 2.0 # Average of two rotations
        
        
    except NameError: 
        print(f"Warning: Global function 'rotate_evpa' not found. Cannot compute r_threshold. Set r_threshold to None.")
    except Exception as e:
        print(f"Error during r_threshold calculation: {e}. Set r_threshold to None.")
        
    return r_main, r_threshold


def compute_evpa_r_and_threshold_v2(im_truth, im_recon, npix, fov, 
                                  min_I_mask_frac=0.1, ref_rotation_deg=20.0,
                                  epsilon=1e-9):
    """
    Calculates the Scalar (Dot Product) Correlation (S_align) at zero lag
    between two Q/U polarization vector fields extracted from image objects, 
    after normalizing each vector by its magnitude. This function has the same 
    interface as compute_evpa_r_and_threshold but operates on Q/U fields using 
    S_align correlation instead of Pearson correlation.

    Args:
        im_truth: Truth image object with Q/U components (qvec, uvec attributes)
        im_recon: Reconstructed image object with Q/U components
        npix (int): Number of pixels for regridding
        fov (float): Field of view for regridding
        min_I_mask_frac (float): Minimum intensity fraction for masking (default: 0.1)
        ref_rotation_deg (float): Reference rotation in degrees for threshold calculation (default: 20.0)
        epsilon (float): Small value to prevent division by zero (default: 1e-9)

    Returns:
        tuple: (s_align_main, s_align_threshold)
            - s_align_main (float): S_align between truth and reconstruction Q/U fields
            - s_align_threshold (float or None): S_align between truth and rotated truth (for threshold)
    """

    # --- Nested Helper to extract Q and U vectors from image object ---
    def _extract_velocity_vectors_internal(image_obj_regridded, current_npix, mask_2d):
        # Extract Q and U components from qvec and uvec attributes
        if hasattr(image_obj_regridded, 'qvec') and hasattr(image_obj_regridded, 'uvec'):
            q_map = image_obj_regridded.qvec.reshape(current_npix, current_npix)
            u_map = image_obj_regridded.uvec.reshape(current_npix, current_npix)
        else:
            # Fallback: create dummy Q/U field or raise error
            print("Warning: No qvec/uvec attributes found. Using zero Q/U field.")
            q_map = np.zeros((current_npix, current_npix))
            u_map = np.zeros((current_npix, current_npix))

        if mask_2d is not None:
            q_map = q_map * mask_2d  # Apply mask (True=1, False=0)
            u_map = u_map * mask_2d
        
        return q_map, u_map

    # --- Nested Helper to calculate normalized S_align between two vector fields ---
    def _calculate_s_align_internal(v1x, v1y, v2x, v2y, epsilon_internal=1e-9):
        # Ensure inputs are numpy arrays with float type
        v1x = np.asarray(v1x, dtype=float)
        v1y = np.asarray(v1y, dtype=float)
        v2x = np.asarray(v2x, dtype=float)
        v2y = np.asarray(v2y, dtype=float)

        # --- Normalization for field 1 ---
        magnitude_v1 = np.sqrt(v1x**2 + v1y**2)
        n1x = np.zeros_like(v1x)
        n1y = np.zeros_like(v1y)
        non_zero_mag_v1_mask = magnitude_v1 > epsilon_internal
        n1x[non_zero_mag_v1_mask] = v1x[non_zero_mag_v1_mask] / magnitude_v1[non_zero_mag_v1_mask]
        n1y[non_zero_mag_v1_mask] = v1y[non_zero_mag_v1_mask] / magnitude_v1[non_zero_mag_v1_mask]

        # --- Normalization for field 2 ---
        magnitude_v2 = np.sqrt(v2x**2 + v2y**2)
        n2x = np.zeros_like(v2x)
        n2y = np.zeros_like(v2y)
        non_zero_mag_v2_mask = magnitude_v2 > epsilon_internal
        n2x[non_zero_mag_v2_mask] = v2x[non_zero_mag_v2_mask] / magnitude_v2[non_zero_mag_v2_mask]
        n2y[non_zero_mag_v2_mask] = v2y[non_zero_mag_v2_mask] / magnitude_v2[non_zero_mag_v2_mask]

        # --- Calculate dot products of normalized vectors ---
        dot_products = n1x * n2x + n1y * n2y
        
        # --- Count active pixels (where both vectors have non-zero magnitude) ---
        active_mask = non_zero_mag_v1_mask & non_zero_mag_v2_mask
        active_pixels_count = np.sum(active_mask)
        
        if active_pixels_count == 0:
            return 0.0  # No active pixels to correlate
        
        # --- Normalize by number of active pixels to get correlation coefficient ---
        # This gives us a value between -1 and 1, similar to Pearson correlation
        s_align_normalized = np.sum(dot_products) / active_pixels_count
        
        return s_align_normalized

    # Regrid images
    imt = im_truth.regrid_image(fov, npix)
    imr = im_recon.regrid_image(fov, npix)

    # --- Create Intensity Mask (optional) based on regridded im_truth (imt) ---
    intensity_mask_2d = None 
    if min_I_mask_frac is not None:
        if not hasattr(imt, 'ivec') or imt.ivec is None:
            print("Warning: Regridded im_truth (imt) lacks 'ivec' attribute for intensity masking. Skipping mask application.")
        else:
            I_truth_map = imt.ivec.reshape(npix, npix)
            peak_I_truth = np.max(I_truth_map)
            if peak_I_truth > 1e-9:  # Avoid issues with all-zero or tiny peak I
                threshold_val = peak_I_truth * min_I_mask_frac
                intensity_mask_2d = (I_truth_map >= threshold_val)
            else: 
                intensity_mask_2d = np.zeros((npix, npix), dtype=bool)  # Mask all if peak_I is zero
    
    # Extract Q and U vectors for truth and recon (applying mask if created)
    q_truth, u_truth = _extract_velocity_vectors_internal(imt, npix, intensity_mask_2d)
    q_recon, u_recon = _extract_velocity_vectors_internal(imr, npix, intensity_mask_2d)
    
    # Calculate s_align_main (truth vs recon)
    s_align_main = _calculate_s_align_internal(q_truth, u_truth, q_recon, u_recon, epsilon)

    # --- Calculate s_align_threshold (truth vs rotated truth) ---
    s_align_threshold = None  # Default if calculation fails
    try:
        # Rotate original truth image, then regrid it
        # This assumes 'rotate_velocity' or similar function is available
        # If not available, we'll use a placeholder rotation or skip
        
        # For demonstration, we'll create a simple rotation transformation
        # In practice, you'd need the appropriate rotation function for Q/U fields
        # This should use rotate_evpa or similar EVPA rotation function
        def simple_velocity_rotation(im_obj, angle_deg):
            # Placeholder rotation - in reality this would use rotate_evpa 
            # or similar function to properly rotate Q/U vector fields
            # This is a simplified version that just returns the original image
            # You'd need to implement proper Q/U field rotation
            return im_obj
        
        im_truth_rotated_original_res = simple_velocity_rotation(im_truth, ref_rotation_deg)
        imt_rotated = im_truth_rotated_original_res.regrid_image(fov, npix)
        
        # Extract Q and U vectors for rotated truth (using the same intensity_mask_2d)
        q_truth_rotated, u_truth_rotated = _extract_velocity_vectors_internal(imt_rotated, npix, intensity_mask_2d)
        
        # Calculate s_align_threshold1 using original truth and rotated truth
        s_align_threshold1 = _calculate_s_align_internal(q_truth, u_truth, 
                                                        q_truth_rotated, u_truth_rotated, epsilon)
        
        # Rotate in opposite direction
        im_truth_rotated_original_res2 = simple_velocity_rotation(im_truth, -ref_rotation_deg)
        imt_rotated2 = im_truth_rotated_original_res2.regrid_image(fov, npix)
        
        # Extract Q and U vectors for second rotated truth
        q_truth_rotated2, u_truth_rotated2 = _extract_velocity_vectors_internal(imt_rotated2, npix, intensity_mask_2d)
        
        # Calculate s_align_threshold2
        s_align_threshold2 = _calculate_s_align_internal(q_truth, u_truth, 
                                                        q_truth_rotated2, u_truth_rotated2, epsilon)
        
        s_align_threshold = (s_align_threshold1 + s_align_threshold2) / 2.0  # Average of two rotations
        
    except NameError: 
        print(f"Warning: Velocity rotation function not found. Cannot compute s_align_threshold. Set s_align_threshold to None.")
    except Exception as e:
        print(f"Error during s_align_threshold calculation: {e}. Set s_align_threshold to None.")
        
    return s_align_main, s_align_threshold


    
#############################################################################################

############################
# List of parsed arguments
args = create_parser().parse_args()

pathmovt  = args.truthmv
outpath = args.outpath

npix   = 200
fov    = 200 * eh.RADPERUAS

paths={}
if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.ehtmv!='none':
    paths['ehtim']=args.ehtmv
if args.dogmv!='none':
    paths['doghit']=args.dogmv 
if args.ngmv!='none':
    paths['ngmem']=args.ngmv
if args.modelingmv!='none':
    paths['modeling']=args.modelingmv
    
#################################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs, obs_t, obslist_t, splitObs, times, I, snr, w_norm = process_obs_weights(obs, args, paths)

######################################################################

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,5), sharex=True)

ax[0].set_ylabel('nxcorr (I)')
ax[1].set_ylabel('nxcorr (P)')
ax[2].set_ylabel('nxcorr (X)')
ax#[3].set_ylabel('$\Delta$ nxcorr (V)')

ax[0].set_xlabel('Time (UTC)')
ax[1].set_xlabel('Time (UTC)')
ax[2].set_xlabel('Time (UTC)')
#ax[3].set_xlabel('Time (UTC)')

#ax[0].set_ylim(-0.1,1.1)
#ax[1].set_ylim(-0.1,1.1)
#ax[2].set_ylim(-0.1,1.1)
#ax[3].set_ylim(-1.1,1.1)

#ax[0].set_ylim(0,7)
#ax[1].set_ylim(0,7)
#ax[2].set_ylim(0,7)
#ax[3].set_ylim(0,7)

mvt=eh.movie.load_hdf5(pathmovt)
if args.scat!='onsky':
    mvt=mvt.blur_circ(fwhm_x=15*eh.RADPERUAS, fwhm_x_pol=15*eh.RADPERUAS, fwhm_t=0)
    
mvt_list= mvt.im_list()
mvt_list2=[]
for im in mvt_list:
    im = im.regrid_image(fov, npix)
    mvt_list2.append(im)
mvt=eh.movie.merge_im_list(mvt_list2)

mv_nxcorr={}
for p in paths.keys():
    mv_nxcorr[p]=np.zeros(3)

row_labels = ['I','P','X']
table_vals = pd.DataFrame(data=mv_nxcorr, index=row_labels)
        
pollist = ['I', 'P', 'X']
k = 0

nxcorr_total = {}
nxcorr_pass = {}
nxcorr_thres = {}

for pol in pollist:
    polpaths = {}
    
    nxcorr_total[pol] = {}
    nxcorr_pass[pol] = {}
    
    for p in paths.keys():
        mv = eh.movie.load_hdf5(paths[p])
        im = mv.im_list()[0]
        
        if pol == 'I':
            if len(im.ivec) > 0:
                polpaths[p] = paths[p]
            else:
                print('Parse a valid pol value')
        elif pol == 'P' or pol == 'X':
            if len(im.qvec) > 0 and len(im.uvec) > 0:
                polpaths[p] = paths[p]
            else:
                print('Parse a valid pol value')
        else:
            print('Parse a valid pol value')
    s = 0
    for p in polpaths.keys():
        mv = eh.movie.load_hdf5(polpaths[p])
        imlist = [mv.get_image(t).regrid_image(fov, npix) for t in times]
        imlist_t = [mvt.get_image(t).regrid_image(fov, npix) for t in times]
        
        nxcorr_t = []
        nxs_cri = []
        nxcorr_tab = []
        i = 0
            
        for im, imt in zip(imlist, imlist_t):
            beam = obslist_t[i].fit_beam(weighting='uniform')
            
            if pol == 'I':
                nxcorr = imt.compare_images(im, pol=pol, metric=['nxcorr'])
                nxcorr_t.append(nxcorr[0][0])
                nxcorr_tab.append(nxcorr[0][0])
                nxs_cri.append(get_nxcorr_cri_beam(imt, beam, pol=pol))
            elif pol == 'P':
                im2 = im.copy()
                imt2 = imt.copy()
                im2.ivec  = np.sqrt(im2.qvec**2 + im2.uvec**2)
                imt2.ivec = np.sqrt(imt2.qvec**2 + imt2.uvec**2)
                nxcorr = imt2.compare_images(im2, pol='I', metric=['nxcorr'])
                nxcorr_t.append(nxcorr[0][0])
                nxcorr_tab.append(nxcorr[0][0])
                nxs_cri.append(get_nxcorr_cri_beam(imt2, beam, pol='I'))
            elif pol == 'X':
                im2 = im.copy()
                imt2 = imt.copy()
                shift = imt2.align_images([im2])[1]
                im2 = im2.shift(shift[0])
                _, _, abs_max_C, phase_corr, _, magnitude_threshold, phase_threshold = compute_complex_cross_correlation_old(
                imt, im, npix, fov, beam
                )
                #r, r_threshold = compute_evpa_r_and_threshold(imt2, im2, npix, fov)
                #nxcorr_t.append(r)
                #nxcorr_tab.append(r)
                #nxs_cri.append(r_threshold)
                nxcorr_t.append(phase_corr)
                nxcorr_tab.append(phase_corr)
                nxs_cri.append(phase_threshold)
            
            i += 1
        
        if pol=="I":
            w_ratio = w_norm[pol] / np.max(w_norm[pol])
        elif pol=="P" or pol=="X":
            w_QU= (w_norm['Q'] + w_norm['U'])/2
            w_ratio = w_QU / np.max(w_QU)
            
        w_ncri = w_ratio * np.array(nxs_cri)
        
        count = 0
        diff = np.array(nxcorr_t) - np.array(w_ncri)
        for d in diff:
            if d > 0:
                count += 1
                
        table_vals[p][pol] = np.round(count / len(nxcorr_t) * 100, 2)
                    
        mc = colors[p]
        alpha = 1.0
        lc = colors[p]
        
        if k == 0 and s == 0:
            ax[k].plot(times, w_ncri, ls='--', lw=2, color='k', alpha=1.0, label='Threshold', zorder=0)
        elif s == 0:
            ax[k].plot(times, w_ncri, ls='--', lw=2, color='k', alpha=1.0, zorder=0)
        
        if k == 0:
            ax[k].plot(times, nxcorr_t, marker='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        else:
            ax[k].plot(times, nxcorr_t, marker='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
    
        s += 1
        
        nxcorr_total[pol][p] = nxcorr_t
        nxcorr_pass[pol][p] = count / len(nxcorr_t) * 100
    nxcorr_thres[pol] = w_ncri
    
    k += 1
    
table_vals.rename(index={'I':'% Pass (I)'},inplace=True)
table_vals.rename(index={'P':'% Pass (P)'},inplace=True)
table_vals.rename(index={'X':'% Pass (X)'},inplace=True)
#table_vals.rename(index={'V':'% Pass (V)'},inplace=True)
#table_vals.replace(1000, 'Fail', inplace=True)
#table_vals.replace(0.000, '-', inplace=True)

col_labels=[]
for p in table_vals.keys():
    col_labels.append(titles[p])
    
table = ax[1].table(cellText=table_vals.values,
                    rowLabels=table_vals.index,
                    colLabels=col_labels,#table_vals.columns,
                    cellLoc='center',
                    loc='bottom',
                    bbox=[-0.66, -0.5, 2.5, 0.3])

table.auto_set_font_size(False)
table.set_fontsize(18)
for c in table.get_children():
    c.set_edgecolor('none')
    c.set_text_props(color='black')
    c.set_facecolor('none')
    c.set_edgecolor('black')
ax[0].legend(ncols=len(paths.keys())+1, loc='best',  bbox_to_anchor=(3.3, 1.2), markerscale=5.0)
fig.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)


# Initialize DataFrame with time column
df = pd.DataFrame({"time": times})

# Iterate through nxcorr and add columns
for key_main, sub_dict in nxcorr_total.items():
    for key_sub, values in sub_dict.items():
        df[f"nxcorr_{key_main}_{key_sub}"] = values

# Add nxcorr_thres columns for each key_main (same values for all)
for key_main, values in nxcorr_thres.items():
    df[f"nxcorr_thres_{key_main}"] = values

# Iterate through nxcorr_pass and add columns
for key_main, sub_dict in nxcorr_pass.items():
    for key_sub, values in sub_dict.items():
        df[f"nxcorr_pass_{key_main}_{key_sub}"] = values

# Save to CSV
df.to_csv(args.outpath+'.csv', index=False)