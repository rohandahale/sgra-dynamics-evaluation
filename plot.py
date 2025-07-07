# ==============================================================================
# Imports
# ==============================================================================
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import ehtim as eh
import ehtplot.color  # Assuming this is a custom module or installed package

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# --- Pipeline Selection ---
# Choose the processing pipeline ('kine', 'another_pipe', etc.)
PIPELINE_NAME = 'kine'

# --- Paths ---
BASE_RESULTS_PATH = './results/'
BASE_EVALUATION_PATH = '/home/share/SgrA_Dynamics/evaluation/april11/20250409_results_kine_april11_besttimes/'
BASE_FIGURES_PATH = '/home/share/SgrA_Dynamics/evaluation/april11/20250409_results_kine_april11_besttimes/figures_test/'
os.makedirs(BASE_FIGURES_PATH, exist_ok=True) # Ensure figure directory exists

# --- Data File Patterns ---
# Using f-strings for easier path construction later
CALIBRATOR_FILE = os.path.join(BASE_RESULTS_PATH, 'J1924_movrec15000_ireg.hdf5')
STATIC_NXCORR_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, '{mod}_LO_onsky_nxcorr_static_threshold.csv')
STATIC_TRUTH_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, 'static+dynamic/{mod}_LO_onsky_truth.hdf5')
STATIC_RECON_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, 'static+dynamic/{mod}_LO_onsky_{pipe}.hdf5')
DYNAMIC_NXCORR_DYN_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, '{mod}_LO_onsky_nxcorr_dynamic_threshold.csv')
DYNAMIC_NXCORR_THRESH_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, '{mod}_LO_onsky_nxcorr_threshold.csv')
DYNAMIC_TRUTH_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, 'static+dynamic/{mod}_LO_onsky_truth.hdf5')
DYNAMIC_RECON_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, 'static+dynamic/{mod}_LO_onsky_{pipe}.hdf5')
DYNAMIC_COMP_TRUTH_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, 'dynamic/{mod}_LO_onsky_truth.hdf5')
DYNAMIC_COMP_RECON_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, 'dynamic/{mod}_LO_onsky_{pipe}.hdf5')
DYNAMIC_VIDA_ALL_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, '{mod}_LO_onsky_dynamic_vida_all.csv')
DYNAMIC_VIDA_TRUTH_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, '{mod}_LO_onsky_dynamic_vida_truth.csv')
DYNAMIC_VIDA_POL_ALL_FILE_TPL = os.path.join(BASE_EVALUATION_PATH, '{mod}_LO_onsky_vida_pol_all.csv')
PATTERN_SPEED_TRUTH_DIR_TPL = os.path.join(BASE_EVALUATION_PATH, 'patternspeed/{mod}_LO_onsky_truth_cylinder_output/')
PATTERN_SPEED_RECON_DIR_TPL = os.path.join(BASE_EVALUATION_PATH, 'patternspeed/{mod}_LO_onsky_{pipe}_cylinder_output/')

# --- Model Lists ---
STATIC_MODELS = ['crescent', 'disk', 'double', 'edisk', 'point', 'ring']
DYNAMIC_MODELS = [
    'mring+hsCW', 'mring+hsCCW', 'mring+hs-cross', 'mring+hs-not-center',
    'mring+hs-incoh', 'mring+hs-pol', 'mring-varbeta2', 'mring+hsCW0.15',
    'mring+hsCW0.60', 'mring+hsCW1.20', 'mring+hsCW20', 'mring+hsCW40'
]
DYNAMIC_MODELS_LADDER = DYNAMIC_MODELS[:7]
DYNAMIC_MODELS_EXTRA = DYNAMIC_MODELS[7:]

# --- Plotting Defaults ---
plt.rcParams['image.cmap'] = 'afmhot'
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams['image.interpolation'] = 'bicubic'

# --- Plotting Parameters ---
# Using dictionaries to group related parameters
PLOT_PARAMS = {
    'calibrator': {'fsize': 18, 'ftsize': 12, 'sk': 1, 'fov_scale': 1.0},
    'static': {'fsize': 18, 'ftsize': 12, 'sk': 8, 'extent': [-60, 60, -60, 60]},
    'dynamic_print': {'fsize': 14, 'ftsize': 10, 'sk': 8, 'extent_im': [-60, 60, -60, 60], 'extent_dyn': [-60, 60, 60, -60], 'extent_ps': [-278, 278, -180, 180]},
    'dynamic_ladder': {'fbsize': 17, 'fsize': 14, 'ftsize': 9, 'sk': 8, 'extent_ps': [-278, 278, -180, 180]},
    'dynamic_extra': {'fbsize': 17, 'fsize': 14, 'ftsize': 10, 'sk': 8, 'extent_ps': [-278, 278, -180, 180]},
    'pol_quiver': {'pivot': 'mid', 'angles': 'uv', 'ec': 'w', 'width': 0.007,
                   'scale': 0.002, 'linewidth': 0, 'headwidth': 0,
                   'headlength': 0, 'headaxislength': 0},
    'pol_norm_frac_threshold': 0.3, # Threshold for plotting polarization vectors based on fractional intensity
    'pol_cmap': 'rainbow',
    'pol_cnorm_vmin': 0.0,
    'pol_cnorm_vmax': 0.8,
    'scale_bar_color': 'w',
    'dynamic_comp_cmap': 'binary',
    'dynamic_comp_contour_color': 'k',
    'dynamic_comp_contour_ls': '-',
    'dynamic_comp_contour_lw': 1,
    'nxcorr_thresh_color': 'k',
    'nxcorr_recon_color': 'dodgerblue',
    'nxcorr_recon_pol_color': 'dodgerblue',
    'nxcorr_recon_chi_color': 'tab:red',
    'vida_gt_color': 'k',
    'vida_recon_color': 'dodgerblue',
    'vida_thresh_color': 'silver',
    'autocorr_cmap': 'YlGnBu_r',
    'autocorr_vmin': -0.5,
    'autocorr_vmax': 1.0,
}

# --- Slices / Cropping ---
# Define slices using tuples or slice objects for clarity
CALIBRATOR_SLICE = (slice(20, 49), slice(31, 60))
MODEL_SLICE = (slice(40, -40), slice(40, -40)) # For static and dynamic models

# --- Dynamic Frame Selection ---
# Define the specific frame indices or times for dynamic plots
DYNAMIC_FRAME_INDICES = {
    'mring+hsCW': [0, 10, 20, 30, 40, 50],
    'mring+hsCCW': [0, 10, 20, 30, 40, 50],
    'mring+hs-cross': [0, 6, 20, 33, 38, 48],
    'mring+hs-incoh': [0, 25, 50, 70, 90, 98],
    'mring+hs-not-center': [0, 18, 36, 54, 72, 90],
    'mring+hs-pol': [0, 10, 20, 30, 40, 50],
    'mring-varbeta2': [5, 15, 25, 35, 45, 54],
    'mring+hsCW0.15': [0, 10, 20, 30, 40, 50],
    'mring+hsCW0.60': [0, 10, 20, 30, 40, 50],
    'mring+hsCW1.20': [0, 10, 20, 30, 40, 50],
    'mring+hsCW20': [20, 25, 30, 35, 38, 48],
    'mring+hsCW40': [8, 14, 19, 22, 32, 38]
}
NUM_DYNAMIC_FRAMES = 6 # Should match the length of lists in DYNAMIC_FRAME_INDICES

# ==============================================================================
# Helper Functions
# ==============================================================================

def load_movie(path):
    """Loads an ehtim movie from an HDF5 file."""
    try:
        return eh.movie.load_hdf5(path)
    except Exception as e:
        print(f"Error loading movie {path}: {e}")
        return None

def get_image_data(image, pol='I', crop_slice=None):
    """Extracts image array for a given polarization, optionally cropping it."""
    if image is None:
        return None
    data = image.imarr(pol=pol)
    if crop_slice:
        return data[crop_slice]
    return data

def calculate_polarization_vectors(imgQ, imgU):
    """Calculates polarization vector components (px, py) for quiver plot."""
    if imgQ is None or imgU is None:
        return None, None
    angle = np.angle(imgQ + 1j * imgU) / 2.0
    magnitude = np.sqrt(imgQ**2 + imgU**2)
    px = -np.sin(angle) * magnitude
    py = np.cos(angle) * magnitude
    return px, py

def calculate_fractional_polarization(imgI, imgQ, imgU):
    """Calculates fractional polarization m."""
    if imgI is None or imgQ is None or imgU is None:
        return None
    pol_flux = np.sqrt(imgQ**2 + imgU**2)
    # Avoid division by zero or near-zero; return NaN or 0 where I is small
    m = np.divide(pol_flux, imgI, out=np.full_like(imgI, np.nan), where=imgI > 1e-9) # Adjust threshold as needed
    return m

def apply_polarization_threshold(px, py, m, imgI, threshold_value):
    """Applies thresholding to polarization vectors based on total intensity."""
    if px is None or py is None or m is None or imgI is None:
       return None, None, None
    mask = imgI < threshold_value
    px_masked = np.where(mask, np.nan, px)
    py_masked = np.where(mask, np.nan, py)
    m_masked = np.where(mask, np.nan, m)
    return px_masked, py_masked, m_masked

def apply_fractional_polarization_threshold(px, py, m, threshold_fraction, pol_flux_max):
    """Applies thresholding based on fraction of max polarized flux."""
    if px is None or py is None or m is None:
        return None, None, None
    pol_flux = np.sqrt(px**2 + py**2)
    mask = pol_flux < (threshold_fraction * pol_flux_max)
    px_masked = np.where(mask, np.nan, px)
    py_masked = np.where(mask, np.nan, py)
    m_masked = np.where(mask, np.nan, m)
    return px_masked, py_masked, m_masked


def load_nxcorr_data(filepath, columns):
    """Loads specific columns from a CSV file using pandas."""
    try:
        df = pd.read_csv(filepath)
        data = {col: df[col] for col in columns}
        return data
    except Exception as e:
        print(f"Error loading nxcorr data from {filepath}: {e}")
        return {col: None for col in columns} # Return dict with None values

def load_vida_data(filepath, columns):
    """Loads specific VIDA columns from a CSV file using pandas."""
    return load_nxcorr_data(filepath, columns) # Same logic for now

def load_pattern_speed_data(dir_path):
    """Loads autocorrelation and pattern speed data from numpy files."""
    data = {'autocorr': None, 'ps': None, 'ps_err_hi': None, 'ps_err_lo': None}
    try:
        data['autocorr'] = np.load(os.path.join(dir_path, 'autocorrelation.npy'))
        ps_data = np.load(os.path.join(dir_path, 'pattern_speed_uncertainty.npy'))
        if ps_data.ndim > 0 and len(ps_data) >= 3:
            data['ps'] = ps_data[0]
            data['ps_err_hi'] = ps_data[1]
            data['ps_err_lo'] = ps_data[2]
        else:
             print(f"Warning: Unexpected format for pattern_speed_uncertainty.npy in {dir_path}")
    except Exception as e:
        print(f"Error loading pattern speed data from {dir_path}: {e}")
    return data

def save_plot(fig, base_filename):
    """Saves the plot in both PNG and PDF formats."""
    fig.savefig(f"{base_filename}.png", bbox_inches='tight', dpi=300)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight')
    print(f"Saved plots: {base_filename}.png, {base_filename}.pdf")

def setup_colorbar(fig, mappable, cax_rect, label, fontsize, pow_limits=(0, 0), use_math_text=True):
    """Adds and configures a colorbar."""
    cax = fig.add_axes(cax_rect)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label=label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    if pow_limits:
        cbar.formatter.set_powerlimits(pow_limits)
        cbar.formatter.set_useMathText(use_math_text)
    return cbar

def add_scale_bar(ax, x_coords, y_coords, text, x_text, y_text, **kwargs):
     """Adds a scale bar and text to an axis."""
     ax.plot(x_coords, y_coords, **kwargs)
     ax.text(x_text, y_text, text, horizontalalignment='center', **kwargs)


# ==============================================================================
# Calibrator Plotting Function
# ==============================================================================

def plot_calibrator(img, crop_slice, params, output_filename):
    """Plots the calibrator image (Total Intensity + Polarization)."""
    print("Plotting Calibrator...")
    if img is None:
        print("Error: Calibrator image is None. Skipping plot.")
        return

    imgI = get_image_data(img, pol='I', crop_slice=crop_slice)
    imgQ = get_image_data(img, pol='Q', crop_slice=crop_slice)
    imgU = get_image_data(img, pol='U', crop_slice=crop_slice)

    if imgI is None or imgQ is None or imgU is None:
        print("Error extracting calibrator image data. Skipping plot.")
        return

    fov = img.psize / eh.RADPERUAS * imgI.shape[0] * params['fov_scale'] # Adjust FOV scaling if needed
    extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]

    # Determine color limits
    vmax = np.nanmax(imgI)
    # Robust vmin: use a percentile or filter out outliers if needed
    vmin_candidate = np.nanmax(get_image_data(img, pol='I')[0:crop_slice[0].start, :]) # Get region outside crop
    vmin = max(vmin_candidate, vmax * 1e-4) # Ensure vmin is positive and smaller than vmax
    if vmin >= vmax:
        vmin = vmax * 1e-4 # Fallback if vmin calculation failed
        print(f"Warning: Calibrator vmin >= vmax. Setting vmin to {vmin:.2e}")
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # Calculate polarization
    px, py = calculate_polarization_vectors(imgQ, imgU)
    # Apply threshold based on intensity (original logic)
    px, py, _ = apply_polarization_threshold(px, py, None, imgI, vmin * 1.5) # Pass dummy m

    # Setup plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 6.5), gridspec_kw={'width_ratios': [1, 0.13]}) # Adjusted width ratio slightly

    # Plot total intensity
    im = ax[0].imshow(imgI, norm=norm, extent=extent, origin='lower', aspect='equal') # Specify origin and aspect

    # Plot polarization quiver
    x, y = np.meshgrid(np.linspace(extent[0], extent[1], imgI.shape[1]),
                       np.linspace(extent[2], extent[3], imgI.shape[0]))
    sk = params['sk']
    if px is not None and py is not None:
        ax[0].quiver(x[::sk, ::sk], y[::sk, ::sk], px[::sk, ::sk], py[::sk, ::sk],
                     pivot='mid', angles='uv', ec='w', fc='navy', # Fixed color for calibrator
                     width=0.007, scale=0.2, linewidth=0, headwidth=0,
                     headlength=0, headaxislength=0) # Using original scale/width

    # Add scale bar
    add_scale_bar(ax[0], [-35, -15], [30, 30], '20 $\mu$as', -25, 32, c='w', fontsize=params['fsize'])

    # Add colorbar
    setup_colorbar(fig, im, [0.87, 0.12, 0.025, 0.76], # Adjusted position/size
                   '(Jy/px)', params['fsize'], pow_limits=None, use_math_text=False)

    # Final settings
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('JC1924', fontsize=params['fsize']) # Use fsize for title

    plt.tight_layout(pad=0.5, rect=[0, 0, 0.85, 1]) # Adjust rect to prevent overlap with colorbar
    save_plot(fig, output_filename)
    plt.close(fig)
    print("Calibrator plot done.")


# ==============================================================================
# Static Models Plotting Function
# ==============================================================================

def load_static_data(models, pipe_name, crop_slice):
    """Loads all necessary data for static model plotting."""
    print("Loading static model data...")
    data = {'nxcorr': {}, 'images': {}}

    # Load NxCorr data
    nxcorr_cols = {
        'I': f'nxcorr_stat_I_{pipe_name}', 'P': f'nxcorr_stat_P_{pipe_name}', 'X': f'nxcorr_stat_X_{pipe_name}',
        'thrI': 'nxcorr_stat_thres_I', 'thrP': 'nxcorr_stat_thres_P', 'thrX': 'nxcorr_stat_thres_X'
    }
    for mod in models:
        filepath = STATIC_NXCORR_FILE_TPL.format(mod=mod)
        nx_data = load_nxcorr_data(filepath, nxcorr_cols.values())
        # Calculate mean, handling potential None values from loading errors
        data['nxcorr'][mod] = {key: np.mean(nx_data[val]) if nx_data[val] is not None else np.nan
                               for key, val in nxcorr_cols.items()}

    # Load images
    data['images'] = {'gt': {}, 're': {}}
    for mod in models:
        pathgt = STATIC_TRUTH_FILE_TPL.format(mod=mod)
        pathre = STATIC_RECON_FILE_TPL.format(mod=mod, pipe=pipe_name)
        imsgt = load_movie(pathgt)
        imsre = load_movie(pathre)
        imgt = imsgt.im_list()[0] if imsgt else None
        imre = imsre.im_list()[0] if imsre else None

        data['images']['gt'][mod] = {
            'I': get_image_data(imgt, 'I', crop_slice),
            'Q': get_image_data(imgt, 'Q', crop_slice),
            'U': get_image_data(imgt, 'U', crop_slice)
        }
        data['images']['re'][mod] = {
            'I': get_image_data(imre, 'I', crop_slice),
            'Q': get_image_data(imre, 'Q', crop_slice),
            'U': get_image_data(imre, 'U', crop_slice)
        }
    print("Static model data loading complete.")
    return data

def plot_static_models(models, data, params, output_filename):
    """Plots the static model comparisons."""
    print("Plotting static models...")
    imgtI_dict = {mod: data['images']['gt'][mod]['I'] for mod in models}
    imreI_dict = {mod: data['images']['re'][mod]['I'] for mod in models}
    imgtQ_dict = {mod: data['images']['gt'][mod]['Q'] for mod in models}
    imgtU_dict = {mod: data['images']['gt'][mod]['U'] for mod in models}
    imreQ_dict = {mod: data['images']['re'][mod]['Q'] for mod in models}
    imreU_dict = {mod: data['images']['re'][mod]['U'] for mod in models}
    nxcorr_dict = data['nxcorr']

    # Check if any essential data is missing
    if any(img is None for img_dict in imgtI_dict.values() for img in img_dict) or \
       any(img is None for img_dict in imreI_dict.values() for img in img_dict):
        print("Error: Missing static image data. Skipping plot.")
        return

    # Determine vmax across all ground truth models for consistent scaling
    valid_max_vals = [np.nanmax(img) for img in imgtI_dict.values() if img is not None and img.size > 0]
    vmax = max(valid_max_vals) if valid_max_vals else 1.0 # Default vmax if no valid data
    vmin = 0 # Static plots use vmin=0

    num_models = len(models)
    fig, ax = plt.subplots(5, num_models + 1, figsize=(16, 7), # Adjusted figsize slightly
                           gridspec_kw={'height_ratios': [0.9, 0.9, 0.2, 0.2, 0.2],
                                        'width_ratios': [1] * num_models + [0.25]})

    # Turn off axes for the last column (colorbar space + labels)
    for i in range(5):
        ax[i, num_models].axis('off')

    # Plotting parameters
    extent = params['extent']
    sk = params['sk']
    fsize = params['fsize']
    ftsize = params['ftsize']
    cnorm = mcolors.Normalize(vmin=PLOT_PARAMS['pol_cnorm_vmin'], vmax=PLOT_PARAMS['pol_cnorm_vmax'])

    # Get representative shape for meshgrid (assuming all cropped images have same shape)
    rep_img = next(iter(imgtI_dict.values()), None)
    if rep_img is None:
        print("Error: No valid static images found for meshgrid creation.")
        plt.close(fig)
        return
    x, y = np.meshgrid(np.linspace(extent[0], extent[1], rep_img.shape[1]),
                       np.linspace(extent[3], extent[2], rep_img.shape[0])) # Flipped y for imshow extent

    # Plot images and polarization
    imI_mappable = None
    imP_mappable = None
    for i, mod in enumerate(models):
        # Ground Truth
        imgtI = imgtI_dict.get(mod)
        imgtQ = imgtQ_dict.get(mod)
        imgtU = imgtU_dict.get(mod)
        if imgtI is not None:
            imI = ax[0, i].imshow(imgtI, vmin=vmin, vmax=vmax, extent=extent, origin='lower', aspect='equal')
            if imI_mappable is None: imI_mappable = imI # Store for colorbar

            px_gt, py_gt = calculate_polarization_vectors(imgtQ, imgtU)
            m_gt = calculate_fractional_polarization(imgtI, imgtQ, imgtU)
            # Apply threshold (original logic: 0.5 * max(I)) - consider revising if needed
            px_gt, py_gt, m_gt = apply_polarization_threshold(px_gt, py_gt, m_gt, imgtI, 0.5 * np.nanmax(imgtI) if np.any(imgtI) else 0)

            if px_gt is not None and py_gt is not None and m_gt is not None:
                imP = ax[0, i].quiver(x[::sk, ::sk], y[::sk, ::sk], px_gt[::sk, ::sk], py_gt[::sk, ::sk], m_gt[::sk, ::sk],
                                    cmap=PLOT_PARAMS['pol_cmap'], norm=cnorm, **PLOT_PARAMS['pol_quiver'])
                if imP_mappable is None: imP_mappable = imP # Store for colorbar

        # Reconstruction
        imreI = imreI_dict.get(mod)
        imreQ = imreQ_dict.get(mod)
        imreU = imreU_dict.get(mod)
        if imreI is not None:
            ax[1, i].imshow(imreI, vmin=vmin, vmax=vmax, extent=extent, origin='lower', aspect='equal')

            px_re, py_re = calculate_polarization_vectors(imreQ, imreU)
            m_re = calculate_fractional_polarization(imreI, imreQ, imreU)
            px_re, py_re, m_re = apply_polarization_threshold(px_re, py_re, m_re, imreI, 0.5 * np.nanmax(imreI) if np.any(imreI) else 0)

            if px_re is not None and py_re is not None and m_re is not None:
                 ax[1, i].quiver(x[::sk, ::sk], y[::sk, ::sk], px_re[::sk, ::sk], py_re[::sk, ::sk], m_re[::sk, ::sk],
                                 cmap=PLOT_PARAMS['pol_cmap'], norm=cnorm, **PLOT_PARAMS['pol_quiver'])

        # Turn off ticks for image axes
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])

        # Set titles
        ax[0, i].set_title(mod, fontsize=fsize)

        # Display NxCorr values
        nx = nxcorr_dict.get(mod, {}) # Get nxcorr for the model, default to empty dict
        ax[2, i].text(0.5, 0.5, f"{nx.get('I', np.nan):.2f} ({nx.get('thrI', np.nan):.2f})",
                      horizontalalignment='center', verticalalignment='center', fontsize=fsize)
        ax[3, i].text(0.5, 0.5, f"{nx.get('P', np.nan):.2f} ({nx.get('thrP', np.nan):.2f})",
                      horizontalalignment='center', verticalalignment='center', fontsize=fsize)
        ax[4, i].text(0.5, 0.5, f"{nx.get('X', np.nan):.2f} ({nx.get('thrX', np.nan):.2f})",
                      horizontalalignment='center', verticalalignment='center', fontsize=fsize)
        # Turn off ticks for text axes
        for j in range(2, 5):
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].axis('off') # Also turn off axis frame for text cells

    # Row labels
    ax[0, 0].set_ylabel('ground truth', fontsize=fsize)
    ax[1, 0].set_ylabel('reconstruction', fontsize=fsize)

    # NxCorr labels (in the last column)
    ax[2, num_models].text(0.5, 0.5, '$\eta_{xcorr}$ I', horizontalalignment='center', verticalalignment='center', fontsize=fsize)
    ax[3, num_models].text(0.5, 0.5, '$\eta_{xcorr}$ P', horizontalalignment='center', verticalalignment='center', fontsize=fsize)
    ax[4, num_models].text(0.5, 0.5, '$\eta_{xcorr}$ $\chi$', horizontalalignment='center', verticalalignment='center', fontsize=fsize)

    # Scale bars
    add_scale_bar(ax[0,0], [-55, -15], [-55, -55], '40 $\mu$as', -35, -50, c=PLOT_PARAMS['scale_bar_color']) #fontsize=ftsize
    add_scale_bar(ax[1,0], [-55, -15], [-55, -55], '40 $\mu$as', -35, -50, c=PLOT_PARAMS['scale_bar_color']) #fontsize=ftsize

    # Colorbars
    if imI_mappable:
        setup_colorbar(fig, imI_mappable, [0.92, 0.633, 0.01, 0.315], '(Jy/px)', ftsize) # Adjusted position/size
    if imP_mappable:
        setup_colorbar(fig, imP_mappable, [0.92, 0.283, 0.01, 0.315], 'm', ftsize, pow_limits=None) # Adjusted position/size

    plt.tight_layout(pad=0.2, rect=[0, 0, 0.91, 1]) # Adjust rect to prevent overlap
    save_plot(fig, output_filename)
    plt.close(fig)
    print("Static models plot done.")


# ==============================================================================
# Dynamic Models - Data Loading Function
# ==============================================================================

def load_dynamic_data(models, pipe_name, frame_indices_map, crop_slice):
    """Loads all necessary data for dynamic model plotting."""
    print("Loading dynamic model data...")
    data = {
        'movies': {'gt': {}, 're': {}, 'dygt': {}, 'dyre': {}},
        'frames': {'gt': {}, 're': {}, 'dygt': {}, 'dyre': {}},
        'frame_times': {},
        'nxcorr': {},
        'vida': {},
        'pattern_speed': {'gt': {}, 're': {}},
        'pass_fail': {}
    }

    # --- Load Movies ---
    movie_types = {
        'gt': DYNAMIC_TRUTH_FILE_TPL, 're': DYNAMIC_RECON_FILE_TPL,
        'dygt': DYNAMIC_COMP_TRUTH_FILE_TPL, 'dyre': DYNAMIC_COMP_RECON_FILE_TPL
    }
    for type_key, path_tpl in movie_types.items():
        for mod in models:
            path = path_tpl.format(mod=mod, pipe=pipe_name)
            data['movies'][type_key][mod] = load_movie(path)

    # --- Get Specific Frames and Times ---
    for mod in models:
        mov_re = data['movies']['re'].get(mod)
        if mov_re and hasattr(mov_re, 'times') and len(mov_re.times) > 0:
             # Ensure indices are within bounds
            indices = [min(idx, len(mov_re.times) - 1) for idx in frame_indices_map.get(mod, [])]
            data['frame_times'][mod] = [mov_re.times[t] for t in indices]

             # Extract frames for all movie types
            for type_key in data['movies']:
                 mov = data['movies'][type_key].get(mod)
                 if mov:
                     ims = [mov.get_image(t) for t in data['frame_times'][mod]]
                     data['frames'][type_key][mod] = {
                         'I': [get_image_data(im, 'I', crop_slice) for im in ims],
                         'Q': [get_image_data(im, 'Q', crop_slice) for im in ims],
                         'U': [get_image_data(im, 'U', crop_slice) for im in ims],
                     }
                 else: # Initialize empty if movie failed to load
                     data['frames'][type_key][mod] = {'I': [None]*len(indices), 'Q': [None]*len(indices), 'U': [None]*len(indices)}

        else:
             print(f"Warning: Could not load reconstruction movie or times for model {mod}. Skipping frame extraction.")
             data['frame_times'][mod] = []
             for type_key in data['movies']:
                 data['frames'][type_key][mod] = {'I': [], 'Q': [], 'U': []}


    # --- Load NxCorr Data ---
    nxcorr_files = {
        'dyn': DYNAMIC_NXCORR_DYN_FILE_TPL,
        'thresh': DYNAMIC_NXCORR_THRESH_FILE_TPL
    }
    nxcorr_cols = {
        'dyn': {'time': 'time', 'thrI': 'nxcorr_dyn_thres_I', f'recI': f'nxcorr_dyn_I_{pipe_name}'},
        'thresh': {'thrP': 'nxcorr_thres_P', f'recP': f'nxcorr_P_{pipe_name}',
                   'thrX': 'nxcorr_thres_X', f'recX': f'nxcorr_X_{pipe_name}'}
    }
    for mod in models:
        data['nxcorr'][mod] = {}
        for key, filepath_tpl in nxcorr_files.items():
             filepath = filepath_tpl.format(mod=mod)
             cols_to_load = list(nxcorr_cols[key].values())
             loaded_data = load_nxcorr_data(filepath, cols_to_load)
             # Map loaded data back to descriptive keys
             for desc_key, orig_col in nxcorr_cols[key].items():
                 data['nxcorr'][mod][desc_key] = loaded_data.get(orig_col)


    # --- Load VIDA Data (Position Angle / Beta2) ---
    vida_files = {
        'all': DYNAMIC_VIDA_ALL_FILE_TPL,
        'truth': DYNAMIC_VIDA_TRUTH_FILE_TPL,
        'pol_all': DYNAMIC_VIDA_POL_ALL_FILE_TPL
    }
    vida_cols = {
        'all': {
            'x0gt': 'model_1_x0_1_truth', 'y0gt': 'model_1_y0_1_truth',
            f'x0re': f'model_1_x0_1_{pipe_name}', f'y0re': f'model_1_y0_1_{pipe_name}'
        },
        'truth': {'time': 'time'},
        'pol_all': {
            'rebeta2gt': 're_betalp_2_truth', 'imbeta2gt': 'im_betalp_2_truth',
            f'rebeta2re': f're_betalp_2_{pipe_name}', f'imbeta2re': f'im_betalp_2_{pipe_name}'
        }
    }
    for mod in models:
        data['vida'][mod] = {}
        for key, filepath_tpl in vida_files.items():
            filepath = filepath_tpl.format(mod=mod)
            cols_to_load = list(vida_cols[key].values())
            loaded_data = load_vida_data(filepath, cols_to_load)
            for desc_key, orig_col in vida_cols[key].items():
                 data['vida'][mod][desc_key] = loaded_data.get(orig_col)

        # Post-process VIDA data (calculate angles, apply conversions)
        vida_mod = data['vida'][mod]
        if vida_mod.get('x0gt') is not None and vida_mod.get('y0gt') is not None:
            vida_mod['thetagt'] = np.rad2deg(np.arctan2(vida_mod['y0gt'], vida_mod['x0gt'])) / eh.RADPERUAS # Convert radians from file? Check units. Assuming input IS already uas.
            vida_mod['x0gt'] = vida_mod['x0gt'] / eh.RADPERUAS # Assuming input IS NOT uas. Correct if needed.
            vida_mod['y0gt'] = vida_mod['y0gt'] / eh.RADPERUAS
        if vida_mod.get('x0re') is not None and vida_mod.get('y0re') is not None:
            vida_mod['thetare'] = np.rad2deg(np.arctan2(vida_mod['y0re'], vida_mod['x0re'])) / eh.RADPERUAS # Convert radians from file? Check units. Assuming input IS already uas.
            vida_mod['x0re'] = vida_mod['x0re'] / eh.RADPERUAS # Assuming input IS NOT uas. Correct if needed.
            vida_mod['y0re'] = vida_mod['y0re'] / eh.RADPERUAS

        # Handle beta2 complex number and angle (slicing [1:-1] as in original)
        for prefix in ['gt', 're']:
             re_key = f'rebeta2{prefix}'
             im_key = f'imbeta2{prefix}'
             arg_key = f'argbeta2{prefix}'
             if vida_mod.get(re_key) is not None and vida_mod.get(im_key) is not None:
                 # Apply slicing safely
                 re_beta = vida_mod[re_key][1:-1] if len(vida_mod[re_key]) > 2 else vida_mod[re_key]
                 im_beta = vida_mod[im_key][1:-1] if len(vida_mod[im_key]) > 2 else vida_mod[im_key]
                 # Convert to numeric, coercing errors
                 re_beta_num = pd.to_numeric(re_beta, errors='coerce')
                 im_beta_num = pd.to_numeric(im_beta, errors='coerce')
                 # Calculate angle where both are valid numbers
                 valid_mask = ~np.isnan(re_beta_num) & ~np.isnan(im_beta_num)
                 vida_mod[arg_key] = np.full_like(re_beta_num, np.nan) # Initialize with NaN
                 vida_mod[arg_key][valid_mask] = np.angle(re_beta_num[valid_mask] + 1j * im_beta_num[valid_mask])


        # Manual phase wrapping (handle potential errors)
        if mod == 'mring+hs-incoh' and vida_mod.get('thetare') is not None:
            try:
                theta_re = vida_mod['thetare'].copy() # Work on a copy
                subset = theta_re[:20]
                mask = subset < -90
                subset[mask] += 360 # Changed from 270 to 360 for standard wrapping? Verify original intent.
                theta_re[:20] = subset
                vida_mod['thetare_wrapped'] = theta_re # Store separately or overwrite 'thetare'
            except Exception as e:
                print(f"Warning: Could not apply phase wrapping for {mod}: {e}")


    # --- Load Pattern Speed Data ---
    for mod in models:
        data['pattern_speed']['gt'][mod] = load_pattern_speed_data(PATTERN_SPEED_TRUTH_DIR_TPL.format(mod=mod))
        data['pattern_speed']['re'][mod] = load_pattern_speed_data(PATTERN_SPEED_RECON_DIR_TPL.format(mod=mod, pipe=pipe_name))

    # --- Load Pass/Fail Percentages ---
    pass_fail_files = {
        'nxcorr_dyn': DYNAMIC_NXCORR_DYN_FILE_TPL,
        'vida_all': DYNAMIC_VIDA_ALL_FILE_TPL,
        'vida_pol_all': DYNAMIC_VIDA_POL_ALL_FILE_TPL
    }
    pass_fail_cols = {
        'nxcorr_dyn': {
            'nxcorrI': f'nxcorr_dyn_pass_I_{pipe_name}',
            'nxcorrP': f'nxcorr_dyn_pass_P_{pipe_name}',
            'nxcorrX': f'nxcorr_dyn_pass_X_{pipe_name}',
        },
        'vida_all': { # Default uses PA
            'pa_x_b': f'pass_percent_pa_{pipe_name}',
        },
         # Specific overrides based on model name (as in original)
        'vida_all_cross': { # Use 'x' for mring+hs-cross
             'pa_x_b': f'pass_percent_x_{pipe_name}',
        },
        'vida_pol_all_varbeta': { # Use 'argbeta2' for mring-varbeta2
             'pa_x_b': f'pass_percent_argbeta2_{pipe_name}',
        }
    }
    for mod in models:
        data['pass_fail'][mod] = {}
        # Load nxcorr pass/fail
        filepath_nx = pass_fail_files['nxcorr_dyn'].format(mod=mod)
        cols_nx = list(pass_fail_cols['nxcorr_dyn'].values())
        loaded_nx = load_nxcorr_data(filepath_nx, cols_nx)
        for desc_key, orig_col in pass_fail_cols['nxcorr_dyn'].items():
             # Take the first value [0], default to NaN if load failed or empty
             data['pass_fail'][mod][desc_key] = loaded_nx[orig_col][0] if loaded_nx.get(orig_col) is not None and len(loaded_nx[orig_col]) > 0 else np.nan

        # Load PA/X/Beta pass/fail with overrides
        cols_pa_x_b = {}
        filepath_pa_x_b = ""
        if mod == 'mring+hs-cross':
            filepath_pa_x_b = pass_fail_files['vida_all'].format(mod=mod) # File is vida_all
            cols_pa_x_b = pass_fail_cols['vida_all_cross']          # Cols use _x_ override
        elif mod == 'mring-varbeta2':
            filepath_pa_x_b = pass_fail_files['vida_pol_all'].format(mod=mod) # File is vida_pol_all
            cols_pa_x_b = pass_fail_cols['vida_pol_all_varbeta']     # Cols use _argbeta2_ override
        else:
            filepath_pa_x_b = pass_fail_files['vida_all'].format(mod=mod) # File is vida_all
            cols_pa_x_b = pass_fail_cols['vida_all']                 # Default cols use _pa_

        cols_to_load_pa = list(cols_pa_x_b.values())
        loaded_pa = load_nxcorr_data(filepath_pa_x_b, cols_to_load_pa)
        for desc_key, orig_col in cols_pa_x_b.items():
             data['pass_fail'][mod][desc_key] = loaded_pa[orig_col][0] if loaded_pa.get(orig_col) is not None and len(loaded_pa[orig_col]) > 0 else np.nan


    print("Dynamic model data loading complete.")
    return data


# ==============================================================================
# Dynamic Models - Plotting Functions
# ==============================================================================

# --- Individual Model Plot ("Printed Version") ---
def plot_dynamic_model_individual(mod, data, params, output_filename_tpl):
    """Plots detailed comparison for a single dynamic model."""
    print(f"Plotting individual dynamic model: {mod}...")

    # Extract data for the specific model
    frames = {key: data['frames'][key][mod] for key in data['frames']}
    frame_times = data['frame_times'].get(mod, [])
    nxcorr = data['nxcorr'].get(mod, {})
    vida = data['vida'].get(mod, {})
    ps_gt = data['pattern_speed']['gt'].get(mod, {})
    ps_re = data['pattern_speed']['re'].get(mod, {})

    if not frame_times or not frames['gt']['I']: # Check if essential data is missing
        print(f"Error: Missing frame data for model {mod}. Skipping plot.")
        return

    # --- Plotting Setup ---
    num_frames = min(NUM_DYNAMIC_FRAMES, len(frame_times)) # Use the configured number or available frames
    fig, ax = plt.subplots(4, num_frames + 1, figsize=(16, 11), # Use num_frames
                           gridspec_kw={'height_ratios': [1, 1, 1, 1],
                                        'width_ratios': [1] * num_frames + [0.25]}) # Adjust width ratios

    # Turn off axes for the last column (colorbar space)
    for i in range(4):
        ax[i, num_frames].axis('off')

    # --- Determine Plot Limits ---
    vmax_I = max(np.nanmax(img) for img in frames['gt']['I'] if img is not None and img.size > 0) if any(img is not None for img in frames['gt']['I']) else 1.0
    vmax_dyn = max(np.nanmax(img) for img in frames['dyre']['I'] if img is not None and img.size > 0) if any(img is not None for img in frames['dyre']['I']) else 1e-12 # Ensure positive
    vmin_I = 0
    vmin_dyn = 0

    # --- Plot Parameters ---
    p = params # Shorter alias
    fsize, ftsize, sk = p['fsize'], p['ftsize'], p['sk']
    extent_im = p['extent_im']
    extent_dyn = p['extent_dyn'] # For contour
    extent_ps = p['extent_ps']
    cnorm = mcolors.Normalize(vmin=PLOT_PARAMS['pol_cnorm_vmin'], vmax=PLOT_PARAMS['pol_cnorm_vmax'])

    # Get representative shape for meshgrid
    rep_img = next((img for img in frames['gt']['I'] if img is not None), None)
    if rep_img is None:
        print(f"Error: No valid GT images for meshgrid creation in model {mod}.")
        plt.close(fig)
        return
    x, y = np.meshgrid(np.linspace(extent_im[0], extent_im[1], rep_img.shape[1]),
                       np.linspace(extent_im[3], extent_im[2], rep_img.shape[0])) # Flipped y

    # --- Plotting Loops ---
    mappables = {'im0': None, 'im1': None, 'im2': None, 'im3': None} # To store mappables for colorbars

    for i in range(num_frames):
        # --- Total Intensity and Polarization (Rows 0, 1) ---
        for row, type_key in enumerate(['gt', 're']):
            imgI = frames[type_key]['I'][i]
            imgQ = frames[type_key]['Q'][i]
            imgU = frames[type_key]['U'][i]

            if imgI is not None:
                im = ax[row, i].imshow(imgI, vmin=vmin_I, vmax=vmax_I, extent=extent_im, origin='lower', aspect='equal')
                if row == 0 and mappables['im0'] is None: mappables['im0'] = im # Store GT intensity mappable

                px, py = calculate_polarization_vectors(imgQ, imgU)
                m = calculate_fractional_polarization(imgI, imgQ, imgU)
                # Threshold based on fraction of max polarized flux for this frame
                pol_flux = np.sqrt(np.array(frames[type_key]['Q'][i])**2 + np.array(frames[type_key]['U'][i])**2) if frames[type_key]['Q'][i] is not None else np.array([0])
                maxp = np.nanmax(pol_flux) if pol_flux.size > 0 else 0
                px, py, m = apply_fractional_polarization_threshold(px, py, m, PLOT_PARAMS['pol_norm_frac_threshold'], maxp)

                if px is not None and py is not None and m is not None:
                    quiv = ax[row, i].quiver(x[::sk, ::sk], y[::sk, ::sk], px[::sk, ::sk], py[::sk, ::sk], m[::sk, ::sk],
                                             cmap=PLOT_PARAMS['pol_cmap'], norm=cnorm, **PLOT_PARAMS['pol_quiver'])
                    if row == 0 and mappables['im1'] is None: mappables['im1'] = quiv # Store GT polarization mappable

            # Axis settings
            ax[row, i].set_xticks([])
            ax[row, i].set_yticks([])
            if i == 0: # Set Y labels only for the first column
                label = 'ground truth' if type_key == 'gt' else 'reconstruction'
                ax[row, i].set_ylabel(label, fontsize=fsize)
            if row == 0: # Set titles only for the top row
                 ax[row, i].set_title(f'{frame_times[i]:.2f} UT', fontsize=fsize)


        # --- Dynamic Component (Row 2) ---
        imgDyReI = frames['dyre']['I'][i]
        imgDyGtI = frames['dygt']['I'][i]
        imgDyReQ = frames['dyre']['Q'][i]
        imgDyReU = frames['dyre']['U'][i]
        imgReI = frames['re']['I'][i] # Needed for m calculation

        if imgDyReI is not None:
             im = ax[2, i].imshow(imgDyReI, vmin=vmin_dyn, vmax=max(vmax_dyn, 1e-12), cmap=PLOT_PARAMS['dynamic_comp_cmap'], extent=extent_im, origin='lower', aspect='equal')
             if mappables['im2'] is None: mappables['im2'] = im

        if imgDyGtI is not None:
            try: # Add contour robustly
                 ax[2, i].contour(gaussian_filter(imgDyGtI, 3), levels=[0.0001], # Low level as in original
                                  colors=PLOT_PARAMS['dynamic_comp_contour_color'],
                                  linestyles=PLOT_PARAMS['dynamic_comp_contour_ls'],
                                  linewidths=PLOT_PARAMS['dynamic_comp_contour_lw'],
                                  extent=extent_dyn) # Use extent_dyn here
            except Exception as e:
                 print(f"Warning: Could not plot contour for dynamic GT frame {i} of {mod}: {e}")


        # Plot dynamic polarization if applicable
        if mod in ['mring+hs-pol', 'mring-varbeta2']:
             pol_thresh = 0.3 if mod == 'mring+hs-pol' else 0.1 # Specific thresholds
             px_dy, py_dy = calculate_polarization_vectors(imgDyReQ, imgDyReU)
             m_re = calculate_fractional_polarization(imgReI, frames['re']['Q'][i], frames['re']['U'][i]) # Use total recon m

             # Threshold based on fraction of max *dynamic* polarized flux
             pol_flux_dy = np.sqrt(np.array(imgDyReQ)**2 + np.array(imgDyReU)**2) if imgDyReQ is not None else np.array([0])
             maxp_dy = np.nanmax(pol_flux_dy) if pol_flux_dy.size > 0 else 0
             px_dy, py_dy, _ = apply_fractional_polarization_threshold(px_dy, py_dy, None, pol_thresh, maxp_dy) # Pass dummy m

             if px_dy is not None and py_dy is not None:
                  ax[2, i].quiver(x[::sk, ::sk], y[::sk, ::sk], px_dy[::sk, ::sk], py_dy[::sk, ::sk], m_re[::sk, ::sk], # Color by total m? Check original
                                  cmap=PLOT_PARAMS['pol_cmap'], norm=cnorm, **PLOT_PARAMS['pol_quiver'])

        # Axis settings for row 2
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])
        if i == 0:
             ax[2, i].set_ylabel('dynamic component', fontsize=fsize)


    # --- Merge Axes for Row 3 (NxCorr, VIDA, Pattern Speed) ---
    gs = ax[0, 0].get_gridspec()
    # Remove original axes in row 3, except the last two for pattern speed
    for i in range(num_frames - 1): # Keep last two for pattern speed originally
        ax[3, i].remove()

    # Create merged axes
    axA = fig.add_subplot(gs[3, 0:2])  # NxCorr plot
    axB = fig.add_subplot(gs[3, 2:4])  # VIDA plot
    # Keep ax[3, 4] and ax[3, 5] if num_frames is 6 for pattern speed
    axPS_gt = ax[3, 4] if num_frames >= 5 else fig.add_subplot(gs[3, 4]) # Create if needed
    axPS_re = ax[3, 5] if num_frames >= 6 else fig.add_subplot(gs[3, 5]) # Create if needed

    # --- Plot NxCorr (axA) ---
    plot_times = nxcorr.get('time')
    if plot_times is not None:
        if mod in ['mring+hs-pol', 'mring-varbeta2']:
            axA.plot(plot_times, nxcorr.get('thrP'), label='threshold P', c=PLOT_PARAMS['nxcorr_thresh_color'])
            axA.plot(plot_times, nxcorr.get('recP'), lw=2, label='reconstruction P', c=PLOT_PARAMS['nxcorr_recon_pol_color'])
            axA.plot(plot_times, nxcorr.get('thrX'), ls='--', label=r'threshold $\chi$', c=PLOT_PARAMS['nxcorr_thresh_color'])
            axA.plot(plot_times, nxcorr.get('recX'), lw=2, label=r'reconstruction $\chi$', c=PLOT_PARAMS['nxcorr_recon_chi_color'])
        else:
            axA.plot(plot_times, nxcorr.get('thrI'), label='threshold I', c=PLOT_PARAMS['nxcorr_thresh_color'])
            axA.plot(plot_times, nxcorr.get('recI'), lw=2, label=f'reconstruction I', c=PLOT_PARAMS['nxcorr_recon_color'])
        axA.set_xlabel('UT time (hr)', fontsize=fsize)
        axA.set_ylabel('cross correlation', fontsize=fsize)
        axA.set_ylim(0, 1.05)
        axA.legend(fontsize=ftsize)
    else:
        axA.text(0.5, 0.5, "NxCorr data unavailable", ha='center', va='center', fontsize=ftsize)
        axA.set_xticks([])
        axA.set_yticks([])


    # --- Plot VIDA (axB) ---
    vida_times = vida.get('time')
    if vida_times is not None:
        if mod == 'mring+hs-cross':
            y_gt, y_re = vida.get('x0gt'), vida.get('x0re') # Plot RA (x)
            ylabel = 'RA ($\mu$as)'
            ylim = None
        elif mod == 'mring-varbeta2':
            y_gt, y_re = vida.get('argbeta2gt'), vida.get('argbeta2re') # Plot beta2 angle
            ylabel = r'$\angle\beta_2$ (rad)'
            ylim = None
        else:
            # Use wrapped PA if available, otherwise original
            theta_re_key = 'thetare_wrapped' if 'thetare_wrapped' in vida else 'thetare'
            y_gt, y_re = vida.get('thetagt'), vida.get(theta_re_key) # Plot PA
            ylabel = 'PA (deg)'
            ylim = (-199, 199)

        if y_gt is not None:
             axB.plot(vida_times, y_gt, marker='.', ms=8, ls='-', lw=1, label='ground truth', c=PLOT_PARAMS['vida_gt_color'])
        if y_re is not None:
             axB.plot(vida_times, y_re, marker='.', ms=8, ls='-', lw=1, label=f'reconstruction', c=PLOT_PARAMS['vida_recon_color'])

        axB.set_ylabel(ylabel, fontsize=fsize)
        if ylim: axB.set_ylim(ylim)
        axB.set_xlabel('UT time (hr)', fontsize=fsize)
        axB.legend(fontsize=ftsize)
    else:
        axB.text(0.5, 0.5, "VIDA data unavailable", ha='center', va='center', fontsize=ftsize)
        axB.set_xticks([])
        axB.set_yticks([])

    # --- Plot Autocorrelation (ax[3, 4] and ax[3, 5]) ---
    if mod != 'mring-varbeta2': # Skip for this model as in original
        autocorr_gt = ps_gt.get('autocorr')
        autocorr_re = ps_re.get('autocorr')
        if autocorr_gt is not None:
             im = axPS_gt.imshow(autocorr_gt, cmap=PLOT_PARAMS['autocorr_cmap'], origin='lower',
                                 extent=extent_ps, vmin=PLOT_PARAMS['autocorr_vmin'], vmax=PLOT_PARAMS['autocorr_vmax'])
             if mappables['im3'] is None: mappables['im3'] = im # Store for colorbar
             axPS_gt.set_aspect((extent_ps[1] - extent_ps[0]) / (extent_ps[3] - extent_ps[2]))

        if autocorr_re is not None:
             axPS_re.imshow(autocorr_re, cmap=PLOT_PARAMS['autocorr_cmap'], origin='lower',
                            extent=extent_ps, vmin=PLOT_PARAMS['autocorr_vmin'], vmax=PLOT_PARAMS['autocorr_vmax'])
             axPS_re.set_aspect((extent_ps[1] - extent_ps[0]) / (extent_ps[3] - extent_ps[2]))

        axPS_gt.set_xlabel('$\Delta$t (GM/c$^3$)', fontsize=fsize)
        axPS_re.set_xlabel('$\Delta$t (GM/c$^3$)', fontsize=fsize)
        axPS_gt.set_ylabel('$\Delta$PA (deg)', fontsize=fsize)
        axPS_re.set_yticks([])
    else:
        # If skipping pattern speed, remove the axes placeholders
        axPS_gt.remove()
        axPS_re.remove()


    # --- Colorbars ---
    cbar_rects = { # Define positions for each colorbar
        'im0': [0.89, 0.697, 0.007, 0.180], # Total Intensity
        'im1': [0.89, 0.502, 0.007, 0.179], # Polarization 'm'
        'im2': [0.89, 0.307, 0.007, 0.177], # Dynamic Intensity
        'im3': [0.89, 0.113, 0.007, 0.181], # Autocorrelation
    }
    cbar_labels = {'im0': '(Jy/px)', 'im1': 'm', 'im2': '(Jy/px)', 'im3': 'autocorrelation'}
    cbar_powlimits = {'im0': (0,0), 'im1': None, 'im2': (0,0), 'im3': None}

    # Add colorbars only if the corresponding mappable exists
    for key, mappable in mappables.items():
        if mappable and (key != 'im3' or mod != 'mring-varbeta2'): # Skip im3 cbar if mod is varbeta2
             setup_colorbar(fig, mappable, cbar_rects[key], cbar_labels[key], ftsize, cbar_powlimits[key])


    # --- Scale Bars ---
    add_scale_bar(ax[0,0], [-55, -15], [-55, -55], '40 $\mu$as', -35, -50, c=PLOT_PARAMS['scale_bar_color'], fontsize=ftsize)
    add_scale_bar(ax[1,0], [-55, -15], [-55, -55], '40 $\mu$as', -35, -50, c=PLOT_PARAMS['scale_bar_color'], fontsize=ftsize)
    add_scale_bar(ax[2,0], [-55, -15], [-55, -55], '40 $\mu$as', -35, -50, c=PLOT_PARAMS['dynamic_comp_contour_color'], fontsize=ftsize) # Use black for dynamic

    # --- Final Adjustments ---
    plt.subplots_adjust(wspace=0.05, hspace=0.1) # Fine-tune spacing

    # Adjust subplot positions (original logic seemed complex, might need tweaking)
    try:
        boxA = axA.get_position()
        boxB = axB.get_position()
        # Attempt to resize based on original ratios - this needs careful adjustment
        boxA.x1 = boxA.x0 + (boxB.x0 - boxA.x0) * (2/4) # Approx 2 units wide
        boxB.x1 = boxB.x0 + (boxB.x0 - boxA.x0) * (2/4) # Approx 2 units wide

        axA.set_position(boxA)
        axB.set_position(boxB)
        # Position pattern speed plots if they exist
        if mod != 'mring-varbeta2':
             boxPS_gt = axPS_gt.get_position()
             boxPS_re = axPS_re.get_position()
             # Reposition relative to axB? Needs logic based on grid layout.
             # Example: boxPS_gt.x0 = boxB.x1 + some_gap
             # axPS_gt.set_position(boxPS_gt)
             # axPS_re.set_position(boxPS_re)
             pass # Placeholder - positioning needs refinement based on exact layout goals
    except Exception as e:
        print(f"Warning: Could not automatically adjust subplot positions for {mod}: {e}")


    # --- Save and Close ---
    output_filename = output_filename_tpl.format(mod=mod)
    save_plot(fig, output_filename)
    plt.close(fig)
    print(f"Individual dynamic model plot done: {mod}")


# --- Validation Ladder Plot ---
def plot_dynamic_ladder(models_subset, data, params, output_filename):
    """Plots the validation ladder comparison for a subset of dynamic models."""
    print("Plotting dynamic validation ladder...")
    nmod = len(models_subset)
    if nmod == 0:
        print("No models specified for ladder plot. Skipping.")
        return

    # --- Plotting Setup ---
    fbsize, fsize, ftsize = params['fbsize'], params['fsize'], params['ftsize']
    extent_ps = params['extent_ps']

    fig, ax = plt.subplots(nmod, 6, figsize=(16, 2.8 * nmod), # Height scales with num models
                           gridspec_kw={'height_ratios': [1] * nmod,
                                        'width_ratios': [1, 1, 1, 1, 1, 1]})
    gs = ax[0, 0].get_gridspec()
    axA = [] # List to hold NxCorr axes
    axB = [] # List to hold VIDA axes

    mappable_ps = None # For autocorrelation colorbar

    # --- Plotting Loop ---
    for i, mod in enumerate(models_subset):
        # Extract data for the model
        nxcorr = data['nxcorr'].get(mod, {})
        vida = data['vida'].get(mod, {})
        ps_gt = data['pattern_speed']['gt'].get(mod, {})
        ps_re = data['pattern_speed']['re'].get(mod, {})
        pass_fail = data['pass_fail'].get(mod, {})

        # --- Merge Axes for NxCorr and VIDA ---
        for j in range(4): # Remove first 4 axes in the row
             ax[i, j].remove()
        axA.append(fig.add_subplot(gs[i, 0:2])) # Span columns 0-1
        axB.append(fig.add_subplot(gs[i, 2:4])) # Span columns 2-3

        # Set model name as Y label for the NxCorr plot
        axA[i].set_ylabel(mod, fontsize=fbsize)

        # --- Plot NxCorr (axA[i]) ---
        plot_times = nxcorr.get('time')
        handles = []
        if plot_times is not None:
            if mod in ['mring+hs-pol', 'mring-varbeta2']:
                # Plot P and Chi
                h1, = axA[i].plot(plot_times, nxcorr.get('thrP'), label='threshold P', c=PLOT_PARAMS['nxcorr_thresh_color'])
                h2, = axA[i].plot(plot_times, nxcorr.get('thrX'), ls='--', label=r'threshold $\chi$', c=PLOT_PARAMS['nxcorr_thresh_color'])
                h3, = axA[i].plot(plot_times, nxcorr.get('recP'), lw=2, label='reconstruction P', c=PLOT_PARAMS['nxcorr_recon_pol_color'])
                h4, = axA[i].plot(plot_times, nxcorr.get('recX'), lw=2, label=r'reconstruction $\chi$', c=PLOT_PARAMS['nxcorr_recon_chi_color'])
                handles.extend([h1, h2, h3, h4])
                # Add pass/fail percentages to legend
                passP = pass_fail.get('nxcorrP', np.nan)
                passX = pass_fail.get('nxcorrX', np.nan)
                handles.append(mpatches.Patch(color='none', label=f'Pass P = {passP:.0f} %' if not np.isnan(passP) else 'Pass P = N/A'))
                handles.append(mpatches.Patch(color='none', label=f'Pass $\chi$ = {passX:.0f} %' if not np.isnan(passX) else 'Pass $\chi$ = N/A'))
                axA[i].legend(handles=handles, ncol=2, fontsize=ftsize, loc='lower left', frameon=False)

            else:
                # Plot I
                h1, = axA[i].plot(plot_times, nxcorr.get('thrI'), label='threshold I', c=PLOT_PARAMS['nxcorr_thresh_color'])
                h2, = axA[i].plot(plot_times, nxcorr.get('recI'), lw=2, label='reconstruction I', c=PLOT_PARAMS['nxcorr_recon_color'])
                handles.extend([h1, h2])
                # Add pass/fail percentage to legend
                passI = pass_fail.get('nxcorrI', np.nan)
                handles.append(mpatches.Patch(color='none', label=f'Pass I = {passI:.0f} %' if not np.isnan(passI) else 'Pass I = N/A'))
                axA[i].legend(handles=handles, ncol=1, fontsize=ftsize, loc='lower left', frameon=False)

            axA[i].set_ylim(0.01, 1.05)
        else:
             axA[i].text(0.5, 0.5, "NxCorr data unavailable", ha='center', va='center', fontsize=ftsize)
             axA[i].set_xticks([]); axA[i].set_yticks([])

        # --- Plot VIDA (axB[i]) ---
        vida_times = vida.get('time')
        handles = []
        if vida_times is not None:
            # Determine which quantity to plot and its threshold band
            if mod == 'mring+hs-cross':
                y_gt, y_re = vida.get('x0gt'), vida.get('x0re')
                label_gt, label_re = r'RA ground truth', r'RA reconstruction'
                thresh_val = 5 # uas threshold
                ylim = None
            elif mod == 'mring-varbeta2':
                y_gt, y_re = vida.get('argbeta2gt'), vida.get('argbeta2re')
                label_gt, label_re = r'$\angle\beta_2$ ground truth', r'$\angle\beta_2$ reconstruction'
                thresh_val = np.deg2rad(20) # 20 degrees threshold in radians
                ylim = None
            else:
                theta_re_key = 'thetare_wrapped' if 'thetare_wrapped' in vida else 'thetare'
                y_gt, y_re = vida.get('thetagt'), vida.get(theta_re_key)
                label_gt, label_re = r'PA ground truth', r'PA reconstruction'
                thresh_val = 20 # degrees threshold
                ylim = (-199, 199)

            # Plot threshold band if GT exists
            if y_gt is not None:
                axB[i].fill_between(vida_times, y_gt - thresh_val, y_gt + thresh_val, color=PLOT_PARAMS['vida_thresh_color'], alpha=0.5)
                h1, = axB[i].plot(vida_times, y_gt, marker='.', ms=4, ls='-', lw=1, label=label_gt, c=PLOT_PARAMS['vida_gt_color'])
                handles.append(h1)
            if y_re is not None:
                h2, = axB[i].plot(vida_times, y_re, marker='.', ms=4, ls='-', lw=1, label=label_re, c=PLOT_PARAMS['vida_recon_color'])
                handles.append(h2)

            if ylim: axB[i].set_ylim(ylim)

            # Add pass/fail percentage to legend
            pass_val = pass_fail.get('pa_x_b', np.nan)
            handles.append(mpatches.Patch(color='none', label=f'Pass = {pass_val:.0f} %' if not np.isnan(pass_val) else 'Pass = N/A'))
            axB[i].legend(handles=handles, ncol=1, fontsize=ftsize, loc='lower left', frameon=False)

        else:
            axB[i].text(0.5, 0.5, "VIDA data unavailable", ha='center', va='center', fontsize=ftsize)
            axB[i].set_xticks([]); axB[i].set_yticks([])


        # --- Plot Autocorrelation (ax[i, 4] and ax[i, 5]) ---
        if mod != 'mring-varbeta2':
            autocorr_gt = ps_gt.get('autocorr')
            autocorr_re = ps_re.get('autocorr')
            ps_val_gt = ps_gt.get('ps')
            ps_val_re = ps_re.get('ps')
            ps_err_hi_gt = ps_gt.get('ps_err_hi')
            ps_err_lo_gt = ps_gt.get('ps_err_lo')
            ps_err_hi_re = ps_re.get('ps_err_hi')
            ps_err_lo_re = ps_re.get('ps_err_lo')

            if autocorr_gt is not None:
                 im = ax[i, 4].imshow(autocorr_gt, cmap=PLOT_PARAMS['autocorr_cmap'], origin='lower',
                                      extent=extent_ps, vmin=PLOT_PARAMS['autocorr_vmin'], vmax=PLOT_PARAMS['autocorr_vmax'])
                 ax[i, 4].set_aspect((extent_ps[1] - extent_ps[0]) / (extent_ps[3] - extent_ps[2]))
                 if mappable_ps is None: mappable_ps = im # Store for colorbar
                 # Add pattern speed text annotation
                 if all(v is not None for v in [ps_val_gt, ps_err_hi_gt, ps_err_lo_gt]):
                     t_gt = ax[i, 4].text(0.95, 0.05, fr'$\omega_p$ = {ps_val_gt:.2f}$^{{+{ps_err_hi_gt:.2f}}}_{{-{ps_err_lo_gt:.2f}}}$',
                                          ha='right', va='bottom', transform=ax[i, 4].transAxes, fontsize=ftsize-1)
                     t_gt.set_bbox(dict(facecolor='w', alpha=0.7, edgecolor='none', pad=0.1))


            if autocorr_re is not None:
                 ax[i, 5].imshow(autocorr_re, cmap=PLOT_PARAMS['autocorr_cmap'], origin='lower',
                                 extent=extent_ps, vmin=PLOT_PARAMS['autocorr_vmin'], vmax=PLOT_PARAMS['autocorr_vmax'])
                 ax[i, 5].set_aspect((extent_ps[1] - extent_ps[0]) / (extent_ps[3] - extent_ps[2]))
                 # Add pattern speed text annotation
                 if all(v is not None for v in [ps_val_re, ps_err_hi_re, ps_err_lo_re]):
                     t_re = ax[i, 5].text(0.95, 0.05, fr'$\omega_p$ = {ps_val_re:.2f}$^{{+{ps_err_hi_re:.2f}}}_{{-{ps_err_lo_re:.2f}}}$',
                                           ha='right', va='bottom', transform=ax[i, 5].transAxes, fontsize=ftsize-1)
                     t_re.set_bbox(dict(facecolor='w', alpha=0.7, edgecolor='none', pad=0.1))


            # Axis labels and ticks for autocorrelation plots
            ax[i, 4].set_ylabel('$\Delta$PA (deg)', fontsize=fsize)
            ax[i, 5].set_yticks([])
            if i < nmod - 1: # Hide x ticks except for the bottom row
                 ax[i, 4].set_xticklabels([])
                 ax[i, 5].set_xticklabels([])
                 axA[i].set_xticklabels([]) # Also hide for NxCorr/VIDA
                 axB[i].set_xticklabels([])
            else: # Set x labels only for the bottom row
                 axA[i].set_xlabel('UT time (hr)', fontsize=fsize)
                 axB[i].set_xlabel('UT time (hr)', fontsize=fsize)
                 ax[i, 4].set_xlabel('$\Delta$t (GM/c$^3$)', fontsize=fsize)
                 ax[i, 5].set_xlabel('$\Delta$t (GM/c$^3$)', fontsize=fsize)

        else: # If model is 'mring-varbeta2'
            ax[i, 4].remove()
            ax[i, 5].remove()
            # Still need to handle x-axis labels/ticks for NxCorr/VIDA
            if i < nmod - 1:
                 axA[i].set_xticklabels([])
                 axB[i].set_xticklabels([])
            else:
                 axA[i].set_xlabel('UT time (hr)', fontsize=fsize)
                 axB[i].set_xlabel('UT time (hr)', fontsize=fsize)


    # --- Titles for Columns ---
    axA[0].set_title('Cross Correlation', fontsize=fsize)
    axB[0].set_title(r'PA (deg) | RA ($\mu$as) | $\angle\beta_2$ (rad)', fontsize=fsize) # Combined title
    ax[0, 4].set_title('Ground Truth Autocorrelation', fontsize=fsize)
    ax[0, 5].set_title('Reconstruction Autocorrelation', fontsize=fsize)

    # --- Autocorrelation Colorbar ---
    if mappable_ps:
        # Position below the plots
        cax = fig.add_axes([0.64, 0.06, 0.26, 0.015]) # Adjust position/size as needed
        setup_colorbar(fig, mappable_ps, cax.get_position(), 'Autocorrelation', fsize, pow_limits=(0, 0), use_math_text=True)
        # Make the colorbar horizontal
        cbar = plt.colorbar(mappable_ps, cax=cax, orientation='horizontal')
        cbar.set_label(label='Autocorrelation', fontsize=fsize)
        cbar.ax.tick_params(labelsize=ftsize)


    # --- Final Adjustments ---
    plt.subplots_adjust(wspace=0.3, hspace=0.1, bottom=0.15, top=0.95) # Adjust spacing

    # Adjust subplot positions (may need tweaking)
    # This part is tricky to get perfect automatically. Manual adjustment might be best.
    # Example: Fine-tune horizontal position of NxCorr/VIDA columns
    # for i in range(nmod):
    #     boxA = axA[i].get_position()
    #     boxB = axB[i].get_position()
    #     # Shift right slightly? Needs trial and error.
    #     # boxA.x0 += 0.02; boxA.x1 += 0.02
    #     # boxB.x0 += 0.02; boxB.x1 += 0.02
    #     axA[i].set_position(boxA)
    #     axB[i].set_position(boxB)


    # --- Save and Close ---
    save_plot(fig, output_filename)
    plt.close(fig)
    print("Dynamic validation ladder plot done.")


# --- Extra Tests Plot (Similar to Ladder) ---
def plot_dynamic_extra(models_subset, data, params, output_filename):
    """Plots the extra tests comparison (similar structure to ladder plot)."""
    print("Plotting dynamic extra tests...")
    # This function is very similar to plot_dynamic_ladder
    # We can potentially combine them by passing the model list and output filename
    # For now, keeping it separate for clarity, reusing ladder logic.

    nmod = len(models_subset)
    if nmod == 0:
        print("No models specified for extra tests plot. Skipping.")
        return

    # --- Plotting Setup ---
    fbsize, fsize, ftsize = params['fbsize'], params['fsize'], params['ftsize']
    extent_ps = params['extent_ps']

    fig, ax = plt.subplots(nmod, 6, figsize=(16, 2.8 * nmod), # Height scales with num models
                           gridspec_kw={'height_ratios': [1] * nmod,
                                        'width_ratios': [1, 1, 1, 1, 1, 1]})
    gs = ax[0, 0].get_gridspec()
    axA = [] # List to hold NxCorr axes
    axB = [] # List to hold VIDA axes

    mappable_ps = None # For autocorrelation colorbar

    # --- Plotting Loop ---
    for i, mod in enumerate(models_subset):
        # Extract data for the model
        nxcorr = data['nxcorr'].get(mod, {})
        vida = data['vida'].get(mod, {})
        ps_gt = data['pattern_speed']['gt'].get(mod, {})
        ps_re = data['pattern_speed']['re'].get(mod, {})
        pass_fail = data['pass_fail'].get(mod, {})

        # --- Merge Axes ---
        for j in range(4): ax[i, j].remove()
        axA.append(fig.add_subplot(gs[i, 0:2]))
        axB.append(fig.add_subplot(gs[i, 2:4]))
        axA[i].set_ylabel(mod, fontsize=fbsize)

        # --- Plot NxCorr --- (Only I for these models based on original last cell)
        plot_times = nxcorr.get('time')
        handles = []
        if plot_times is not None:
            h1, = axA[i].plot(plot_times, nxcorr.get('thrI'), label='threshold I', c=PLOT_PARAMS['nxcorr_thresh_color'])
            h2, = axA[i].plot(plot_times, nxcorr.get('recI'), lw=2, label='reconstruction I', c=PLOT_PARAMS['nxcorr_recon_color'])
            handles.extend([h1, h2])
            passI = pass_fail.get('nxcorrI', np.nan)
            handles.append(mpatches.Patch(color='none', label=f'Pass I = {passI:.0f} %' if not np.isnan(passI) else 'Pass I = N/A'))
            axA[i].legend(handles=handles, ncol=1, fontsize=ftsize, loc='lower left', frameon=False)
            axA[i].set_ylim(0.01, 1.05)
        else:
             axA[i].text(0.5, 0.5, "NxCorr data unavailable", ha='center', va='center', fontsize=ftsize)
             axA[i].set_xticks([]); axA[i].set_yticks([])

        # --- Plot VIDA --- (Assuming PA for all extra tests)
        vida_times = vida.get('time')
        handles = []
        if vida_times is not None:
            theta_re_key = 'thetare_wrapped' if 'thetare_wrapped' in vida else 'thetare'
            y_gt, y_re = vida.get('thetagt'), vida.get(theta_re_key)
            label_gt, label_re = r'PA ground truth', r'PA reconstruction'
            thresh_val = 20 # degrees threshold
            ylim = (-199, 199)

            if y_gt is not None:
                axB[i].fill_between(vida_times, y_gt - thresh_val, y_gt + thresh_val, color=PLOT_PARAMS['vida_thresh_color'], alpha=0.5)
                h1, = axB[i].plot(vida_times, y_gt, marker='.', ms=4, ls='-', lw=1, label=label_gt, c=PLOT_PARAMS['vida_gt_color'])
                handles.append(h1)
            if y_re is not None:
                h2, = axB[i].plot(vida_times, y_re, marker='.', ms=4, ls='-', lw=1, label=label_re, c=PLOT_PARAMS['vida_recon_color'])
                handles.append(h2)

            axB[i].set_ylim(ylim)
            pass_val = pass_fail.get('pa_x_b', np.nan)
            handles.append(mpatches.Patch(color='none', label=f'Pass = {pass_val:.0f} %' if not np.isnan(pass_val) else 'Pass = N/A'))
            axB[i].legend(handles=handles, ncol=1, fontsize=ftsize, loc='lower left', frameon=False)
        else:
            axB[i].text(0.5, 0.5, "VIDA data unavailable", ha='center', va='center', fontsize=ftsize)
            axB[i].set_xticks([]); axB[i].set_yticks([])

        # --- Plot Autocorrelation --- (Assuming not varbeta2)
        autocorr_gt = ps_gt.get('autocorr')
        autocorr_re = ps_re.get('autocorr')
        ps_val_gt = ps_gt.get('ps')
        ps_val_re = ps_re.get('ps')
        ps_err_hi_gt = ps_gt.get('ps_err_hi')
        ps_err_lo_gt = ps_gt.get('ps_err_lo')
        ps_err_hi_re = ps_re.get('ps_err_hi')
        ps_err_lo_re = ps_re.get('ps_err_lo')

        if autocorr_gt is not None:
             im = ax[i, 4].imshow(autocorr_gt, cmap=PLOT_PARAMS['autocorr_cmap'], origin='lower',
                                  extent=extent_ps, vmin=PLOT_PARAMS['autocorr_vmin'], vmax=PLOT_PARAMS['autocorr_vmax'])
             ax[i, 4].set_aspect((extent_ps[1] - extent_ps[0]) / (extent_ps[3] - extent_ps[2]))
             if mappable_ps is None: mappable_ps = im
             if all(v is not None for v in [ps_val_gt, ps_err_hi_gt, ps_err_lo_gt]):
                 t_gt = ax[i, 4].text(0.95, 0.05, fr'$\omega_p$ = {ps_val_gt:.2f}$^{{+{ps_err_hi_gt:.2f}}}_{{-{ps_err_lo_gt:.2f}}}$',
                                      ha='right', va='bottom', transform=ax[i, 4].transAxes, fontsize=ftsize-1)
                 t_gt.set_bbox(dict(facecolor='w', alpha=0.7, edgecolor='none', pad=0.1))

        if autocorr_re is not None:
             ax[i, 5].imshow(autocorr_re, cmap=PLOT_PARAMS['autocorr_cmap'], origin='lower',
                             extent=extent_ps, vmin=PLOT_PARAMS['autocorr_vmin'], vmax=PLOT_PARAMS['autocorr_vmax'])
             ax[i, 5].set_aspect((extent_ps[1] - extent_ps[0]) / (extent_ps[3] - extent_ps[2]))
             if all(v is not None for v in [ps_val_re, ps_err_hi_re, ps_err_lo_re]):
                 t_re = ax[i, 5].text(0.95, 0.05, fr'$\omega_p$ = {ps_val_re:.2f}$^{{+{ps_err_hi_re:.2f}}}_{{-{ps_err_lo_re:.2f}}}$',
                                       ha='right', va='bottom', transform=ax[i, 5].transAxes, fontsize=ftsize-1)
                 t_re.set_bbox(dict(facecolor='w', alpha=0.7, edgecolor='none', pad=0.1))

        # --- Axis labels and ticks ---
        ax[i, 4].set_ylabel('$\Delta$PA (deg)', fontsize=fsize)
        ax[i, 5].set_yticks([])
        if i < nmod - 1:
             ax[i, 4].set_xticklabels([])
             ax[i, 5].set_xticklabels([])
             axA[i].set_xticklabels([])
             axB[i].set_xticklabels([])
        else:
             axA[i].set_xlabel('UT time (hr)', fontsize=fsize)
             axB[i].set_xlabel('UT time (hr)', fontsize=fsize)
             ax[i, 4].set_xlabel('$\Delta$t (GM/c$^3$)', fontsize=fsize)
             ax[i, 5].set_xlabel('$\Delta$t (GM/c$^3$)', fontsize=fsize)

    # --- Titles, Colorbar, Adjustments, Save --- (Same as ladder)
    axA[0].set_title('Cross Correlation', fontsize=fsize)
    axB[0].set_title(r'PA (deg)', fontsize=fsize) # Assuming PA for extras
    ax[0, 4].set_title('Ground Truth Autocorrelation', fontsize=fsize)
    ax[0, 5].set_title('Reconstruction Autocorrelation', fontsize=fsize)

    if mappable_ps:
        cax = fig.add_axes([0.64, 0.06, 0.26, 0.015])
        setup_colorbar(fig, mappable_ps, cax.get_position(), 'Autocorrelation', fsize, pow_limits=(0, 0), use_math_text=True)
        cbar = plt.colorbar(mappable_ps, cax=cax, orientation='horizontal')
        cbar.set_label(label='Autocorrelation', fontsize=fsize)
        cbar.ax.tick_params(labelsize=ftsize)

    plt.subplots_adjust(wspace=0.3, hspace=0.1, bottom=0.15, top=0.95)

    save_plot(fig, output_filename)
    plt.close(fig)
    print("Dynamic extra tests plot done.")


# ==============================================================================
# Main Execution Block
# ==============================================================================

def main():
    # --- 1. Calibrator ---
    cal_mov = load_movie(CALIBRATOR_FILE)
    if cal_mov:
         cal_img = cal_mov.im_list()[0]
         plot_calibrator(cal_img, CALIBRATOR_SLICE, PLOT_PARAMS['calibrator'],
                         os.path.join(BASE_FIGURES_PATH, 'jc1924'))
    else:
        print("Skipping calibrator plot due to loading error.")

    # --- 2. Static Models ---
    static_data = load_static_data(STATIC_MODELS, PIPELINE_NAME, MODEL_SLICE)
    plot_static_models(STATIC_MODELS, static_data, PLOT_PARAMS['static'],
                       os.path.join(BASE_FIGURES_PATH, 'staticmodels'))

    # --- 3. Dynamic Models ---
    dynamic_data = load_dynamic_data(DYNAMIC_MODELS, PIPELINE_NAME, DYNAMIC_FRAME_INDICES, MODEL_SLICE)

    # Individual ("Printed") Plots
    for mod in DYNAMIC_MODELS:
        plot_dynamic_model_individual(mod, dynamic_data, PLOT_PARAMS['dynamic_print'],
                                      os.path.join(BASE_FIGURES_PATH, '{mod}'))

    # Validation Ladder Plot
    plot_dynamic_ladder(DYNAMIC_MODELS_LADDER, dynamic_data, PLOT_PARAMS['dynamic_ladder'],
                       os.path.join(BASE_FIGURES_PATH, 'evaluation_all1'))

    # Extra Tests Plot
    plot_dynamic_extra(DYNAMIC_MODELS_EXTRA, dynamic_data, PLOT_PARAMS['dynamic_extra'],
                       os.path.join(BASE_FIGURES_PATH, 'evaluation_all2')) # Changed name to all2

    print("\nAll plotting tasks complete.")

if __name__ == "__main__":
    main()