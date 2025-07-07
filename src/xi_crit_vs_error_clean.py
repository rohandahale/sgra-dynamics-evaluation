'''
Simple cleaned script that optimizes xi_crit based on minimum RMSE in test problems. NC April 2025.

Takes in an imput table with xi_crit and pattern speed values for each method/model, calculates RMSE, prints/plots best xi_crit val.

Currently works on DAR style xi_crit table.

Things to fix before we can use on latest style table:
-Anywhere a directory is defined. E.g. whenever os.path.join is used, or np.load is used, and the definition of base_data_dir, output_dir_error, output_dir, table_name... [Partially addressed by rewrite, review other paths]
-the format for table in make_error_table_function(). It's set to work on the old input table format. In the new format, we'd change:
    xicrit_obs = np.round(float(table[i, 0]), 1)   # (it's now in collumn 0, not 2)
    and load omegap_truth from some other file, or from a dictionary. NC can write if need be [Kept as is, may need update]
-potentially generalize to include in the analysis script...
-Optional: for the cylinder_mcmc.py script, we need to measure the error in the width of bestbet xi_crit (I believe in flat units). the
 bestbet xi_crit uncertainty sigma is ~0.7, with wide error. NC can update the gaussian fit to get the xi_crit sigma error later if we want to update that sigma
-Optional: discuss what's the optimal set of models to do this optimization on (e.g. we probably don't want to use static geometric models for optimizing xi_crit)


'''

### Import
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.stats import truncnorm, norm, skewnorm
from scipy.optimize import curve_fit

### Define arrays and directories
reconstruction_array = [
    # 'truth', # Truth is handled separately now
    'kine',
    'resolve_mean', # Corresponds to 'resolve' in the path requirement
    'modeling_mean',
    'doghit',
    'ehtim',
    'ngmem'
    ]

model_name_array = [
#    'SGRA',                                                          ## real data                    -- Don't use to optimize xi_crit, as we don't have ground truth value
    'grmhd1', 'grmhd2', 'grmhd8',                                    ## validation ladder GRMHD
    'grmhd3', 'grmhd4', 'grmhd6', 'grmhd7',                          ## extra tests GRMHD
    'grmhd5',                                                        ## extra tests GRMHD edge-on
    'grmhd2+hs1', 'grmhd2+hs2',                                      ## GRMHD+HS
    #'mring+hsCCW',  'mring+hsCW',                                    ## validation ladder HS
    #'mring+hsCW40', 'mring+hsCW20',                                  ## extra tests HS speed limits
#    'mring+hsCW0.15', 'mring+hsCW0.60', 'mring+hsCW1.20'             ## extra tests HS flux limits   -- Potentially don't use, as extremely dim/bright HS are quite different from data
#    'crescent', 'ring',                                              ## static geometric models      -- Don't use, as static geometric models are quite different from data
    ]

# NOTE: flat_threshold is no longer used to determine the input filename based on the prompt requirements.
#       The script now specifically looks for 'pattern_speed_vs_xicrit_stdunits.npy'.
#       Set flat_threshold = 0 for xi_crit units in output label, keep as 1 if using flat units elsewhere.
flat_threshold = 0 # if 1, xi_crit is e.g. 0.8; if 0, it's e.g. 0.8 * STD(autocorrelation)

# --- User Configuration ---
# Base directory where model/band/method specific folders reside
base_data_dir = '/home/share/SgrA_Dynamics/evaluation/april11/20250501_patternspeed_survery_results/patternspeed/' # <<< --- UPDATE THIS PATH --- <<<

# Output directory for the error summary table and plots
output_base_dir = '/home/share/SgrA_Dynamics/evaluation/april11/20250501_patternspeed_survery_results/patternspeed/' # <<< --- UPDATE THIS PATH --- <<<
output_dir_error = os.path.join(output_base_dir, 'error_summary')
# --- End User Configuration ---

### Define Functions
def make_error_table_function(reconstruction_array, model_name_array, base_data_dir, save_error_table=False, output_dir_error_path=None):
    """
    Generates a table comparing observed pattern speeds from reconstructions
    against truth values for various models, methods, and xi_crit thresholds.

    Args:
        reconstruction_array (list): List of reconstruction method names.
        model_name_array (list): List of model names.
        base_data_dir (str): The base directory containing the data folders.
        save_error_table (bool): If True, saves the generated table to a .npy file.
        output_dir_error_path (str): Path to the directory to save the error table.

    Returns:
        numpy.ndarray: An array containing error metrics for each valid combination.
                       Columns: [reconstruction_method, model_name, omegap_truth,
                                 omegap_obs, error, percent_error, xicrit_obs,
                                 flat_threshold_setting]
    """
    error_table = []

    for reconstruction_method in reconstruction_array:
        # Determine the correct band based on the reconstruction method
        if reconstruction_method == 'resolve_mean' or reconstruction_method == 'modeling_mean': # Check if this is the correct name for 'resolve'
            band = 'LO+HI'
        else:
            band = 'LO'

        for model_name in model_name_array:
            try:
                # Construct the path according to the new structure
                data_folder = f'{model_name}_{band}_onsky_{reconstruction_method}'
                data_dir = os.path.join(base_data_dir, data_folder)
                table_name = os.path.join(data_dir, 'pattern_speed_vs_xicrit_stdunits.npy')

                # Load the pattern speed vs xi_crit table for the reconstruction
                table = np.load(table_name)

            except FileNotFoundError:
                print(f'(*) Reconstruction: {reconstruction_method}, Model: {model_name}. FILE NOT FOUND at: {table_name}')
                continue
            except Exception as e:
                print(f'(*) Reconstruction: {reconstruction_method}, Model: {model_name}. Error loading file {table_name}: {e}')
                continue

            if model_name != 'SGRA': # SGRA doesn't have a ground truth
                try:
                    # --- Determine path for ground truth data ---
                    # NOTE: This path might need adjustment based on your actual truth data structure.
                    # Assuming truth data follows a similar pattern or is in a specific 'truth' folder.
                    # Option 1: Assume truth is in a folder like '<model_name>_truth_cylinder_output' (original logic)
                    # truth_output_dir = os.path.join(base_data_dir, f'{model_name}_truth_cylinder_output')
                    # truth_table_name = os.path.join(truth_output_dir, 'cylinder_data.npy')

                    # Option 2: Assume truth is in a folder like '<model_name>_LO_onsky_truth' (matching new structure)
                    truth_data_folder = f'{model_name}_LO_onsky_truth' # Assuming truth is always LO band? Adjust if needed.
                    truth_output_dir = os.path.join(base_data_dir, truth_data_folder)
                    # Adjust the filename for truth data if necessary
                    truth_table_name = os.path.join(truth_output_dir, 'pattern_speed_vs_xicrit_stdunits.npy') # Or cylinder_data.npy?

                    # Load the ground truth data
                    truth_table = np.load(truth_table_name)
                    # Assuming truth value is constant regardless of xi_crit for the truth run
                    # Check indices based on your truth table format: [row, column]
                    # Original used truth_table[0, 1]
                    omegap_truth = truth_table[0, 1] # <<< --- VERIFY THIS INDEX --- <<<

                except FileNotFoundError:
                    print(f'(*) Reconstruction: {reconstruction_method}, Model: {model_name}. TRUTH FILE NOT FOUND at: {truth_table_name}')
                    continue
                except Exception as e:
                    print(f'(*) Reconstruction: {reconstruction_method}, Model: {model_name}. Error loading TRUTH file {truth_table_name}: {e}')
                    continue


                for i in range(table.shape[0]):
                    # Extract observed pattern speed and xi_crit
                    # Check indices based on your table format: [row, column]
                    # Original used table[i, 1] for omegap_obs and table[i, 2] for xicrit_obs
                    # If format changed (see comment in original script), update indices here.
                    omegap_obs = table[i, 1] # <<< --- VERIFY THIS INDEX --- <<<
                    # Assuming xi_crit is now in column 0 as per original comment:
                    # xicrit_obs = np.round(float(table[i, 0]), 1) # <<< --- VERIFY THIS INDEX --- <<<
                    # Using original index 2 for now:
                    xicrit_obs = np.round(float(table[i, 2]), 1) # <<< --- VERIFY THIS INDEX --- <<<


                    # Calculate errors
                    try:
                        error = float(omegap_obs) - float(omegap_truth)
                        percent_error = error / float(omegap_truth) if float(omegap_truth) != 0 else np.inf
                    except ValueError:
                         print(f'(*) Reconstruction: {reconstruction_method}, Model: {model_name}. Could not convert omegap values to float.')
                         error = np.nan
                         percent_error = np.nan


                    error_row = [reconstruction_method, model_name, omegap_truth, omegap_obs, error, percent_error, xicrit_obs, flat_threshold]
                    error_table.append(error_row)

    error_table = np.array(error_table)

    if save_error_table and output_dir_error_path:
        os.makedirs(output_dir_error_path, exist_ok=True)
        output_file = os.path.join(output_dir_error_path, 'error_table.npy')
        try:
            np.save(output_file, error_table)
            print(f'Error table saved to {output_file}')
        except Exception as e:
            print(f"Error saving error table to {output_file}: {e}")
    elif save_error_table and not output_dir_error_path:
        print("Warning: save_error_table is True, but output_dir_error_path is not provided. Table not saved.")


    return error_table

def truncnorm_gauss(x, mu, sigma, A):
    """Truncated Gaussian function."""
    a_trunc = 0 # Define truncation bounds if needed, currently [0, 1]
    b_trunc = 1
    a, b = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
    return A * truncnorm.pdf(x, a, b, loc=mu, scale=sigma)

def fit_truncnorm_gauss(x_data, y_data):
    """Fits a truncated Gaussian to the data (currently unused)."""
    # popt, pcov = curve_fit(truncnorm_gauss, bin_centers, n, p0=[0, 0.1, 1000])
    # shift_y_data = np.max(y_data)
    # y_data = -1*y_data + 2*shift_y_data ## flip y_data about mean so it's positive/concave
    y_data = [1/y for y in y_data] # Example transformation (inverse)

    try:
        popt, pcov = curve_fit(truncnorm_gauss, x_data, y_data, p0=[0.5, 0.25, 1])
        print("Fit parameters (mu, sigma, A):", popt)
        return popt, pcov
    except Exception as e:
        print(f"Could not fit truncated Gaussian: {e}")
        return None, None


def analyze_error_table(error_table, save_plot=True, output_dir_plot=None):
    """
    Analyzes the error table to find optimal xi_crit based on minimum RMSE
    and plots RMSE vs xi_crit for different methods.

    Args:
        error_table (numpy.ndarray): The error table generated by make_error_table_function.
        save_plot (bool): If True, saves the generated plot.
        output_dir_plot (str): Path to the directory to save the plot.
    """
    if error_table.size == 0:
        print("Error table is empty. Cannot perform analysis.")
        return

    ### Define arrays from table columns (adjust indices if table format changed)
    try:
        xicrit_obs_values = error_table[:, 6].astype(float)
        errors = error_table[:, 4].astype(float) # Use absolute error for RMSE
        models = error_table[:, 1]
        reconstruction_methods = error_table[:, 0]
        # percent_errors = error_table[:, 5].astype(float) # Currently unused
    except IndexError:
        print("Error accessing columns in error_table. Check table format and indices.")
        return
    except ValueError:
        print("Error converting error_table columns to float. Check for non-numeric data.")
        return


    unique_xicrit_obs = np.unique(xicrit_obs_values)
    # Use reconstruction methods present in the table
    unique_reconstruction_methods = np.unique(reconstruction_methods)

    ### Calculate RMSE for all models/methods combined
    xicrit_obs_array = []
    RMSE_allmodels_array = []

    for xicrit_obs in unique_xicrit_obs:
        # Find indices where xi_crit matches and error is not NaN
        indices = np.where((xicrit_obs_values == xicrit_obs) & (~np.isnan(errors)))[0]
        if len(indices) > 0:
            errors_for_xicrit = errors[indices]
            RMSE = np.sqrt(np.mean(errors_for_xicrit**2))
            xicrit_obs_array.append(xicrit_obs)
            RMSE_allmodels_array.append(RMSE)
        # else: # Skip xi_crit if no valid data points exist
        #     print(f"No valid data points for xi_crit = {xicrit_obs}")

    if not xicrit_obs_array:
        print("No valid data points found across all xi_crit values. Cannot plot average.")
        return

    xicrit_obs_array = np.array(xicrit_obs_array)
    RMSE_allmodels_array = np.array(RMSE_allmodels_array)

    # Find minimum RMSE for the overall average
    min_index_all = np.argmin(RMSE_allmodels_array)
    min_xicrit_all = xicrit_obs_array[min_index_all]
    min_RMSE_all = RMSE_allmodels_array[min_index_all]

    ### Start plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    ### Plot overall average first
    ax.plot(xicrit_obs_array, RMSE_allmodels_array, label=f'Average (min at {min_xicrit_all:.2f})', color='black', linewidth=2, zorder=10)
    ax.scatter(min_xicrit_all, min_RMSE_all, color='black', s=60, zorder=11, marker='*')

    print('\n~~~ Results! ~~~')
    print(f"Overall minimum RMSE: {min_RMSE_all:.3f} at xi_crit = {min_xicrit_all:.3f}")

    ### Now plot each method individually
    method_results = {}
    for reconstruction_method in unique_reconstruction_methods:
        xicrit_method_array = []
        RMSE_method_array = []

        for xicrit_obs in unique_xicrit_obs:
            # Find indices for this specific method and xi_crit, excluding NaNs
            indices = np.where(
                (xicrit_obs_values == xicrit_obs) &
                (reconstruction_methods == reconstruction_method) &
                (~np.isnan(errors))
            )[0]

            if len(indices) > 0:
                errors_for_method_xicrit = errors[indices]
                RMSE = np.sqrt(np.mean(errors_for_method_xicrit**2))
                xicrit_method_array.append(xicrit_obs)
                RMSE_method_array.append(RMSE)
            # else: # Optionally skip if no data for this specific method/xi_crit
                 #pass

        if not xicrit_method_array: # Skip method if it has no valid data points
            print(f"No valid data points found for method: {reconstruction_method}")
            continue

        xicrit_method_array = np.array(xicrit_method_array)
        RMSE_method_array = np.array(RMSE_method_array)

        # Find minimum for this method
        min_index = np.nanargmin(RMSE_method_array)
        min_xicrit = xicrit_method_array[min_index]
        min_RMSE = RMSE_method_array[min_index]

        # Store results
        method_results[reconstruction_method] = {'min_rmse': min_RMSE, 'min_xicrit': min_xicrit}

        # Plot
        line, = ax.plot(xicrit_method_array, RMSE_method_array, linestyle='--', label=f'{reconstruction_method} (min at {min_xicrit:.2f})')
        ax.scatter(min_xicrit, min_RMSE, color=line.get_color(), s=40, zorder=5)

        print(f"{reconstruction_method}: minimum RMSE = {min_RMSE:.3f} at xi_crit = {min_xicrit:.3f}")

    ### Final plot formatting
    if flat_threshold == 1:
        ax.set_xlabel(r'$\xi_{crit}$ [$\xi$]') # Using LaTeX formatting
    else:
        ax.set_xlabel(r'$\xi_{crit}$ [$\sigma_{\xi}$]') # Using LaTeX formatting
    ax.set_ylabel('RMSE [Deg per M]') # Adjust units if necessary
    ax.set_title(r'RMSE vs $\xi_{crit}$ for Different Reconstruction Methods') # Using LaTeX formatting
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    if save_plot and output_dir_plot:
        os.makedirs(output_dir_plot, exist_ok=True)
        plot_filename = 'RMSE_vs_xicrit_std_threshold.png' if flat_threshold == 0 else 'RMSE_vs_xicrit_flat_threshold.png'
        plot_filepath = os.path.join(output_dir_plot, plot_filename)
        try:
            plt.savefig(plot_filepath, bbox_inches='tight')
            print(f"Plot saved to {plot_filepath}")
        except Exception as e:
            print(f"Error saving plot to {plot_filepath}: {e}")
        plt.close(fig) # Close the plot figure after saving
    elif save_plot and not output_dir_plot:
        print("Warning: save_plot is True, but output_dir_plot is not provided. Plot not saved.")
        plt.show() # Show plot interactively if not saving
    else:
        plt.show() # Show plot interactively if save_plot is False

    return method_results


### Main execution block
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(output_dir_error, exist_ok=True)

    ### Option 1: Generate new error table
    generate_new_table = True # Set to False to load existing table
    if generate_new_table:
        print("Generating new error table...")
        error_table = make_error_table_function(
            reconstruction_array=reconstruction_array,
            model_name_array=model_name_array,
            base_data_dir=base_data_dir,
            save_error_table=True,
            output_dir_error_path=output_dir_error # Pass the specific path here
        )
    ### Option 2: Load existing error table
    else:
        error_table_path = os.path.join(output_dir_error, 'error_table.npy')
        try:
            error_table = np.load(error_table_path, allow_pickle=True) # allow_pickle needed if saving objects/strings
            print(f'Loaded error table from {error_table_path}')
            # Check if loaded table is empty
            if error_table.size == 0:
                 print("Loaded error table is empty. Trying to generate a new one.")
                 error_table = make_error_table_function(
                     reconstruction_array=reconstruction_array,
                     model_name_array=model_name_array,
                     base_data_dir=base_data_dir,
                     save_error_table=True,
                     output_dir_error_path=output_dir_error
                 )

        except FileNotFoundError:
            print(f'Error table not found at {error_table_path}. Generating a new one.')
            error_table = make_error_table_function(
                reconstruction_array=reconstruction_array,
                model_name_array=model_name_array,
                base_data_dir=base_data_dir,
                save_error_table=True,
                output_dir_error_path=output_dir_error
            )
        except Exception as e:
            print(f"Error loading error table from {error_table_path}: {e}. Exiting.")
            exit() # Exit if table cannot be loaded or generated


    ### Analyze loaded/generated error table
    if 'error_table' in locals() and error_table.size > 0:
         print("\nAnalyzing error table...")
         # Specify the directory to save the plot (can be the same as error table dir or different)
         plot_output_dir = os.path.join(output_dir_error, 'plots') # Example: save plots in a subfolder
         results = analyze_error_table(error_table, save_plot=True, output_dir_plot=plot_output_dir)
         print("\nAnalysis complete.")
    else:
        print("\nNo valid error table available to analyze.")

###
###
###