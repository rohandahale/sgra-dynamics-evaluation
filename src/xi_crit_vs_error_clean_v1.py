### 
'''
Simple cleaned script that optimizes xi_crit based on minimum RMSE in test problems. NC April 2025.

Takes in an imput table with xi_crit and pattern speed values for each method/model, calculates RMSE, prints/plots best xi_crit val. 

Currently works on DAR style xi_crit table. 

Things to fix before we can use on latest style table:
-Anywhere a directory is defined. E.g. whenever os.path.join is used, or np.load is used, and the definition of base_data_dir, output_dir_error, output_dir, table_name... 
-the format for table in make_error_table_function(). It's set to work on the old input table format. In the new format, we'd change:
    xicrit_obs = np.round(float(table[i, 0]), 1)   # (it's now in collumn 0, not 2)
    and load omegap_truth from some other file, or from a dictionary. NC can write if need be
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
    #'truth',
    'kine',
    'resolve_mean',
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
    'mring+hsCCW',  'mring+hsCW', 'mring+hs-not-center',                                    ## validation ladder HS 
    'mring+hsCW40', 'mring+hsCW20',                                  ## extra tests HS speed limits
#    'mring+hsCW0.15', 'mring+hsCW0.60', 'mring+hsCW1.20'             ## extra tests HS flux limits   -- Potentially don't use, as extremely dim/bright HS are quite different from data 
#    'crescent', 'ring',                                              ## static geometric models      -- Don't use, as static geometric models are quite different from data
    ]

flat_threshold = 0  # if 1, xi_crit is e.g. 0.8; if 0, it's e.g. 0.8 * STD(autocorrelation)
base_data_dir = '/home/share/SgrA_Dynamics/evaluation/april11/20250501_patternspeed_survery_results/patternspeed/'

output_dir_error = '/home/share/SgrA_Dynamics/evaluation/april11/20250501_patternspeed_survery_results/patternspeed/'
output_dir_error += 'error_summary/'

### Define Functions
def make_error_table_function(reconstruction_array, model_name_array, save_error_table=False):
    error_table = []

    for reconstruction_method in reconstruction_array:
        if reconstruction_method == 'truth':
            print("Skipping 'truth' method for error calculation.")
            continue
        # Determine the correct band based on the reconstruction method
        if reconstruction_method == 'resolve_mean' or reconstruction_method == 'modeling_mean': # Check if this is the correct name for 'resolve'
            band = 'LO+HI'
        else:
            band = 'LO'
            
        for model_name in model_name_array:
            try:
                output_dir = os.path.join(base_data_dir, f'{model_name}_{band}_onsky_{reconstruction_method}')
                
                if flat_threshold == 1:
                    table_name = os.path.join(output_dir, 'cylinder_data_flat_threshold.npy')
                else:
                    table_name = os.path.join(output_dir, 'pattern_speed_vs_xicrit_stdunits.npy')
                    
                table = np.load(table_name)

            except FileNotFoundError:
                print(f'(*) Reconstruction_method: {reconstruction_method}. Model: {model_name}. CANT FIND FILE! at', output_dir )
                continue

            if model_name != 'SGRA':
            
                truth_output_dir = os.path.join(base_data_dir, f'{model_name}_LO_onsky_truth')
                truth_table_name = os.path.join(truth_output_dir, 'pattern_speed_vs_xicrit_stdunits.npy')
                truth_table = np.load(truth_table_name)
                
                for i in range(table.shape[1]):
                    if 'hs' in model_name:
                        xi_crit_bestbet = 2.0
                        index_bestbet = np.where(np.round(truth_table[0],2) == xi_crit_bestbet)[0][0]
                        omegap_truth = truth_table[1][index_bestbet]
                    else: ## GRMHD
                        xi_crit_bestbet = 3.0
                        index_bestbet = np.where(np.round(truth_table[0],2) == xi_crit_bestbet)[0][0]
                        omegap_truth = truth_table[1][index_bestbet]
                    
                    #omegap_truth = truth_table[0, 1]
                    omegap_obs = table[1][i]
                    xicrit_obs = np.round(float(table[0][i]), 1)

                    error = abs(float(omegap_obs) - float(omegap_truth))
                    percent_error = error / float(omegap_truth)
                    

                    error_row = [reconstruction_method, model_name, omegap_truth, omegap_obs, error, percent_error, xicrit_obs, flat_threshold]
                    error_table.append(error_row)

    error_table = np.array(error_table)

    if save_error_table:
        output_dir_error = os.path.join(base_data_dir, 'error_summary')
        os.makedirs(output_dir_error, exist_ok=True)
        
        output_file = os.path.join(output_dir_error, 'error_table.npy')
        np.save(output_file, error_table)
        print(f'Error table saved to {output_file}')
        
    return error_table
    
def truncnorm_gauss(x, mu, sigma, A):
    a_trunc = 0
    b_trunc = 1
    a, b = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
    return A * truncnorm.pdf(x, a, b, loc=mu, scale=sigma)

def fit_truncnorm_gauss(x_data, y_data):
    # popt, pcov = curve_fit(truncnorm_gauss, bin_centers, n, p0=[0, 0.1, 1000])
    # shift_y_data = np.max(y_data)
    # y_data = -1*y_data + 2*shift_y_data ## flip y_data about mean so it's positive/concave
    y_data = [1/y for y in y_data]
    
    popt, pcov = curve_fit(truncnorm_gauss, x_data, y_data, p0=[0.5, 0.25, 1])
    print(popt)

def analyze_error_table(error_table, save_plot=True):
    ### Define arrays
    xicrit_obs_values = error_table[:, 6].astype(float) 
    percent_errors = error_table[:, 5].astype(float)     
    errors = error_table[:, 4].astype(float) 
    models = error_table[:, 1]
    reconstruction_methods = error_table[:, 0]

    unique_xicrit_obs = np.unique(xicrit_obs_values)
    unique_reconstruction_methods = ['modeling_mean', 'resolve_mean', 'doghit', 'ehtim', 'kine', 'ngmem']

    ### Define RMSE  for all models/methods
    xicrit_obs_array = []
    RMSE_allmodels_array = []

    for xicrit_obs in unique_xicrit_obs:
        indices = np.where(xicrit_obs_values == xicrit_obs)[0]
        errors_for_model = errors[indices]
        RMSE = np.sqrt(np.mean(errors_for_model**2))
        xicrit_obs_array.append(xicrit_obs)
        RMSE_allmodels_array.append(RMSE)

    xicrit_obs_array = np.array(xicrit_obs_array)
    RMSE_allmodels_array = np.array(RMSE_allmodels_array)

    # Find minimum for all models/methods
    min_index_all = np.argmin(RMSE_allmodels_array)
    min_xicrit_all = xicrit_obs_array[min_index_all]
    min_RMSE_all = RMSE_allmodels_array[min_index_all]

    ### Start plotting
    fig = plt.figure(figsize=(10, 8))

    ### Plot overall average first
    plt.plot(xicrit_obs_array, RMSE_allmodels_array, label=f'Average (min at {min_xicrit_all:.2f})', color='black', linewidth=2)
    plt.scatter(min_xicrit_all, min_RMSE_all, color='black', zorder=3)

    print('~~~ Results! ~~~')
    print(f"Overall minimum RMSE: {min_RMSE_all:.3f} at xi_crit = {min_xicrit_all:.3f}")

    ### Now plot each method individually
    for reconstruction_method in unique_reconstruction_methods:
        xicrit_method_array = []
        RMSE_method_array = []

        for xicrit_obs in unique_xicrit_obs:
            indices = np.where((xicrit_obs_values == xicrit_obs) & (reconstruction_methods == reconstruction_method))[0]
            errors_for_model = errors[indices]

            if len(errors_for_model) > 0:
                RMSE = np.sqrt(np.mean(errors_for_model**2))
            else:
                RMSE = np.nan  # In case there are no models for this xi_crit and method

            xicrit_method_array.append(xicrit_obs)
            RMSE_method_array.append(RMSE)

        xicrit_method_array = np.array(xicrit_method_array)
        RMSE_method_array = np.array(RMSE_method_array)

        # Find minimum for this method
        min_index = np.nanargmin(RMSE_method_array)
        min_xicrit = xicrit_method_array[min_index]
        min_RMSE = RMSE_method_array[min_index]

        plt.plot(xicrit_method_array, RMSE_method_array, linestyle='--', label=f'{reconstruction_method} (min at {min_xicrit:.2f})')
        plt.scatter(min_xicrit, min_RMSE, zorder=3)

        print(f"{reconstruction_method}: minimum RMSE = {min_RMSE:.3f} at xi_crit = {min_xicrit:.3f}")

    ### Final plot formatting
    if flat_threshold == 1:
        plt.xlabel(r'$\xi _{crit}$  [$\xi$]')
    else:
        plt.xlabel(r'$\xi _{crit}$  [$\sigma_\xi$]')
    plt.ylabel('RMSE [Deg per M]')
    plt.title(r'RMSE vs $\xi_{crit}$ for Different Methods')
    plt.legend()

    if save_plot:
        if flat_threshold == 1:
            plt.savefig(output_dir_error + 'error_by_method/all_methods_combined_flat_threshold.png', bbox_inches='tight')
        else:
            plt.savefig(output_dir_error + 'error_by_method/all_methods_combined_std_threshold.png', bbox_inches='tight')

    plt.close(fig)


### Now run functions
### first load old or generate new error table
generate_new_table = True 
if generate_new_table:
    error_table = make_error_table_function(
        reconstruction_array=reconstruction_array,
        model_name_array=model_name_array,
        save_error_table=True
    )
else:
    error_table_path = os.path.join(base_data_dir, 'error_summary', 'error_table.npy')
    error_table = np.load(error_table_path)
    print(f'Loaded error table from {error_table_path}')

### Analyze loaded error table
analyze_error_table(error_table, save_plot=True)

###
###
###







