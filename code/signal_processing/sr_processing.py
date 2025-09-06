import numpy as np
import pandas as pd 
import pickle
from sklearn.preprocessing import StandardScaler
from typing import List, Dict,Union, Tuple
from scipy.ndimage import gaussian_filter1d
import utils.data_extraction as datrac



def add_epsilon(array, epsilon=1e-10):
    """Add a small number to prevent zeros."""
    return array + epsilon

def get_corr_mat(data,epsilon=1e-5):
    v = add_epsilon(data, epsilon=epsilon)
    correlation_matrix = np.corrcoef(v)
    return correlation_matrix


    
def spike_rate_zscore(data):
    scaler = StandardScaler()
    data_T = data.T 
    z_scored_data_T = scaler.fit_transform(data_T)
    z_scored_data = z_scored_data_T.T
    return z_scored_data

def get_spikerate_data(filename):
    loaded_data = datrac.load_pickle(filename)
    bin_counts = loaded_data['bin_counts']
    bin_centers = bin_counts[-2]
    bin_edges  = bin_counts[-1]
    bin_counts = np.array(bin_counts[:-2])
    data_dic ={'s_counts':bin_counts,
              'bin_centers':bin_centers,
              'bin_edges':bin_edges}
    return data_dic


def calculate_trial_averages(data_collection):
    """
    Calculates the trial-averaged firing rates across axis 0.
    This is the fastest method when all input arrays have the same shape.

    Args:
        data_collection (dict): A dictionary where each value is another
                                dictionary containing 'raw_fr' and 'zscored_fr' arrays
                                of identical shape.

    Returns:
        dict: A dictionary with the trial-averaged 2D arrays for raw and z-scored data.
    """
    # 1. Collect all the arrays into lists
    raw_fr_list = [d['raw_fr'] for d in data_collection.values()]
    zscored_fr_list = [d['zscored_fr'] for d in data_collection.values()]

    # Handle the case where the input dictionary is empty
    if not raw_fr_list:
        return {'avg_raw_fr': np.array([]), 'avg_zscored_fr': np.array([])}

    # 2. Stack the lists into 3D arrays and average across the new 'trials' axis (axis=0)
    # This entire operation is done in highly optimized C code by NumPy.
    avg_raw_fr = np.mean(np.stack(raw_fr_list, axis=0), axis=0)
    avg_zscored_fr = np.mean(np.stack(zscored_fr_list, axis=0), axis=0)

    # 3. Return the final 2D averaged arrays
    results = {
        'raw_fr': avg_raw_fr,
        'zscored_fr': avg_zscored_fr
    }
    
    return results

def trim_binsz_ts(data_set):
    corrected_data_set = {}
    min_timestamp_len = min([d['bin_centers'].shape[0] for d in data_set.values()])
    for key, inner_dict in data_set.items():
        corrected_data_set[key] = {
            # Slice the 1D timestamp array
            'bin_centers': inner_dict['bin_centers'][:min_timestamp_len],

            # Slice the 2D firing rate arrays on their second dimension
            's_counts': inner_dict['s_counts'][:, :min_timestamp_len],
            'bin_edges': inner_dict['bin_edges'][:min_timestamp_len+1]
        }
    return corrected_data_set

#extract_dic_maps
def get_index_maps(df_trials):
    df_trials = df_trials.fillna(7777)
    df_trials['orientation'] = df_trials['orientation'].astype(int)
    df_trials['posX'] = df_trials['posX'].astype(int)
    df_trials['posY'] = df_trials['posY'].astype(int)
    columns_of_interest = ['orientation', 'posX', 'posY']
    grouped = df_trials.groupby(columns_of_interest)
    unique_combinations = list(grouped.groups.keys())
    num_unique_combinations = len(unique_combinations)
    index_map = {name: group.index.tolist() for name, group in grouped}
    return index_map

def psth_cal(trial_arrays):
    stacked_trial_rates = np.stack(trial_arrays, axis=0)
    per_neuron_psth = np.mean(stacked_trial_rates, axis=0)
    return per_neuron_psth

def calculate_population_tuning_changes(
    curves1: np.ndarray,
    curves2: np.ndarray,
    coords: np.ndarray,
    use_smoothing: bool = True,
    smoothing_sigma: float = 1.0,
    circular: bool = False,
    stimulus_range: float = 180.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates tuning changes for a population of neurons in a vectorized way.

    Args:
        curves1: Array of shape (neurons, tuning points) for condition 1.
        curves2: Array of shape (neurons, tuning points) for condition 2.
        coords: 1D array of stimulus intensity values (the x-axis).
        use_smoothing: If True, applies a Gaussian filter along axis 1. Recommended.
        smoothing_sigma: Sigma for the Gaussian smoothing kernel.
        circular: If True, calculates the shift using circular logic.
        stimulus_range: The full range of the circular variable (e.g., 180 for
                        orientation, 360 for direction). Used only if circular=True.

    Returns:
        A tuple of two 1D arrays (each with length = num_neurons):
        - amplitude_changes: The change in peak amplitude (peak2 - peak1).
        - stimulus_shifts: The shift in the preferred stimulus (stimulus2 - stimulus1).
    """
    assert curves1.shape == curves2.shape, "Input curve arrays must have the same shape."

    # --- Step 1: Smooth the data for all neurons at once ---
    if use_smoothing:
        # The `axis=1` argument applies the filter to each row independently
        c1_proc = gaussian_filter1d(curves1, sigma=smoothing_sigma, axis=1)
        c2_proc = gaussian_filter1d(curves2, sigma=smoothing_sigma, axis=1)
    else:
        c1_proc = curves1
        c2_proc = curves2

    # --- Step 2: Calculate amplitude changes for all neurons ---
    # `np.max` with `axis=1` finds the max value in each row
    peak_amplitudes1 = np.max(c1_proc, axis=1)
    peak_amplitudes2 = np.max(c2_proc, axis=1)
    amplitude_changes = peak_amplitudes2 - peak_amplitudes1

    # --- Step 3: Calculate stimulus shifts for all neurons ---
    # `np.argmax` with `axis=1` finds the index of the max value in each row
    peak_indices1 = np.argmax(c1_proc, axis=1)
    peak_indices2 = np.argmax(c2_proc, axis=1)

    # Use these index arrays to look up the preferred stimulus from `coords`
    preferred_stimuli1 = coords[peak_indices1]
    preferred_stimuli2 = coords[peak_indices2]

    stimulus_shifts = preferred_stimuli2 - preferred_stimuli1
    
    # --- Step 4 (Optional): Correct for circular stimulus space ---
    if circular:
        half_range = stimulus_range / 2.0
        # This formula works element-wise on the entire array of shifts
        stimulus_shifts = (stimulus_shifts + half_range) % stimulus_range - half_range

    return amplitude_changes, stimulus_shifts


def calculate_per_neuron_zscore(firing_rate_dict, mean_fr_per_neuron, std_fr_per_neuron):
    """
    Calculates the z-score for each neuron based on its individual mean and std.

    Args:
        firing_rate_dict (dict): Dictionary where keys are condition tuples
            and values are 1D NumPy arrays of mean firing rates for all neurons.
        mean_fr_per_neuron (np.ndarray): A 1D array of each neuron's baseline
            mean firing rate. Shape: (num_neurons,).
        std_fr_per_neuron (np.ndarray): A 1D array of each neuron's baseline
            standard deviation. Shape: (num_neurons,).

    Returns:
        dict: A new dictionary with the same keys but with z-scored firing rates
              as values.
    """
    # --- Safety Check: Handle division by zero ---
    # Create a copy of the std array to avoid modifying the original
    stds_safe = std_fr_per_neuron.copy()
    
    # Find where std is 0
    zero_std_mask = (stds_safe == 0)
    
    # If any stds are zero, print a warning and replace them with 1.
    # Replacing with 1 means the z-score for that neuron will be (rate - mean),
    # preventing an error and avoiding infinitely large z-scores.
    if np.any(zero_std_mask):
        num_zero = np.sum(zero_std_mask)
        print(f"Warning: Found {num_zero} neuron(s) with a standard deviation of 0. "
              "Their z-scores will be calculated as (rate - mean) to avoid division by zero.")
        stds_safe[zero_std_mask] = 1.0

    # --- The Z-Score Calculation using a Dictionary Comprehension ---
    # This is highly efficient due to NumPy's vectorized operations.
    zscore_dict = {
        key: (rates - mean_fr_per_neuron) / stds_safe
        for key, rates in firing_rate_dict.items()
    }

    return zscore_dict



'''
def calculate_per_neuron_zscore(firing_rate_dict, mean_fr_per_neuron, std_fr_per_neuron):
    """
    Calculates the z-score for each neuron based on its individual mean and std.
    
    If a neuron's standard deviation is 0, its z-score is set to 0.

    Args:
        firing_rate_dict (dict): Dictionary where keys are condition tuples
            and values are 1D NumPy arrays of mean firing rates for all neurons.
        mean_fr_per_neuron (np.ndarray): A 1D array of each neuron's baseline
            mean firing rate. Shape: (num_neurons,).
        std_fr_per_neuron (np.ndarray): A 1D array of each neuron's baseline
            standard deviation. Shape: (num_neurons,).

    Returns:
        dict: A new dictionary with the same keys but with z-scored firing rates
              as values.
    """
    # --- Safety Check: Handle division by zero ---
    # Create a copy of the std array to avoid modifying the original
    stds_safe = std_fr_per_neuron.copy()
    
    # Find where std is 0. This boolean mask is the key to the solution.
    zero_std_mask = (stds_safe == 0)
    
    # To prevent a RuntimeWarning from NumPy for 0/0, we still replace 
    # the zeros in our divisor array. The `np.where` will handle the logic,
    # but this ensures the calculation for other neurons doesn't raise a warning.
    if np.any(zero_std_mask):
        num_zero = np.sum(zero_std_mask)
        #print(f"Warning: Found {num_zero} neuron(s) with a standard deviation of 0. "
        #      "Their z-scores will be set to 0.")
        stds_safe[zero_std_mask] = 0

    # --- The Z-Score Calculation using a Dictionary Comprehension ---
    zscore_dict = {}
    for key, rates in firing_rate_dict.items():
        # Calculate the raw z-scores first
        raw_zscores = (rates - mean_fr_per_neuron) / stds_safe
        
        # Use np.where to apply the condition:
        # where(condition, value_if_true, value_if_false)
        # If the std was zero (mask is True), set z-score to 0.
        # Otherwise, keep the calculated raw_zscore.
        final_zscores = np.where(zero_std_mask, 0.0, raw_zscores)
        
        zscore_dict[key] = final_zscores

    return zscore_dict
'''