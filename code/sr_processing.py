import numpy as np
import pandas as pd 
import pickle
from sklearn.preprocessing import StandardScaler


def add_epsilon(array, epsilon=1e-10):
    """Add a small number to prevent zeros."""
    return array + epsilon

def get_corr_mat(data):
    v = add_epsilon(data, epsilon=1e-7)
    correlation_matrix = np.corrcoef(v)
    return correlation_matrix

def load_pickle(path):
    """Load a pickle file from the given path."""
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def spike_rate_zscore(data):
    scaler = StandardScaler()
    data_T = data.T 
    z_scored_data_T = scaler.fit_transform(data_T)
    z_scored_data = z_scored_data_T.T
    return z_scored_data

def get_spikerate_data(filename):
    loaded_data = load_pickle(filename)
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

def subset_and_organize_matrices(regions, layers, matrix, target_region):
    """
    Subset by region and organize by layers, returning matrices.
    
    Args:
        regions: array of region strings
        layers: array of layer strings  
        matrix: (neuron x time_points) matrix
        target_region: string of region to subset
    
    Returns:
        organized_matrix: (neuron x time_points) matrix with all neurons from target region
        layer_labels: array of layer strings corresponding to each neuron row
        layer_info: dict with layer statistics
    """
    # Convert to numpy arrays
    regions = np.array(regions)
    layers = np.array(layers)
    
    # Subset by region
    region_mask = regions == target_region
    subset_layers = layers[region_mask]
    subset_matrix = matrix[region_mask, :]
    
    # Get unique layers and sort them
    unique_layers = sorted(np.unique(subset_layers))
    
    # Reorganize data by layer order
    organized_rows = []
    organized_layer_labels = []
    layer_info = {}
    
    for layer in unique_layers:
        layer_mask = subset_layers == layer
        layer_data = subset_matrix[layer_mask, :]
        
        organized_rows.append(layer_data)
        organized_layer_labels.extend([layer] * np.sum(layer_mask))
        
        # Store layer info
        layer_info[layer] = {
            'start_idx': len(organized_layer_labels) - np.sum(layer_mask),
            'end_idx': len(organized_layer_labels),
            'n_neurons': np.sum(layer_mask)
        }
    
    # Concatenate all layers
    organized_matrix = np.vstack(organized_rows)
    organized_layer_labels = np.array(organized_layer_labels)
    
    return organized_matrix, organized_layer_labels, layer_info

def subset_and_organize_corr_matrix(regions, layers, corr_matrix, target_region):
    """
    Subsets a correlation matrix by region and reorganizes it by layer.
    
    Args:
        regions (np.ndarray): Array of region strings for each neuron (length N).
        layers (np.ndarray): Array of layer strings for each neuron (length N).
        corr_matrix (np.ndarray): The full (N x N) correlation matrix.
        target_region (str): The string of the region to subset.
    
    Returns:
        organized_matrix (np.ndarray): The subsetted (M x M) correlation matrix, 
                                     with rows/columns sorted by layer.
        organized_layer_labels (np.ndarray): Array of layer strings for the new sorted matrix.
        layer_info (dict): Dictionary with statistics for each layer in the new matrix.
    """
    # Ensure inputs are numpy arrays for efficient boolean indexing
    regions = np.array(regions)
    layers = np.array(layers)

    # 1. Find the indices of all neurons in the target region
    region_mask = (regions == target_region)
    
    # Check if any neurons were found for the target region
    if not np.any(region_mask):
        print(f"Warning: No neurons found for target region '{target_region}'. Returning empty results.")
        return np.array([]), np.array([]), {}

    # 2. Subset the correlation matrix on both dimensions (rows and columns)
    subset_matrix = corr_matrix[np.ix_(region_mask, region_mask)]
    
    # 3. Get the layers corresponding to only the subsetted neurons
    subset_layers = layers[region_mask]

    # 4. Get the sorting order based on layer labels for the subset
    sort_indices = np.argsort(subset_layers, kind='stable')
    
    # 5. Reorganize the subsetted matrix and layer labels using the sorting order
    organized_matrix = subset_matrix[np.ix_(sort_indices, sort_indices)]
    organized_layer_labels = subset_layers[sort_indices]

    # 6. Create the layer_info dictionary from the new, organized data
    layer_info = {}
    unique_layers, counts = np.unique(organized_layer_labels, return_counts=True)
    
    current_idx = 0
    for layer, n_neurons in zip(unique_layers, counts):
        layer_info[layer] = {
            'start_idx': current_idx,
            'end_idx': current_idx + n_neurons,
            'n_neurons': n_neurons
        }
        current_idx += n_neurons
        
    return organized_matrix, organized_layer_labels, layer_info
