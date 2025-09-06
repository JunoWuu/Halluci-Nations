import os
import re
import numpy as np
from typing import List, Dict,Union, Tuple,Any


def find_directories_with_rfmapping(root_directory):
    """
    Scans subdirectories to find which ones contain 3 or 4 folders
    with the pattern 'RFMapping_X'.

    Args:
        root_directory (str): The absolute path to the directory to start the scan from.

    Returns:
        tuple: A tuple containing two lists of paths:
               - The first list for directories with 3 RFMapping folders.
               - The second list for directories with 4 RFMapping folders.
    """
    dirs_with_3 = []
    dirs_with_4 = []

    # Walk through the directory tree starting from the root
    for dirpath, dirnames, _ in os.walk(root_directory):
        rf_mapping_count = 0

        # Count the subdirectories that match the pattern
        for dirname in dirnames:
            if dirname.startswith("RFMapping_"):
                # Optional: Add a check to ensure 'X' is a number
                try:
                    int(dirname.split('_')[1])
                    rf_mapping_count += 1
                except (ValueError, IndexError):
                    # This folder doesn't match the "RFMapping_NUMBER" pattern
                    pass

        # If RFMapping folders were found, check the count
        if rf_mapping_count == 3:
            dirs_with_3.append(dirpath)
        elif rf_mapping_count == 4:
            dirs_with_4.append(dirpath)

        # To prevent recounting in nested directories, you can optionally
        # remove the found RFMapping directories from further traversal.
        dirnames[:] = [d for d in dirnames if not d.startswith("RFMapping_")]

    return dirs_with_3, dirs_with_4


def extract_subject_id(path):
    """
    Extracts the subject ID from a file path string.

    The function assumes the subject ID is a sequence of numbers
    that directly follows the 'ecephys_' prefix in the path.

    Args:
        path (str): The file path to parse.

    Returns:
        str: The extracted subject ID, or None if no match is found.
    """
    # First, get the final directory or file name from the path. [7, 9]
    # This makes the pattern matching more specific.
    basename = os.path.basename(path)

    # The regular expression looks for 'ecephys_', followed by one or
    # more digits (\d+). The parentheses capture the digits. [1, 2, 5]
    match = re.search(r'ecephys_(\d+)', basename)

    if match:
        # The subject ID is the first captured group. [1]
        return match.group(1)
    else:
        return None
    
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


def group_data_by_hardcoded_layers_compatible(
    labels_array: np.ndarray, 
    data_array: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Groups data into predefined layers based on substrings in a label array.
    
    *** This version is compatible with older NumPy versions (< 1.14) ***
    
    The function is hard-coded to search for labels containing the following:
    - 'layer_1': contains '1'
    - 'layer_2_3': contains '2/3'
    - 'layer_4': contains '4'
    - 'layer_5': contains '5'
    - 'layer_6': contains '6'
    """
    if len(labels_array) != len(data_array):
        raise ValueError("Input arrays 'labels_array' and 'data_array' must have the same length.")

    layer_definitions = {
        'layer_1': '1',
        'layer_2_3': '2/3',
        'layer_4': '4',
        'layer_5': '5',
        'layer_6': '6'
    }
    
    grouped_results = {}

    for layer_name, substring_to_find in layer_definitions.items():
        # --- THIS IS THE MODIFIED LINE ---
        # Create a boolean mask using a list comprehension. This works on all Python/NumPy versions.
        mask = np.array([substring_to_find in label for label in labels_array])
        # ---------------------------------
        
        indices = np.where(mask)[0]
        subset_data = data_array[indices]
        
        grouped_results[layer_name] = {
            'data': subset_data
        }
        
    return grouped_results

def calculate_inter_regional_correlation(regions, layers, matrix, target_region_y, target_region_x):
    """
    Subsets, organizes, and computes the correlation matrix between two target regions.

    This function streamlines the process by returning the final cross-correlation
    matrix (Y-axis vs. X-axis) and the necessary metadata for plotting.

    Args:
        regions (array-like): An array of region strings for each neuron.
        layers (array-like): An array of layer strings for each neuron.
        matrix (np.ndarray): The data matrix, typically (total_neurons x time_points).
        target_region_y (str): The name of the region for the y-axis.
        target_region_x (str): The name of the region for the x-axis.

    Returns:
        tuple: A tuple containing:
        (
            correlation_matrix (np.ndarray): The (neurons_y x neurons_x) correlation matrix.
            layer_info_y (dict): Dictionary with layer statistics for the y-axis region.
            layer_info_x (dict): Dictionary with layer statistics for the x-axis region.
        )
    """
    # --- Helper function to process one region (same as before) ---
    def process_region(target_region):
        regions_np = np.array(regions)
        layers_np = np.array(layers)
        region_mask = regions_np == target_region
        
        if not np.any(region_mask):
            return np.empty((0, matrix.shape[1])), {} # Return empty matrix and info

        subset_layers = layers_np[region_mask]
        subset_matrix = matrix[region_mask, :]
        unique_layers = sorted(np.unique(subset_layers))
        
        organized_rows, layer_info, current_index = [], {}, 0
        for layer in unique_layers:
            layer_mask = subset_layers == layer
            n_neurons = np.sum(layer_mask)
            if n_neurons > 0:
                organized_rows.append(subset_matrix[layer_mask, :])
                layer_info[layer] = {'start_idx': current_index, 'end_idx': current_index + n_neurons, 'n_neurons': n_neurons}
                current_index += n_neurons
        
        organized_matrix = np.vstack(organized_rows) if organized_rows else np.empty((0, matrix.shape[1]))
        return organized_matrix, layer_info

    # --- 1. Subset and organize data for both regions ---
    matrix_y, info_y = process_region(target_region_y)
    matrix_x, info_x = process_region(target_region_x)

    # --- 2. Handle cases with no neurons ---
    if matrix_y.shape[0] == 0 or matrix_x.shape[0] == 0:
        # Return an empty matrix and the (possibly empty) info dicts
        return np.empty((matrix_y.shape[0], matrix_x.shape[0])), info_y, info_x

    # --- 3. Calculate the full correlation matrix ---
    # This matrix contains Y-Y, Y-X, X-Y, and X-X correlations
    full_corr_matrix = np.corrcoef(matrix_y, matrix_x)

    # --- 4. Extract only the Y-X cross-correlation block ---
    num_neurons_y = matrix_y.shape[0]
    correlation_matrix = full_corr_matrix[:num_neurons_y, num_neurons_y:]
    
    return correlation_matrix, info_y, info_x


def group_and_sort_data_by_layer(
    labels_array: np.ndarray, 
    data_array: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Groups data into predefined layers and ensures the output keys are sorted.
    
    The function searches for labels containing specific substrings and returns
    a dictionary with keys sorted in an anatomically correct order.
    
    Hard-coded layers:
    - 'layer_1': contains '1'
    - 'layer_2_3': contains '2/3'
    - 'layer_4': contains '4'
    - 'layer_5': contains '5'
    - 'layer_6': contains '6'
    """
    if len(labels_array) != len(data_array):
        raise ValueError("Input arrays 'labels_array' and 'data_array' must have the same length.")

    # Definitions of what substring corresponds to each layer name
    layer_definitions = {
        'layer_1': '1',
        'layer_2_3': '2/3',
        'layer_4': '4',
        'layer_5': '5',
        'layer_6': '6'
    }
    
    # --- THIS IS THE KEY CHANGE ---
    # Define the desired order of keys for the output dictionary.
    sorted_layer_keys = ['layer_1', 'layer_2_3', 'layer_4', 'layer_5', 'layer_6']
    # ----------------------------
    
    grouped_results = {}

    # Iterate through the pre-sorted list of keys, not the dictionary directly.
    # This guarantees the insertion order.
    for layer_name in sorted_layer_keys:
        # Check if the layer is defined, then proceed
        if layer_name in layer_definitions:
            substring_to_find = layer_definitions[layer_name]
            
            # Create a boolean mask to find matching labels
            mask = np.array([substring_to_find in str(label) for label in labels_array])
            
            # Get the indices and subset the data
            indices = np.where(mask)[0]
            subset_data = data_array[indices]
            
            # Only add to results if data was found for that layer
            if subset_data.size > 0:
                grouped_results[layer_name] = {
                    'data': subset_data
                }
        
    return grouped_results


def expand_labels_by_neuron_count(layer_info: Dict[str, Any]) -> List[str]:
    """
    Creates a full list of labels by repeating each layer name by its neuron count.

    The order of labels in the output list is determined by the insertion order
    of the keys in the input dictionary.

    Args:
        layer_info (Dict[str, Any]): 
            A dictionary where keys are layer labels (e.g., 'SSp-bfd1') and
            values are dictionaries containing an 'n_neurons' key.

    Returns:
        List[str]: 
            A 1D list containing the repeated labels for every neuron.
    """
    expanded_labels = []
    
    # Iterate through the dictionary's items. In Python 3.7+, this preserves insertion order.
    for layer_name, info in layer_info.items():
        # Get the number of neurons for the current layer
        num_neurons = info['n_neurons']
        
        # Create a temporary list by repeating the layer name
        # and extend the main list with it.
        # e.g., ['SSp-bfd1'] * 14 becomes ['SSp-bfd1', 'SSp-bfd1', ...]
        expanded_labels.extend([layer_name] * num_neurons)
        
    return expanded_labels