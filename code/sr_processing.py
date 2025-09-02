import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler


def spike_rate_zscore(data):
    scaler = StandardScaler()
    data_T = data.T 
    z_scored_data_T = scaler.fit_transform(data_T)
    z_scored_data = z_scored_data_T.T
    return z_scored_data

def get_spikerate_data(path):
    data = np.load(path)['arr_0']
    ts = data[-1,:]
    data = data[:-1,:]
    zscored_data = spike_rate_zscore(data)
    return {'timestamp':ts,'raw_fr':data,'zscored_fr':zscored_data}

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
    min_timestamp_len = min([d['timestamp'].shape[0] for d in data_set.values()])
    for key, inner_dict in data_set.items():
        corrected_data_set[key] = {
            # Slice the 1D timestamp array
            'timestamp': inner_dict['timestamp'][:min_timestamp_len],

            # Slice the 2D firing rate arrays on their second dimension
            'raw_fr': inner_dict['raw_fr'][:, :min_timestamp_len],
            'zscored_fr': inner_dict['zscored_fr'][:, :min_timestamp_len]
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
