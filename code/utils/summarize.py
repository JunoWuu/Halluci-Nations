import pandas as pd
import numpy as np

def create_tidy_dataframe(data_dict):
    """
    Converts a nested dictionary into a tidy (long-format) DataFrame.

    The resulting DataFrame will have columns for the RFMapping ID, the layer name,
    and the calculated mean and median of the data.

    Args:
        data_dict (dict): The input dictionary with RFMapping keys.
                          The structure is expected to be:
                          {'RFMapping_X': {'layer_Y': {'data': np.array([...])}}}

    Returns:
        pandas.DataFrame: A tidy DataFrame with columns: 'RFmaps', 'layers', 'mean', 'median'.
    """
    processed_records = []

    # Iterate through each RFMapping session in the dictionary
    for rf_mapping_id, layers in data_dict.items():
        
        # Iterate through each layer within that session
        for layer_name, layer_data in layers.items():
            data_array = layer_data.get('data')

            # Ensure the data array exists and is not empty
            if data_array is not None and data_array.size > 0:
                
                # Create a dictionary for the current row
                record = {
                    'RFmaps': rf_mapping_id,
                    'layers': layer_name,
                    'mean': np.mean(data_array),
                    'median': np.median(data_array) # Also including median for completeness
                }
                processed_records.append(record)

    # Create the final DataFrame from the list of records
    if not processed_records:
        return pd.DataFrame(columns=['RFmaps', 'layers', 'mean', 'median'])

    df = pd.DataFrame(processed_records)

    return df


def create_tidy_dataframe2(data_dict):
    """
    Converts a nested dictionary into a tidy (long-format) DataFrame.

    The resulting DataFrame will have columns for the RFMapping ID, the layer name,
    and the calculated mean and median of the data.

    Args:
        data_dict (dict): The input dictionary with RFMapping keys.
                          The structure is expected to be:
                          {'RFMapping_X': {'layer_Y': {'data': np.array([...])}}}

    Returns:
        pandas.DataFrame: A tidy DataFrame with columns: 'RFmaps', 'layers', 'mean', 'median'.
    """
    processed_records = []

    # Iterate through each RFMapping session in the dictionary
    for rf_mapping_id, layers in data_dict.items():
        if rf_mapping_id == 'Spontaneous_1' or rf_mapping_id == 'Spontaneous_2':
            # Iterate through each layer within that session
            for layer_name, layer_data in layers.items():
                data_array = layer_data.get('data')

                # Ensure the data array exists and is not empty
                if data_array is not None and data_array.size > 0:
                    
                    # Create a dictionary for the current row
                    record = {
                        'RFmaps': rf_mapping_id,
                        'layers': layer_name,
                        'mean': np.mean(data_array),
                        'median': np.median(data_array) # Also including median for completeness
                    }
                    processed_records.append(record)

    # Create the final DataFrame from the list of records
    if not processed_records:
        return pd.DataFrame(columns=['RFmaps', 'layers', 'mean', 'median'])

    df = pd.DataFrame(processed_records)

    return df