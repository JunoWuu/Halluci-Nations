import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
import utils.plot_utils as plu
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from signal_processing import sr_processing as sr


class matrics_loader:
    def __init__(self,session):
        self.session = session
        self.df_data_path = os.path.join(session,'data.pkl')
        self.corr_path = os.path.join(session,'corr_map.pkl')
        self.rmap_path = os.path.join(session,'receptive_map.pkl')
        self.tunning_path = os.path.join(session,'tunning.pkl')
        self.orien_coords_path = os.path.join(session,'orientation_coords.pkl')
        self.rmap_coords_path = os.path.join(session,'receptive_map_coords.pkl')

    def load_session_matrics(self):
        try:
            self.df_data = pd.read_pickle(self.df_data_path)
            self.label_dic = get_labels(self.session)
            self.corr = read_pickle_file(self.corr_path)
            self.rmap = read_pickle_file(self.rmap_path)
            self.tunning = read_pickle_file(self.tunning_path)
            self.orien_coords = read_pickle_file(self.orien_coords_path)
            self.rmap_coords = read_pickle_file(self.rmap_coords_path)
        except Exception as e:
            raise Exception(f"Unexpected error in loading data: {e}")
        

def get_labels(session):
    data_dic = {}
    df_path = os.path.join(session,'data.pkl')
    df = pd.read_pickle(df_path)
    data_dic = {'cell_type':df.cell_type.values,
                'region':df.region.values,
                'layer': df.layer.values,
                'probe':df.probe.values,
                'ks_unit_id':df.ks_unit_id.values,
                'estimated_x':df.estimated_x.values,
                'estimated_y':df.estimated_z.values,
                'estimated_z':df.estimated_y.values
                }
    return data_dic

def read_pickle_file(filepath):
    """
    Read and return the contents of a pickle file.
    
    Args:
        filepath (str): Path to the pickle file
        encoding (str): Encoding to use when loading (default: 'latin1')
                       Use 'latin1' for Python 2/3 compatibility
    
    Returns:
        object: The unpickled Python object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file is corrupted or invalid
        PermissionError: If there are permission issues reading the file
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Error unpickling file {filepath}: {e}")
    
    except Exception as e:
        raise Exception(f"Unexpected error reading pickle file {filepath}: {e}")