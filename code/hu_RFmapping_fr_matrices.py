import os
import numpy as np
import pandas as pd 
import signal_processing.sr_processing as sr
import signal_processing.receptive_mapping as rmap
from joblib import Parallel, delayed
from collections import defaultdict
import pickle
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class hu_RFmapping_fr_matrices:
    def __init__(self,main_path):
        self.main_path = main_path
        self.df_data = pd.read_pickle(os.path.join(main_path,'data.pkl'))
        self.df_epochs = pd.read_pickle(os.path.join(main_path,'epochs.pkl'))
        self.vis_epochs = [i for i in self.df_epochs['stim_name'] if 'RFMapping' in i]
        self.cell_type = self.df_data.cell_type.values
        self.region = self.df_data.region.values
        self.layer = self.df_data.layer.values
        self.probe = self.df_data.probe
        self.ks_unit_id = self.df_data.ks_unit_id
        self.x_coords = self.df_data.estimated_x
        self.y_coords = self.df_data.estimated_z
        self.z_coords = self.df_data.estimated_y
    def load_data(self):
        self.epoch_paths = [os.path.join(self.main_path,ep) for ep in self.vis_epochs]
        self.trial_info = {ep: pd.read_csv(os.path.join(ep_path,'trial_info.csv')) for
                         ep,ep_path in zip(self.vis_epochs ,self.epoch_paths)}
        self.trial_dic = {ep: sr.get_index_maps(self.trial_info[ep]) for ep in
                         self.vis_epochs}
        self.trial_data = {ep:sr.load_pickle(os.path.join(ep_path,'firing_rates_zscored.pkl'))
                           for ep,ep_path in zip(self.vis_epochs ,self.epoch_paths)}
        self.trial_data_counts = {ep:sr.load_pickle(os.path.join(ep_path,'firing_rates.pkl'))
                    for ep,ep_path in zip(self.vis_epochs ,self.epoch_paths)}
        return
    def get_vis_epochs_keys(self):
        e_k_length = []
        for epoch in self.vis_epochs:
            k_s1 = list(self.trial_dic[epoch].keys())
            e_k_length.append(len(set(k_s1)))
        if len(set(e_k_length)) == 1:
            self.trial_type_keys = k_s1
            return 
        else:
            print('Trial types do not match across trials')
    def get_cor_matrices(self):
        self.main_dic = {}
        for vis_epoch in self.vis_epochs:
            items = self.trial_data[vis_epoch].items()
            
            results = Parallel(n_jobs=-1)(delayed(compute_corr_matrix)(item) for item in items)
            
            self.main_dic[vis_epoch] = dict(results)
        savepath = os.path.join(self.main_path,'corr_map.pkl') 
        with open(savepath, 'wb') as f:
            pickle.dump(self.main_dic, f)
        return
    
    def comput_mean_fr(self):
        self.mean_fr = {}
        for vis_epochs in self.vis_epochs:
            mean_fr = {key: np.mean(value, axis=1) for key, 
                       value in self.trial_data[vis_epochs].items()}
            self.mean_fr[vis_epochs] = mean_fr

        return
    
    def gen_rf_maps(self):
        self.receptive_maps = {}
        for vis_epoch in self.vis_epochs:
            rf_map,x_coords,y_coords = rmap.create_receptive_field(self.mean_fr[vis_epoch])
            #np.save(savepath,)
            self.receptive_maps[vis_epoch] = rf_map
        savepath = os.path.join(self.main_path,'receptive_map.pkl') 
        with open(savepath, 'wb') as f:
            pickle.dump(self.receptive_maps, f)
        self.rf_x_coords = x_coords
        self.rf_y_coords = y_coords
        return 

    def get_tunning_curves(self):
        self.tunning_curves = {}
        for vis_epoch,vis_epoch_path in zip(self.vis_epochs,self.epoch_paths):
            tunning_curves = []
            for neuron_id in self.receptive_maps[vis_epoch].keys():
                idx_flat = np.argmax(self.receptive_maps[vis_epoch][neuron_id])
                idx_coords = np.unravel_index(idx_flat, 
                                              self.receptive_maps[vis_epoch][neuron_id].shape)
                max_x_id = self.rf_x_coords[idx_coords[1]]
                max_y_id = self.rf_y_coords[idx_coords[0]]
                subselected_data = {
                    key[0]: value[neuron_id] for key, value in self.mean_fr[vis_epoch].items()
                    if key[1] == max_x_id and key[2] == max_y_id
                }
                sorted_dict = dict(sorted(subselected_data.items()))
                tune = np.array(list(sorted_dict.values()))
                savepath = os.path.join(vis_epoch_path,'tunning.npy') 
                np.save(savepath,tune)
                tunning_curves.append(tune)
            self.tunning_curves[vis_epoch] = tunning_curves
        savepath = os.path.join(vis_epoch_path,'tunning.pkl') 
        with open(savepath, 'wb') as f:
            pickle.dump(self.tunning_curves, f)
        return
    def get_tunning_csv(self):
        self.df_tunning = {}
        for vis_epoch,vis_epoch_path in zip(self.vis_epochs,self.epoch_paths): 
            data_dict = {
                'cell_type': self.cell_type,
                'region': self.region,
                'layer': self.layer,
                'probe': self.probe,
                'ks_unit_id': self.ks_unit_id,
                'x_coords': self.x_coords,
                'y_coords': self.y_coords,
                'z_coords': self.z_coords,
                'tuning_curves': self.tunning_curves[vis_epoch]
            }
            df = pd.DataFrame(data_dict)
            savepath = os.path.join(vis_epoch_path,'tunning.h5') 
            df.to_hdf(savepath ,key='tunning', mode='w', format='fixed')
        self.df_tunning[vis_epoch] = df
        return 

def compute_corr_matrix(item):
    i, v = item
    return i, sr.get_corr_mat(v)