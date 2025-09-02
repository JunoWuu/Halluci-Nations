import numpy as np
import pandas as pd
import pickle









def extract_spikes(df_interest,stim_ep,main_path):
    stim_ep_path = os.path.join(main_path,stim_ep)
    df_interest = update_stim_eps_end_time(df_visualstim_path, stim_ep)
    if not os.path.exists(stim_ep_path):
        os.makedirs(stim_ep_path)
    else:
        pass
    df_interest.to_csv(os.path.join(stim_ep_path,'trial_info.csv'))
    spikes = df_data[f"{stim_ep}_spikes"].values
    # saves trials (each individual epoch) as its indiviual pickle file for more efficient loading
    pbar = tqdm(total = len(df_interest),leave = True,position = 0)
    for i in range(len(df_interest)):
        start_time = df_interest.iloc[0]['start_time']
        stop_time = df_interest.iloc[0]['stop_time']
        dur = stop_time - start_time
        new_array = [filter_with_numpy(spikes[i], start_time, stop_time) for i in range(spikes.shape[0])]
        new_array.append(dur)
        filename = os.path.join(stim_ep_path, f'trial_array_{i}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(new_array, f)
        pbar.update(1)
    return



'''
stim_ep_path = '/scratch/RFMapping_0/raw'
save_path = '/scratch/RFMapping_0/firing_rate'
time_res = 0.025
pbar = tqdm(total =len(df_interest),leave=True, position=0)
for trial_number in range(len(df_interest)):
    new_array = []
    file_path = os.path.join(stim_ep_path, f'trial_array_{trial_number}.pkl')
    with open(file_path,'rb') as f:
        spike_times = pickle.load(f)
    end_time = spike_times[-1]
    spike_times = spike_times[:-1]
    for i in range(len(spike_times)):
        bin_count,spike_rates,ts = bin_spikes(spike_times[i], end_time,time_res)
        new_array.append(spike_rates)
    new_array.append(ts)
    new_array=np.array(new_array)
    filename = os.path.join(save_path, f'trial_array_{trial_number}.npz')
    np.savez(filename, new_array)
    pbar.update(1)



'''