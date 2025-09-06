import numpy as np
import pandas as pd
import pickle




layers = []
for unit_idx in range(len(analysis_table)):
    spike_times = analysis_table.iloc[unit_idx]['Spontaneous_0_spikes']
    if spike_times is None or len(spike_times) == 0:
        continue
    layers.append(analysis_table.iloc[unit_idx]['layer'])
layers = np.array(layers)
fr_matrix = firing_rate_matrix_spon0.copy()
# 2. Sort neurons by layer
sorted_indices = np.argsort(layers)
sorted_layers = layers[sorted_indices]
fr_matrix_sorted = fr_matrix[sorted_indices]
# 3. Compute correlation matrix
corr_matrix = np.corrcoef(fr_matrix_sorted)
# 4. Plot
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='vlag', center=0, square=True, xticklabels=False, yticklabels=False)
plt.title("Neuron × Neuron Correlation (Grouped by Layer)", fontsize=14)
# 5. Optionally draw lines to show layer boundaries
unique_layers, counts = np.unique(sorted_layers, return_counts=True)
layer_boundaries = np.cumsum(counts)
# Draw horizontal and vertical lines to separate layers
for boundary in layer_boundaries[:-1]:
    plt.axhline(boundary, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(boundary, color='gray', linestyle='--', linewidth=0.5)
# 6. Compute tick positions (center of each group)
tick_positions = layer_boundaries - counts // 2
# :white_check_mark: Fix: Make sure tick_positions and labels match in length
assert len(tick_positions) == len(unique_layers)
# Add layer labels
plt.yticks(tick_positions, unique_layers, fontsize=10)
plt.xticks(tick_positions, unique_layers, fontsize=10, rotation=90)
plt.tight_layout()
plt.show()


# Sort the arrays
reference_array = im2plot[0]
first_max_indices = np.argmax(reference_array, axis=1)
sort_indices = np.argsort(first_max_indices)

# Apply sorting to all arrays
sorted_im2plot = [arr[sort_indices] for arr in im2plot]

# Visualize before and after
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Original stacked image
original_stacked = np.hstack(im2plot)
im1 = ax1.imshow(original_stacked, aspect='auto', cmap='viridis')
ax1.set_title('Original Arrays')

# Sorted stacked image
sorted_stacked = np.hstack(sorted_im2plot)
im2 = ax2.imshow(sorted_stacked, aspect='auto', cmap='viridis')
ax2.set_title('Sorted by First Appearance of Maximum Value')

# Calculate tick positions based on actual array shapes
if 'titles' in locals():
    for ax in [ax1, ax2]:
        tick_positions = []
        array_widths = [arr.shape[1] for arr in im2plot]  # Get width of each array
        
        current_position = 0
        for width in array_widths:
            # Place tick at center of each array
            tick_positions.append(current_position + width/2 - 0.5)
            current_position += width
        
        tick_labels = [str(title) for title in titles]
        ax.set_xticks(tick_positions, tick_labels, rotation=45, ha='right')
        
        # Add separation lines based on actual widths
        current_position = 0
        for i, width in enumerate(array_widths[:-1]):  # Don't add line after last array
            current_position += width
            ax.axvline(x=current_position - 0.5, color='white', linestyle='-', linewidth=2, alpha=0.8)

plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()

print(f"Rows are now ordered by column position of first max:")
print(f"Array shapes: {[arr.shape for arr in im2plot]}")
print(f"Array widths: {[arr.shape[1] for arr in im2plot]}")

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

titles =  [i[0] for i in test[0]]
im2plot = [i[1] for i in test[0]]
reference_array = im2plot[0]  # The array to use for determining sort order

# Get the maximum value in each row
row_max_values = np.max(reference_array, axis=1)

# Get the indices that would sort by descending order (highest first)
sort_indices = np.argsort(row_max_values)[::-1]

# Apply this ordering to all arrays
sorted_array1 = im2plot[0][sort_indices]
sorted_array2 = im2plot[1][sort_indices]
sorted_array3 = im2plot[2][sort_indices]








session = '/home/tony/Halluci-Nations/scratch/ecephys_717033_2024-06-04_13-01-40_nwb_2025-08-03_21-11-22'
data_dic = {}
metrics_test = matrics_loader(session)
metrics_test.load_session_matrics()
cell_type = metrics_test.label_dic['cell_type']
regions = metrics_test.label_dic['region']
layers = metrics_test.label_dic['layer']
coords = metrics_test.orien_coords
test = metrics_test.tunning['RFMapping_0']
test2 = metrics_test.tunning['RFMapping_1']
test3 = metrics_test.tunning['RFMapping_2']
test4 = metrics_test.tunning['RFMapping_3']
tar_reg = 'VISp' 
ace1 = test[regions ==tar_reg ]
ace2 = test2[regions ==tar_reg ]
ace3 = test3[regions ==tar_reg ]
ace4 = test4[regions ==tar_reg ]
amps, tunning = sr.calculate_population_tuning_changes(ace1,ace2,coords,circular=True,stimulus_range=180)
amps2, tunning2 = sr.calculate_population_tuning_changes(ace1,ace3,coords,circular=True,stimulus_range=180)
amps3, tunning3 = sr.calculate_population_tuning_changes(ace1,ace4,coords,circular=True,stimulus_range=180)
amp1_mean = np.mean(amps)
amp2_mean = np.mean(amps2)
amp3_mean = np.mean(amps3)
amp1_med = np.median(amps)
amp2_med = np.median(amps2)
amp3_med = np.median(amps3)
tunning1_mean = np.mean(tunning)
tunning2_mean = np.mean(tunning2)
tunning3_mean = np.mean(tunning3)
tunning1_med = np.median(tunning)
tunning2_med = np.median(tunning2)
tunning3_med = np.median(tunning3)
data_dic['amp_mean'] = [amp1_mean,amp2_mean,amp3_mean]
data_dic['amp_med'] = [amp1_med,amp2_med,amp3_med]
data_dic['tunning_mean'] = [tunning1_mean,tunning2_mean,tunning3_mean]
data_dic['tunning_med'] = [tunning1_med,tunning2_med,tunning3_med]

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