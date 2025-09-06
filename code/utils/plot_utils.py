import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict,Union, Tuple
import pandas as pd



def raster_bny_cell_type(spike_times, cell_types):
    unique_labels = np.unique(cell_types)
    cmap = plt.cm.get_cmap('tab10', len(unique_labels)) # A colormap good for distinct categories
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}


    # 2. Create a list of colors, one for each neuron, based on its label
    neuron_colors = [color_map[label] for label in cell_types]


    # 3. Create the raster plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # The eventplot function is perfect for this task.
    # It takes the list of spike times and a corresponding list of colors.
    ax.eventplot(spike_times, colors=neuron_colors, linelengths=20)

    ax.set_title('Raster Plot Organized by Cell Type')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(left=0) # Ensure the plot starts at time 0
    ax.set_ylim(-1, len(spike_times)) # Set y-axis limits correctly


    # 4. Create a custom legend to identify the cell types
    legend_elements = [Line2D([0], [0], color=color_map[label], lw=4, label=label)
                       for label in unique_labels]
    ax.legend(handles=legend_elements, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')


    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return


def plot_raster(spike_times,title,lsize = 5):
    fig, ax = plt.subplots(figsize=(10, 5))

    # The `eventplot` function is ideal for raster plots
    ax.eventplot(spike_times, color='black', linelengths=lsize)

    # --- 3. Customize the Plot ---
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron ID')

    # Set the y-axis to show neuron indices
    #ax.set_yticks(range(len(spike_times)))
    #ax.set_yticklabels([str(i) for i in range(len(spike_times))])
    ax.set_ylim(-1, len(spike_times)) # Give some padding to the y-axis

    plt.tight_layout()
    return



def generate_matrices_comparison(matrices,layer_info,
                                 unique_layers,epochs,
                                 vmin=-1,vmax=1,
                                 title = 'Name this graph'):
    # Create a figure

    # --- CHANGE 1: Define a grid ---
    # Create a grid with 1 row and (num_matrices + 1) columns.
    # The last column will be thin and reserved for the colorbar.
    # `width_ratios` controls the relative size of the columns.
    num_matrices = len(matrices)
    fig = plt.figure(figsize=(6 * num_matrices, 5.5))
    gs = GridSpec(1, num_matrices + 1, width_ratios=[40] * num_matrices + [1])

    # Create the main axes for the plots
    axes = []
    for i in range(num_matrices):
        # Share the Y axis with the first plot (if it exists)
        share_y_ax = axes[0] if i > 0 else None
        axes.append(fig.add_subplot(gs[0, i], sharey=share_y_ax))

    # Create a dedicated axis for the colorbar in the last grid column
    cbar_ax = fig.add_subplot(gs[0, -1])

    # --- Loop and plot as before ---
    for i, correlation_matrix in enumerate(matrices):
        ax = axes[i]
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax)

        y_positions = []
        layer_names = []
        for layer in unique_layers:
            start_idx = layer_info[layer]['start_idx']
            end_idx = layer_info[layer]['end_idx']
            if start_idx > 0:
                ax.axhline(y=start_idx - 0.5, color='white', linewidth=2)
                ax.axvline(x=start_idx - 0.5, color='white', linewidth=2)
            middle_pos = (start_idx + end_idx - 1) / 2
            y_positions.append(middle_pos)
            layer_names.append(f'L{layer}')

        ax.set_xticks(y_positions)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_xlabel('Neurons (by Layer)')
        ax.set_title(f'Condition: {epochs[i]}')

    # --- 3. Final Figure-Level Adjustments ---
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(layer_names)
    axes[0].set_ylabel('Neurons (by Layer)')

    # Hide y-tick labels on subsequent plots for clarity
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # --- CHANGE 2: Add the colorbar to its dedicated axis ---
    fig.colorbar(im, cax=cbar_ax, label='Correlation')

    fig.suptitle(f'{title} Correlation Matrices Organized by Layer', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return

def plot_3d_scatter_by_label(df, x_col, y_col, z_col, color_col, 
                             filter_col=None, filter_values=None,
                             title=None, figsize=(12, 9), marker_size=50, cmap='viridis'):
    """
    Generates a 3D scatter plot from a DataFrame, with optional filtering.

    - Color-codes points by a specified column ('color_col').
    - Optionally filters the data to only include rows where 'filter_col'
      contains values from 'filter_values'.
    - Rows where the 'color_col' value is NaN are always ignored.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column for the X-axis.
        y_col (str): The name of the column for the Y-axis.
        z_col (str): The name of the column for the Z-axis.
        color_col (str): The name of the column for color-coding the points.
        filter_col (str, optional): The column to apply a filter on. Defaults to None.
        filter_values (list or str, optional): A list of values (or a single value)
                                               to keep from the filter_col. Defaults to None.
        title (str, optional): The title of the plot. Defaults to a generated title.
        figsize (tuple, optional): The size of the figure. Defaults to (12, 9).
        marker_size (int, optional): The size of the markers. Defaults to 50.
        cmap (str, optional): The name of the colormap to use. Defaults to 'viridis'.
    """
    # 1. Filter the DataFrame based on the new optional parameters
    if filter_col and filter_values is not None:
        # Ensure filter_values is a list to handle single-item filters gracefully
        if not isinstance(filter_values, (list, tuple, set)):
            filter_values = [filter_values]
        # Apply the filter
        filtered_df = df[df[filter_col].isin(filter_values)]
    else:
        # If no filter is specified, use the entire DataFrame
        filtered_df = df

    # 2. Handle missing labels in the color column from the (now possibly filtered) data
    plot_df = filtered_df.dropna(subset=[color_col]).copy()
    
    # Check if there is any data left to plot
    if plot_df.empty:
        print(f"No data to plot after applying filters and removing NaNs.")
        return

    # 3. Set up the plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 4. Get unique labels and assign colors
    unique_labels = sorted(plot_df[color_col].unique())
    colors = plt.cm.get_cmap(cmap, len(unique_labels))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # 5. Plot each label group with its assigned color
    for label_name, group_df in plot_df.groupby(color_col):
        ax.scatter(group_df[x_col], group_df[y_col], group_df[z_col],
                   color=color_map[label_name], label=label_name, s=marker_size)

    # 6. Customize and show the plot
    if title is None:
        title = f'3D Scatter Plot of {color_col.title()}'
        if filter_col:
            title += f" (Filtered by {filter_col.title()})"

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_zlabel(z_col.replace('_', ' ').title())
    
    ax.legend(title=color_col.replace('_', ' ').title())


def plot_grouped_layer_barplot_robust(
    data_dic: Dict,
    # ... all other arguments are the same ...
    filter_threshold: float = 90.0,
    errorbar_setting: Union[str, Tuple[str, float]] = ('ci', 68),
    capsize: float = 0.1,
    title: str = 'Comparison of Conditions Across Cortical Layers',
    xlabel: str = 'Cortical Layer',
    ylabel: str = 'Mean Value (filtered)',
    figsize: Tuple[int, int] = (12, 8),
    layer_label_map: Dict[str, str] = None,
    legend_title: str = 'Condition'
) -> None:
    """
    Generates a grouped bar plot. This robust version can handle both a
    multi-condition dictionary and a single-condition dictionary as input.
    """
    # --- NEW: Check the input data structure ---
    if not data_dic:
        print("Warning: Input dictionary is empty.")
        return

    # Check the first value in the dictionary to guess the structure
    first_value = next(iter(data_dic.values()))
    # If the first value is a dict and has a 'data' key, we likely have a single condition dict.
    if isinstance(first_value, dict) and 'data' in first_value:
        print("Warning: Input appears to be a single-condition dictionary. Wrapping it.")
        # Wrap it in another dictionary to create the structure the function expects
        data_dic = {'Default Condition': data_dic}
    
    # --- The rest of the function is the same as before ---
    
    # 1. Data Transformation
    records = []
    for condition, layer_dict in data_dic.items():
        for layer_key, data_obj in layer_dict.items():
            for value in data_obj['data']:
                records.append({
                    'Condition': condition,
                    'Layer_Key': layer_key,
                    'Value': value
                })
    # ... (the rest of the function code for filtering, mapping, and plotting is identical)
    if not records:
        print("Warning: No data found after processing data_dic. Cannot create plot.")
        return

    df = pd.DataFrame(records)
    df_filtered = df[df['Value'] < filter_threshold].copy()
    if df_filtered.empty:
        print(f"Warning: All data filtered out with threshold {filter_threshold}. Cannot create plot.")
        return

    if layer_label_map is None:
        layer_label_map = {
            'layer_1': 'Layer 1', 'layer_2_3': 'Layer 2/3', 'layer_4': 'Layer 4',
            'layer_5': 'Layer 5', 'layer_6': 'Layer 6'
        }
    df_filtered['Layer'] = df_filtered['Layer_Key'].map(layer_label_map).fillna(df_filtered['Layer_Key'])

    unique_layers = df_filtered['Layer'].unique()
    try:
        sorted_layers = sorted(unique_layers, key=lambda x: (int(x.split(' ')[1].split('/')[0]), float(x.split(' ')[1].split('/')[1]) if '/' in x else 0) if 'Layer' in x else x)
    except (ValueError, IndexError):
        sorted_layers = sorted(unique_layers)
    
    df_filtered['Layer'] = pd.Categorical(df_filtered['Layer'], categories=sorted_layers, ordered=True)

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_filtered, x='Layer', y='Value', hue='Condition',
        errorbar=errorbar_setting, capsize=capsize
    )
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title=legend_title, fontsize=12)
    plt.tight_layout()




def plot_neuron_tuning_curves(
    curves1: np.ndarray,
    curves2: np.ndarray,
    stimulus_axis: np.ndarray,
    num_to_plot: int = 20,
    labels: tuple = ('Condition 1', 'Condition 2')
) -> None:
    """
    Plots tuning curves for two conditions for multiple neurons on subplots.

    Args:
        curves1: Array (neurons x tuning curve) for the first condition.
        curves2: Array (neurons x tuning curve) for the second condition.
        stimulus_axis: 1D array for the x-axis (stimulus values).
        num_to_plot: The number of neurons to plot from the start of the array.
        labels: A tuple of strings for the legend labels.
    """
    # Determine the grid layout (e.g., 4 rows, 5 columns for 20 plots)
    ncols = 5
    nrows = int(np.ceil(num_to_plot / ncols))

    # Create the figure and the grid of subplots
    # sharex and sharey make the plot cleaner by using common axes
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 4, nrows * 3),
        sharex=True,
        sharey=True
    )
    
    # Flatten the axes array to make it easy to loop over
    axes = axes.flatten()

    # Loop through the number of neurons you want to plot
    for i in range(num_to_plot):
        # Check if we have enough neurons in the data
        if i >= curves1.shape[0]:
            break # Stop if we run out of neurons

        ax = axes[i]
        
        # Plot the two tuning curves for the i-th neuron on the same subplot
        ax.plot(stimulus_axis, curves1[i, :], label=labels[0], color='dodgerblue')
        ax.plot(stimulus_axis, curves2[i, :], label=labels[1], color='orangered')
        
        # Add a title to each subplot to identify the neuron
        ax.set_title(f'Neuron {i}')
        ax.grid(True, linestyle='--', alpha=0.5)

    # --- Formatting for the whole figure ---
    # Add a single, shared legend to the figure to avoid clutter
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper right', fontsize=12)

    # Add shared x and y labels
    fig.supxlabel('Orenitation (degrees)', fontsize=16)
    fig.supylabel('Mean Firing Rate (z-scored)', fontsize=16)

    # Add a main title for the entire figure
    fig.suptitle(f'Tuning Curve Comparison for the First {num_to_plot} Neurons', fontsize=20)
    
    # Clean up the layout to prevent titles/labels from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for the suptitle

    # Hide any unused subplots if num_to_plot isn't a perfect multiple of ncols
    for i in range(num_to_plot, len(axes)):
        axes[i].set_visible(False)
        

def generate_inter_regional_comparison(matrices, layer_info_y, layer_info_x, epochs,
                                     region_y_name, region_x_name,
                                     vmin=-1, vmax=1,
                                     title='Inter-Regional Comparison'):
    """
    Generates a side-by-side comparison of inter-regional correlation matrices.

    Args:
        matrices (list of np.ndarray): A list of the correlation matrices to plot.
        layer_info_y (dict): Layer information dictionary for the Y-axis.
        layer_info_x (dict): Layer information dictionary for the X-axis.
        epochs (list of str): A list of names for each matrix (e.g., ['Saline', 'Drug']).
        region_y_name (str): The name of the region on the Y-axis (e.g., 'VISp').
        region_x_name (str): The name of the region on the X-axis (e.g., 'SSp').
        vmin (float, optional): Minimum value for the color scale. Defaults to -1.
        vmax (float, optional): Maximum value for the color scale. Defaults to 1.
        title (str, optional): The main title for the entire figure.
    
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    num_matrices = len(matrices)
    
    fig = plt.figure(figsize=(6 * num_matrices, 7))
    gs = GridSpec(1, num_matrices + 1, width_ratios=[40] * num_matrices + [1])

    axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, num_matrices):
        axes.append(fig.add_subplot(gs[0, i], sharey=axes[0]))

    cbar_ax = fig.add_subplot(gs[0, -1])

    y_layer_names = sorted(layer_info_y.keys())
    y_positions = [(layer_info_y[layer]['start_idx'] + layer_info_y[layer]['end_idx'] - 1) / 2 for layer in y_layer_names]
    
    x_layer_names = sorted(layer_info_x.keys())
    x_positions = [(layer_info_x[layer]['start_idx'] + layer_info_x[layer]['end_idx'] - 1) / 2 for layer in x_layer_names]

    im = None
    for i, correlation_matrix in enumerate(matrices):
        ax = axes[i]
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='auto')

        for layer in y_layer_names:
            start_idx = layer_info_y[layer]['start_idx']
            if start_idx > 0:
                ax.axhline(y=start_idx - 0.5, color='white', linewidth=2)

        for layer in x_layer_names:
            start_idx = layer_info_x[layer]['start_idx']
            if start_idx > 0:
                ax.axvline(x=start_idx - 0.5, color='white', linewidth=2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_layer_names, rotation=45, ha='right')
        # --- FIXED LINE ---
        ax.set_xlabel(f'Neurons in {region_x_name} (by Layer)')
        ax.set_title(f'Condition: {epochs[i]}')

    # --- FIXED LINE ---
    axes[0].set_ylabel(f'Neurons in {region_y_name} (by Layer)')
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_layer_names)

    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    fig.colorbar(im, cax=cbar_ax, label='Pearson Correlation')

    # --- UPDATED SUITE TITLE ---
    fig.suptitle(f'{title}: {region_y_name} to {region_x_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

