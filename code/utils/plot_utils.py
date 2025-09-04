import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns




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
    plt.show()
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
    plt.show()
    return



def generate_matrices_comparison(matrices,layer_info,
                                 unique_layers,epochs,title = 'Name this graph'):
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
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

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
    plt.show()
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
    plt.show()