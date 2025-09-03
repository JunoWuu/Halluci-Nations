import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_3Dcells(x_coords,y_coords,z_coords,cell_labels):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # 2. Map unique string labels to distinct colors
    unique_labels = np.unique(cell_labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # 3. Plot each group of points with its assigned color
    for label in unique_labels:
        # Find the indices of the points belonging to the current label
        indices = np.where(cell_labels == label)

        # Plot these points with the correct color and label for the legend
        ax.scatter(x_coords[indices], y_coords[indices], z_coords[indices],
                   color=color_map[label], label=label, s=50, alpha=0.8)

    ax.set_zlim
    # 4. Customize the plot
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Scatter Plot of Cell Positions by Type')
    ax.legend(title="Cell Types")
    ax.grid(True)
    return



def raster_bny_cell_type(spike_times, cell_types):
    unique_labels = np.unique(cell_labels)
    cmap = plt.cm.get_cmap('tab10', len(unique_labels)) # A colormap good for distinct categories
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}


    # 2. Create a list of colors, one for each neuron, based on its label
    neuron_colors = [color_map[label] for label in cell_labels]


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



def generate_matrices_comparison(matrices,layer_info,unique_layers,epochs):
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

    fig.suptitle('SSps Correlation Matrices Organized by Layer', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return