import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D



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