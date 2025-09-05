import numpy as np
from collections import defaultdict


def create_receptive_field(neuron_firing_rates, exclude_key=(7777, 7777, 7777)):
    """
    Transforms firing rates into receptive field maps, ignoring a specific key.

    Args:
        neuron_firing_rates (dict): A dictionary where keys are tuples of
            (orientation, posx, posy) and values are NumPy arrays of
            mean firing rates for each neuron.
        exclude_key (tuple, optional): The specific key to ignore during analysis.
            Defaults to (7777, 7777, 7777).

    Returns:
        tuple: A tuple containing:
            - receptive_field_maps (dict): Maps for each neuron.
            - sorted_posx (list): X-coordinates for the map columns.
            - sorted_posy (list): Y-coordinates for the map rows.
    """
    # --- ADDITION: Filter the data once at the very beginning ---
    filtered_items = [
        (k, v) for k, v in neuron_firing_rates.items() if k != exclude_key
    ]
    # -----------------------------------------------------------

    if not filtered_items:
        return {}, [], []

    # Step 1: Discover coordinates using only the filtered data.
    posx_coords = set()
    posy_coords = set()
    for key, _ in filtered_items:
        _, posx, posy = key
        posx_coords.add(posx)
        posy_coords.add(posy)

    # Step 2: Create the canonical sorted arrangement.
    sorted_posx = sorted(list(posx_coords))
    sorted_posy = sorted(list(posy_coords))
    posx_to_col = {x: i for i, x in enumerate(sorted_posx)}
    posy_to_row = {y: i for i, y in enumerate(sorted_posy)}

    # Step 3: Gather data using only the filtered data.
    intermediate_data = defaultdict(lambda: defaultdict(list))
    # Get number of neurons from the first valid item
    num_neurons = len(filtered_items[0][1])

    # Iterate over the clean list, not the original dictionary
    for key, rates in filtered_items:
        _, posx, posy = key
        for neuron_idx in range(num_neurons):
            intermediate_data[neuron_idx][(posx, posy)].append(rates[neuron_idx])

    # Step 4: Build the final receptive field maps.
    receptive_field_maps = {}
    map_shape = (len(sorted_posy), len(sorted_posx))
    for neuron_idx, pos_data in intermediate_data.items():
        rf_map = np.full(map_shape, np.nan)
        for (posx, posy), rates in pos_data.items():
            row = posy_to_row[posy]
            col = posx_to_col[posx]
            rf_map[row, col] = np.max(rates) # Uses max, can be changed to np.mean
        receptive_field_maps[neuron_idx] = rf_map

    return receptive_field_maps, sorted_posx, sorted_posy