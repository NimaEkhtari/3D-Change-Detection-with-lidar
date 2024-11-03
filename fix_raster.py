# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:17:22 2024

@author: nekhtari
"""

import rasterio
import numpy as np

# Specify the input file path
input_file = 'trans_icp_results_0925_1018_150_25.tif'
const = 'norm'

# Define the values you want to drop for each layer
# drop_values = [-0.219395167,-0.064641833,-0.386200167]  # For 0906-0925
# drop_values = [0.194423167, -0.4835325, -0.389634667]  # For 0906-1018
drop_values = [0.274856,	-0.4731875,	-0.065748167]  # For 0925-1018

# Open the input GeoTIFF file
with rasterio.open(input_file) as src:
    # Copy metadata for creating the output files
    metadata = src.meta.copy()

    # Update metadata to handle NaN values as no data
    metadata.update({"dtype": "float64", "nodata": np.nan, "count": 1})

    # Iterate over each layer
    for i in range(src.count):  # src.count gives the number of layers/bands
        layer = src.read(i + 1)  # Read each layer, indexing starts from 1 in rasterio
        drop_value = drop_values[i]  # Get the corresponding drop value

        # Adjust the layer
        layer = layer - drop_value
        layer[(layer < -3) | (layer > 3)] = np.nan

        # Update output file name for each layer
        output_file = f'{input_file}_{const}_band_{i + 1}.tif'

        # Write each modified layer to its own file
        with rasterio.open(output_file, 'w', **metadata) as dst:
            dst.write(layer, 1)  # Write the single band

        print(f"Modified band {i + 1} saved successfully as {output_file}")

