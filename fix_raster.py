# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:17:22 2024

@author: nekhtari
"""

import rasterio
import numpy as np

# Specify the input and output file paths
input_file = 'trans_icp_results_0925_1018_1_150_50.tif'
output_file = 'trans_icp_results_0925_1018_1_150_50_norm.tif'

# Define the values you want to drop for each layer
drop_values = [0.194423167, -0.4835325,	-0.389634667]  # Replace with the specific values for each layer

# Open the input GeoTIFF file
with rasterio.open(input_file) as src:
    # Copy metadata for creating the output file later
    metadata = src.meta.copy()
    
    # Initialize an empty array to store modified data
    modified_data = np.empty_like(src.read())

    # Iterate over each layer
    for i in range(src.count):  # src.count gives the number of layers/bands
        layer = src.read(i + 1)  # Read each layer, indexing starts from 1 in rasterio
        drop_value = drop_values[i]  # Get the corresponding drop value
        
        # Set values equal to drop_value to NaN
        layer = layer - drop_value
        
        # Set values less than -3 and greater than 3 to NaN
        layer[(layer < -3) | (layer > 3)] = np.nan
        
        # Store the modified layer
        modified_data[i] = layer

# Update metadata to handle NaN values as no data
metadata.update({"dtype": "float64", "nodata": np.nan})

# Write the modified data to a new GeoTIFF file
with rasterio.open(output_file, 'w', **metadata) as dst:
    dst.write(modified_data)

print("Modified GeoTIFF saved successfully as", output_file)
