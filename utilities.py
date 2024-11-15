# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:02:00 2024

@author: nekhtari
"""

import json
import pdal
import os
import numpy as np


# Function to be used with GPU-parallel processing
def extract_and_save_tile(x, y, tile_size, margin, B, A, N, output_dir):
    """
    Extracts a tile with a margin and saves it to disk as a .npy file.
    
    Parameters:
    - x, y: Bottom-right corner of the tile (smallest x and y coordinates of the tile).
    - tile_size: Size of the tile in units.
    - margin: Additional margin to add around the tile.
    - B: Numpy array of pre-event 3D coordinates.
    - A: Numpy array of post-event 3D coordinates.
    - N: Numpy array of normal vectors for post-event points.
    - output_dir: Directory to save the extracted tiles.
    
    Returns:
    - The paths to the saved .npy files for B, A, and N.
    """
    # Define boundaries for filtering
    x_min = x - margin
    x_max = x + tile_size + margin
    y_min = y - margin
    y_max = y + tile_size + margin

    # Apply logical indexing to select points within the boundary
    b = B[(B[:, 0] >= x_min) & (B[:, 0] <= x_max) & (B[:, 1] >= y_min) & (B[:, 1] <= y_max)]
    mask = (A[:, 0] >= x_min) & (A[:, 0] <= x_max) & (A[:, 1] >= y_min) & (A[:, 1] <= y_max)
    a = A[mask]
    n = N[mask]

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each tile with a unique filename based on its coordinates
    tile_id = f"x{x}_y{y}"
    b_path = os.path.join(output_dir, f"B_{tile_id}.npy")
    a_path = os.path.join(output_dir, f"A_{tile_id}.npy")
    n_path = os.path.join(output_dir, f"N_{tile_id}.npy")

    np.save(b_path, b)
    np.save(a_path, a)
    np.save(n_path, n)

    return b_path, a_path, n_path





def get_metadata(file):
    if file.endswith('json'):
        reader = 'readers.ept'
    elif file.endswith('laz'):
        reader = 'readers.las'
    
    pipeline = {
        "pipeline": [
            {
                "type": reader,
                "filename": file
            },
            {
                "type":"filters.decimation",
                "step": 100
            },
            {
                "type": "filters.stats",  # Add filter to calculate statistics
                "dimensions": "X,Y"  # Specify X and Y dimensions for which stats are calculated
            }
        ]
    }
    
    pipeline_obj = pdal.Pipeline(json.dumps(pipeline))
    pipeline_obj.execute()
    
    # Directly use pipeline_obj.metadata since it's already a dict
    metadata = pipeline_obj.metadata
    
    Xmin = metadata['metadata']['filters.stats']['statistic'][0]['minimum']
    Xmax = metadata['metadata']['filters.stats']['statistic'][0]['maximum']
    Ymin = metadata['metadata']['filters.stats']['statistic'][1]['minimum']
    Ymax = metadata['metadata']['filters.stats']['statistic'][1]['maximum']
    bounds = [Xmin, Xmax, Ymin, Ymax]
    
    dimensions = [field[0] for field in pipeline_obj.arrays[0].dtype.descr]
    if 'NormalX' in dimensions:
        has_normal = True
    else:
        has_normal = False

    
    return (bounds, has_normal)




    
    
    
    
    
    
    
    
    