# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:56:16 2024

@author: nekhtari
"""

import numpy as np
import pdal
import json
import icp
import utilities
import matplotlib.pyplot as plt
import numpy.ma as ma


operation = 'ticp'
# Path to pre- and post-event indexed point clouds (EPT)
pre_event = r'D:\Working\SCE\Landslide\Data\KlondikeCanyon\Entwine\20240906\ModelKey\ept.json'
pos_event = r'D:\Working\SCE\Landslide\Data\KlondikeCanyon\Entwine\20241018\ModelKey\ept.json'


bounds_pre, has_normals_pre = utilities.get_metadata(pre_event)
bounds_pos, has_normals_pos = utilities.get_metadata(pos_event)

bounds = []
bounds.append(int(min(bounds_pre[0], bounds_pos[0])))
bounds.append(int(max(bounds_pre[1], bounds_pos[1])))
bounds.append(int(min(bounds_pre[2], bounds_pos[2])))
bounds.append(int(max(bounds_pre[3], bounds_pos[3])))



operation = 'translation_only'
classes = [8]

# bounds = [6451884, 6452800, 1726600, 1727600]
# has_normals_pos = False




if operation == 'translation_only':
    configs = {
    'classes': classes,
    'bounds' : bounds,
    'method' : 'translation_only',
    'threshold' : 20,
    'window_size' : 150,
    'step_size' : 50,
    'margin': 15,
    'min_points' : 500,
    'Tconverge' : 0.0005,
    'Tmax_iter' : 20,
    'outlier_multiplier' : 5,
    'outlier_percent' : 0.95,
    'has_normal_post' : has_normals_pos,
    'null': -99,
    'output_basename' : 'trans_icp_results'
    }

    config = icp.icp_configs(configs)
    res, disp = icp.run_transicp(pre_event, pos_event, config)



''' ------------------------------------------------------------------------------------- '''
'''                          Plotting -------------- '''

# Calculate the length of each displacement vector
d = np.linalg.norm(res[:, 2:4], axis=1)

# Filter out displacements longer than 1 meter
mask = d <= 2.0
filtered_X = res[mask, 0]
filtered_Y = res[mask, 1]
filtered_delta_X = res[mask, 2]
filtered_delta_Y = res[mask, 3]

# Plot the ICP vectors
plt.figure()
plt.quiver(filtered_X, filtered_Y, filtered_delta_X, filtered_delta_Y, angles='xy',
           scale_units='xy', headwidth=2.5, headlength=4)
plt.axis('equal')
plt.show()





def plot_all(res, disp, Th):
    
    # Calculate dh and apply filtering
    dx, dy, dz = disp[:, :, 0], disp[:, :, 1], disp[:, :, 2]
    dh = np.sqrt(dx**2 + dy**2)
    
    # Set dx, dy, dz values to -99 where dh > 2.5 meters
    dx[dh > Th] = -99
    dy[dh > Th] = -99
    dz[dh > Th] = -99
    
    # Mask the -99 values
    dx = ma.masked_equal(dx, -99)
    dy= ma.masked_equal(dy, -99)
    dz = ma.masked_equal(dz, -99)
    
    
    # Set up a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot dx raster
    cax1 = axs[0, 0].imshow(dx, cmap='viridis', origin='lower')
    axs[0, 0].set_title('dx across area')
    fig.colorbar(cax1, ax=axs[0, 0])
    
    # Plot dy raster
    cax2 = axs[0, 1].imshow(dy, cmap='viridis', origin='lower')
    axs[0, 1].set_title('dy across area')
    fig.colorbar(cax2, ax=axs[0, 1])
    
    # Plot dz raster
    cax3 = axs[1, 0].imshow(dz, cmap='viridis', origin='lower')
    axs[1, 0].set_title('dz across area')
    fig.colorbar(cax3, ax=axs[1, 0])
    
    
    
    
    # Calculate the length of each displacement vector
    d = np.linalg.norm(res[:, 2:4], axis=1)

    # Filter out displacements longer than 1 meter
    mask = d <= Th
    filtered_X = res[mask, 0]
    filtered_Y = res[mask, 1]
    filtered_delta_X = res[mask, 2]
    filtered_delta_Y = res[mask, 3]
    
    # Plot quiver plot in the fourth subplot
    axs[1, 1].quiver(filtered_X, filtered_Y, filtered_delta_X, filtered_delta_Y, angles='xy',
               scale_units='xy', cmap='viridis', headwidth=2.5, headlength=4)
    axs[1, 1].set_title('Displacement vectors (filtered)')
    
    plt.tight_layout()
    plt.show()


plot_all(res, disp, 2)







import rasterio
from rasterio.transform import from_origin

def write_disp_rasters(disp, output_file, transform, crs="EPSG:6424"):
    # Extract dx, dy, dz from the disp matrix
    dx, dy, dz = disp[:, :, 0], disp[:, :, 1], disp[:, :, 2]

    # Set up metadata for the GeoTIFF
    height, width = dx.shape
    metadata = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 3,  # Three bands for dx, dy, dz
        'width': width,
        'height': height,
        'crs': crs,  # Set CRS to EPSG:6424
        'transform': transform  # Affine transform
    }

    # Write the three bands to a GeoTIFF
    with rasterio.open(output_file, 'w', **metadata) as dst:
        dst.write(dx, 1)  # Write dx as the first band
        dst.write(dy, 2)  # Write dy as the second band
        dst.write(dz, 3)  # Write dz as the third band

    print(f"Raster written to {output_file} with CRS {crs}")

# Example usage
# Assume the disp matrix and the upper-left corner coordinates and pixel size are known

transform = from_origin(min(res[:, 0]), max(res[:, 1]), configs['step_size'], configs['step_size'])  # Replace with your actual top-left coordinates and pixel size

write_disp_rasters(np.flipud(disp), "displacement_layers_ticp_150_50.tif", transform)













