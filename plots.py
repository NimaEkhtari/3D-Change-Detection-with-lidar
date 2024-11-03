# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:37:34 2024

@author: nekhtari
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:12:47 2024

@author: nekhtari
"""
import os
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import numpy as np
from simplekml import Kml, OverlayXY, ScreenXY, Units, RotationXY

sd = '0925'
ed = '1018'

files = [f'trans_icp_results_{sd}_{ed}_150_25_norm_band_1.tif',
         f'trans_icp_results_{sd}_{ed}_150_25_norm_band_2.tif',
         f'trans_icp_results_{sd}_{ed}_150_25_norm_band_3.tif']

output_image = f"displacement_{sd}_{ed}.png"
output_kml = f"displacement_{sd}_{ed}.kml"




# Load the GeoTIFF files using rasterio
with rasterio.open(files[0]) as src:
    dX = src.read(1)  # Read the first band
    dX_transform = src.transform  # Get transform for coordinates
    region = [
        src.bounds.left,
        src.bounds.right,
        src.bounds.bottom,
        src.bounds.top
    ]
    source = src.crs

with rasterio.open(files[1]) as src:
    dY = src.read(1)  # Read the first band

with rasterio.open(files[2]) as src:
    dZ = src.read(1)  # Read the first band




# Check that all grids have the same shape
assert dX.shape == dY.shape == dZ.shape, "dX, dY, and dZ must have the same shape"



# Define block size for averaging
block_size = 10

# Compute the averaged displacements and mask NaN values
def block_average(array, block_size):
    shape = (array.shape[0] // block_size, block_size,
             array.shape[1] // block_size, block_size)
    block_means = np.nanmean(array.reshape(shape), axis=(1, 3))
    return block_means

dX_avg = block_average(dX, block_size)
dY_avg = block_average(dY, block_size)

# Mask to ignore blocks with NaN values in either dX_avg or dY_avg
mask = ~np.isnan(dX_avg) & ~np.isnan(dY_avg)
dX_avg = dX_avg[mask]
dY_avg = dY_avg[mask]

# Generate the X and Y coordinate grids for the averaged displacement
x_coords = np.linspace(region[0], region[1], dX.shape[1] // block_size)
y_coords = np.linspace(region[2], region[3], dX.shape[0] // block_size)
X, Y = np.meshgrid(x_coords, y_coords)
X = X[mask]
Y = Y[mask]

# Plot the raster and arrows
fig, ax = plt.subplots(figsize=(10, 10))

# Display the dZ raster with a colormap (adjust limits as necessary)
cmap = plt.get_cmap("RdYlBu")
norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
img = ax.imshow(dZ, cmap=cmap, norm=norm, extent=region, origin='upper')

# Add color bar
cbar = plt.colorbar(img, ax=ax, label="Z Displacement (Feet)", orientation="horizontal", pad=0.2, shrink=0.6)

# Plot arrows for the averaged displacements
q = ax.quiver(X, Y, dX_avg, dY_avg, color='blue', scale=0.002, scale_units='xy', angles='xy', width = 0.006, headwidth=3, headlength=4, headaxislength=3)

# Add an arrow to the legend representing a 2-unit displacement
legend_arrow = ax.quiverkey(q, 0.45, -0.15, 2, "2 Feet", color="blue", labelpos="E", coordinates="axes")

# Add axis labels and title
ax.set_xlabel("X Coordinate (Feet)")
ax.set_ylabel("Y Coordinate (Feet)")
ax.set_title("Horizontal displacement Field Overlayed on vertical Displacement")

plt.show()

# Save the figure as an image with geographic coordinates
if os.path.exists(output_image):
    os.remove(output_image)
fig.savefig(output_image)

''' ---------------------------------------------------------------------------------------- '''


# Calculate pixel-based dimensions of the figure and the extent of margins
fig_width, fig_height = fig.get_size_inches() * fig.dpi  # Width and height in pixels
image_width, image_height = img.get_size()  # Inner image width and height in pixels

# Calculate margin size in pixels
margin_x_pixels_left = 263
margin_x_pixels_right = 237
margin_y_pixels_up = 120
margin_y_pixels_down = 380

# Calculate geographic units per pixel based on the original geographic extent
x_per_pixel = (region[1] - region[0]) / (fig_width - (margin_x_pixels_right + margin_x_pixels_left))
y_per_pixel = (region[3] - region[2]) / (fig_height - (margin_y_pixels_up + margin_y_pixels_down))

# Adjust bounds to include the margin in geographic units
west_adjusted = region[0] - (margin_x_pixels_left * x_per_pixel)
east_adjusted = region[1] + (margin_x_pixels_right * x_per_pixel)
south_adjusted = region[2] - (margin_y_pixels_down * y_per_pixel)
north_adjusted = region[3] + (margin_y_pixels_up * y_per_pixel)



# Set up the transformer using the extracted CRS from the GeoTIFF
transformer = Transformer.from_crs(source, "epsg:4326", always_xy=True)

# Convert adjusted bounds from California State Plane (in feet) to latitude and longitude
west, south = transformer.transform(west_adjusted, south_adjusted)
east, north = transformer.transform(east_adjusted, north_adjusted)






# Create the KML file with SimpleKML
kml = Kml()
ground = kml.newgroundoverlay(name=f"Displacement field {sd} - {ed}")
ground.icon.href = output_image  # Reference to the saved image

# Add a description to the overlay
ground.description = f"Displacement map for {sd} - {ed} period. Arrows show horizontal displacement, raster values show vertical displacement. Values are given in feet."

# Assign the geographic boundaries in lat/lon to the overlay
ground.latlonbox.north = north
ground.latlonbox.south = south
ground.latlonbox.east = east
ground.latlonbox.west = west

# Set altitude and altitudeMode to keep the overlay just above the ground
ground.altitude = 0  # 3 feet converted to meters
ground.altitudemode = "relativeToGround"  # Keeps the overlay at a constant height above the terrain


# Center the overlay
ground.overlayxy = OverlayXY(x=0.5, y=0.5, xunits=Units.fraction, yunits=Units.fraction)
ground.screenxy = ScreenXY(x=0.5, y=0.5, xunits=Units.fraction, yunits=Units.fraction)
ground.rotationxy = RotationXY(x=0.5, y=0.5, xunits=Units.fraction)
ground.rotation = 0

# Save the KML file
kml.save(output_kml)


