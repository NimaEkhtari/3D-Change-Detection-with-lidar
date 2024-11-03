# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:12:47 2024

@author: nekhtari
"""
from pyproj import Transformer
import pygmt
import rasterio
import numpy as np
import xarray as xr  # Import xarray for DataArray conversion
from simplekml import Kml, OverlayXY, ScreenXY, Units, RotationXY


files = ['trans_icp_results_0925_1018_150_25_norm_band_1.tif',
         'trans_icp_results_0925_1018_150_25_norm_band_2.tif',
         'trans_icp_results_0925_1018_150_25_norm_band_3.tif']


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

with rasterio.open(files[1]) as src:
    dY = src.read(1)  # Read the first band

with rasterio.open(files[2]) as src:
    dZ = src.read(1)  # Read the first band




# Check that all grids have the same shape
assert dX.shape == dY.shape == dZ.shape, "dX, dY, and dZ must have the same shape"

# Create X and Y coordinate arrays based on the shape of dZ
x = np.linspace(region[0], region[1], dZ.shape[1])
y = np.linspace(region[3], region[2], dZ.shape[0])

# Convert the dZ array into an xarray.DataArray
dZ_dataarray = xr.DataArray(
    dZ,
    coords={"y": y, "x": x},
    dims=["y", "x"]
)

# Define block size for averaging
block_size = 10

# Compute average displacements in 10x10 blocks, skipping np.nan values
def block_average(array, block_size):
    """Compute average in blocks while ignoring NaNs."""
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

# Compute the averaged X and Y coordinates, applying the mask
x_avg = x.reshape(-1, block_size).mean(axis=1)
y_avg = y.reshape(-1, block_size).mean(axis=1)
X_avg, Y_avg = np.meshgrid(x_avg, y_avg)
X_avg = X_avg[mask]
Y_avg = Y_avg[mask]

# Create a color palette for Z displacement from -3 to 3 units
# pygmt.makecpt(cmap="polar", series=[-2, 2, 0.1], output="polar_test.cpt")

# Initialize the PyGMT figure
fig = pygmt.Figure()

# Plot the Z displacement raster
fig.grdimage(
    grid=dZ_dataarray,
    region=region,
    projection="X10c/10c",  # 10x10 cm plot, adjust if needed
    cmap="polar_test.cpt",
    frame="af",
    nan_transparent=True
)

# Add a color bar for the Z displacement
fig.colorbar(cmap="polar_test.cpt", frame=["x+lZ Displacement"])

# Plot the quiver plot for the averaged X and Y displacements
angles = np.arctan2(dY_avg, dX_avg) * 180 / np.pi
magnis = np.linalg.norm(np.stack([dX_avg, dY_avg], axis = 1), axis = 1)
fig.plot(
    x=X_avg.flatten(),
    y=Y_avg.flatten(),
    direction=[angles, magnis],
    style="v0.2c+e",  # Vector style, arrow size 0.2 cm
    pen="0.75p,blue"
)

# fig.plot( x=[1000], y=[1000], direction=[[100], [100]], style="V0.25c+e", pen="0.5p,white", label="Velocity (cm/a)" )
# fig.legend(position="JRT+jRT+o0.2c", box="+gdarkgrey+p1p")


# Show the plot
fig.show()




''' ---------------------------------------------------------------------------------------- '''
# Define file names
output_image = "displacement_overlay.png"
output_kml = "displacement_0925-1018.kml"
# Save the figure as an image with geographic coordinates
fig.savefig(output_image)

# Load the GeoTIFF file and read CRS information
with rasterio.open(files[0]) as src:
    crs = src.crs  # Get CRS from the GeoTIFF file
    bounds = src.bounds  # Get geographic bounds in the source CRS

# Set up the transformer using the extracted CRS from the GeoTIFF
transformer = Transformer.from_crs(crs, "epsg:4262", always_xy=True)  # Convert from California State Plane to WGS84

# Convert bounds from California State Plane (in feet) to latitude and longitude
west, south = transformer.transform(bounds.left, bounds.bottom)
east, north = transformer.transform(bounds.right, bounds.top)

# Create the KML file with SimpleKML
kml = Kml()
ground = kml.newgroundoverlay(name="Displacement Overlay")
ground.icon.href = output_image  # Reference to the saved image

# Assign the geographic boundaries in lat/lon to the overlay
ground.latlonbox.north = north
ground.latlonbox.south = south
ground.latlonbox.east = east
ground.latlonbox.west = west

# Set altitude and altitudeMode to keep the overlay just above the ground
ground.altitude = 0.9144  # 3 feet converted to meters
ground.altitudemode = "relativeToGround"  # Keeps the overlay at a constant height above the terrain


# Center the overlay
ground.overlayxy = OverlayXY(x=0.5, y=0.5, xunits=Units.fraction, yunits=Units.fraction)
ground.screenxy = ScreenXY(x=0.5, y=0.5, xunits=Units.fraction, yunits=Units.fraction)
ground.rotationxy = RotationXY(x=0.5, y=0.5, xunits=Units.fraction)
ground.rotation = 0

# Save the KML file
kml.save(output_kml)


