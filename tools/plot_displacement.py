# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:37:34 2024

@author: nekhtari

Map of a TICP result: vertical change (dz) as a colored raster with the
horizontal change (dx, dy) as arrows on top. The arrows are averaged over
blocks of BLOCK x BLOCK pixels so the map does not get too busy.

Optionally writes a KML ground overlay to look at the result in Google Earth
(needs pyproj and simplekml).

Spyder / IDE: set the parameters below and run the file.
Terminal:     python tools/plot_displacement.py results/ticp_a_to_b_150_25_norm.tif --kml
"""

import os
import argparse
import warnings
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


''' ------------------------ Set the following parameters ------------------------ '''
INPUT_FILE = r'results/ticp_20240906_to_20240925_150_25_norm.tif'
BLOCK = 10                    # average the arrows over BLOCK x BLOCK pixels
DZ_LIMIT = 2                  # color scale of dz goes from -DZ_LIMIT to +DZ_LIMIT. None = automatic
ARROW_KEY = 2                 # length of the reference arrow in the legend. None = automatic
UNITS = 'ft'
WRITE_KML = False
''' ------------------------------------------------------------------------------ '''



def block_average(array, block):
    """ nanmean over block x block pixels, the edges are padded with NaN """
    rows = -(-array.shape[0] // block) * block
    cols = -(-array.shape[1] // block) * block
    padded = np.full((rows, cols), np.nan)
    padded[:array.shape[0], :array.shape[1]] = array
    blocks = padded.reshape(rows // block, block, cols // block, block)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # blocks with only NaN
        return np.nanmean(blocks, axis=(1, 3))



def read_bands(file):
    with rasterio.open(file) as src:
        dX, dY, dZ = src.read([1, 2, 3]).astype(float)
        if src.nodata is not None and not np.isnan(src.nodata):
            for b in (dX, dY, dZ):
                b[b == src.nodata] = np.nan
        b = src.bounds
        return dX, dY, dZ, [b.left, b.right, b.bottom, b.top], src.res[0], src.crs



def arrows(dX, dY, region, pixel, block):
    """ Block-averaged arrows and the coordinates of the block centers """
    u = block_average(dX, block)
    v = block_average(dY, block)
    xc = region[0] + (np.arange(u.shape[1]) + 0.5) * block * pixel
    yc = region[3] - (np.arange(u.shape[0]) + 0.5) * block * pixel     # row 0 is north
    X, Y = np.meshgrid(xc, yc)
    ok = np.isfinite(u) & np.isfinite(v)
    return X[ok], Y[ok], u[ok], v[ok]



def draw(ax, dZ, X, Y, U, V, region, dz_limit, scale):
    if dz_limit is None:
        dz_limit = np.nanpercentile(np.abs(dZ), 98)
    norm = mcolors.TwoSlopeNorm(vmin=-dz_limit, vcenter=0, vmax=dz_limit)
    img = ax.imshow(dZ, cmap='RdBu', norm=norm, extent=region, origin='upper', interpolation='nearest')
    q = ax.quiver(X, Y, U, V, color='black', angles='xy', scale_units='xy', scale=scale,
                  width=0.004, headwidth=3, headlength=4, headaxislength=3)
    return img, q



def plot_displacement(input_file, block=10, dz_limit=2, arrow_key=2, units='ft', write_kml=False):
    dX, dY, dZ, region, pixel, crs = read_bands(input_file)
    X, Y, U, V = arrows(dX, dY, region, pixel, block)

    # Arrow scale: the largest arrows (95th percentile) are about one block long
    h = np.hypot(U, V)
    big = np.percentile(h, 95) if len(h) else 0
    scale = big / (block * pixel) if big > 0 else 1
    if arrow_key is None:
        arrow_key = float(f'{big:.1g}') if big > 0 else 1

    fig, ax = plt.subplots(figsize=(10, 10))
    img, q = draw(ax, dZ, X, Y, U, V, region, dz_limit, scale)
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    fig.colorbar(img, ax=ax, label=f'Vertical displacement ({units})',
                 orientation='horizontal', pad=0.08, shrink=0.6)
    ax.quiverkey(q, 0.80, -0.08, arrow_key, f'{arrow_key:g} {units}', labelpos='E', coordinates='axes')
    ax.set_xlabel(f'X ({units})')
    ax.set_ylabel(f'Y ({units})')
    ax.set_title('Horizontal displacement over vertical displacement', loc='left')
    ax.ticklabel_format(useOffset=False, style='plain')

    output_image = input_file.rsplit('.', 1)[0] + '.png'
    fig.savefig(output_image, dpi=200, bbox_inches='tight')
    print(f'Figure saved as {output_image}')

    if write_kml:
        write_overlay_kml(input_file, dZ, X, Y, U, V, region, dz_limit, scale, crs, units)
    plt.show()



def write_overlay_kml(input_file, dZ, X, Y, U, V, region, dz_limit, scale, crs, units):
    """
    Google Earth overlay. The image is drawn without any axes or margins so its
    corners are exactly the corners of the raster (no manual margin tuning).
    """
    from pyproj import Transformer
    from simplekml import Kml

    w, h = region[1] - region[0], region[3] - region[2]
    fig = plt.figure(figsize=(10, 10 * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    draw(ax, dZ, X, Y, U, V, region, dz_limit, scale)
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.axis('off')
    overlay_image = input_file.rsplit('.', 1)[0] + '_overlay.png'
    fig.savefig(overlay_image, dpi=200, transparent=True)
    plt.close(fig)

    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    west, south = transformer.transform(region[0], region[2])
    east, north = transformer.transform(region[1], region[3])

    name = os.path.basename(input_file).rsplit('.', 1)[0]
    kml = Kml()
    ground = kml.newgroundoverlay(name=name)
    ground.icon.href = os.path.basename(overlay_image)
    ground.description = (f'Arrows show horizontal displacement, colors show vertical '
                          f'displacement. Values are in {units}.')
    ground.latlonbox.north, ground.latlonbox.south = north, south
    ground.latlonbox.east, ground.latlonbox.west = east, west
    output_kml = input_file.rsplit('.', 1)[0] + '.kml'
    kml.save(output_kml)
    print(f'KML saved as {output_kml} (keep it next to {os.path.basename(overlay_image)})')



if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Plot a TICP result raster (dz colors + dx/dy arrows).')
    p.add_argument('input_file', nargs='?', default=INPUT_FILE)
    p.add_argument('--block', type=int, default=BLOCK)
    p.add_argument('--dz-limit', type=float, default=DZ_LIMIT)
    p.add_argument('--arrow-key', type=float, default=ARROW_KEY)
    p.add_argument('--units', default=UNITS)
    p.add_argument('--kml', action=argparse.BooleanOptionalAction, default=WRITE_KML)
    args = p.parse_args()

    plot_displacement(args.input_file, args.block, args.dz_limit, args.arrow_key, args.units, args.kml)
