# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:17:22 2024

@author: nekhtari

Removes a constant offset from each band of a TICP result raster.

Even after georeferencing, two lidar surveys are usually off by a few
centimeters to decimeters as a whole. That offset shows up in every window, so
it is subtracted here to leave only the local change. The offset of each band is
best measured over an area that is known to be stable (average dx, dy, dz there).
If no offsets are given, the median of each band is used, which works when most
of the area did not move.

Values that are still larger than MAX_ABS after the correction are set to NaN.
The output is written next to the input as <name>_norm.tif.

Spyder / IDE: set the parameters below and run the file.
Terminal:     python tools/remove_bias.py results/ticp_a_to_b_150_25.tif --offsets -0.22 -0.06 -0.39
"""

import argparse
import numpy as np
import rasterio


''' ------------------------ Set the following parameters ------------------------ '''
INPUT_FILE = r'results/ticp_20240906_to_20240925_150_25.tif'
OFFSETS = None                # [dx, dy, dz] to subtract, e.g. [-0.219, -0.065, -0.386]. None = band medians
MAX_ABS = 3                   # values beyond +/- this after the correction become NaN. None = keep all
''' ------------------------------------------------------------------------------ '''



def remove_bias(input_file, offsets=None, max_abs=None):
    output_file = input_file.rsplit('.', 1)[0] + '_norm.tif'

    with rasterio.open(input_file) as src:
        profile = src.profile.copy()
        profile.update(dtype='float32', nodata=np.nan)
        data = src.read().astype('float32')
        descriptions = src.descriptions
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    if offsets is None:
        offsets = [float(np.nanmedian(data[i])) for i in range(3)]

    # Only dx, dy and dz are corrected; the rmse band (4) is copied as it is
    for i, offset in enumerate(offsets):
        data[i] -= offset
        if max_abs is not None:
            data[i][np.abs(data[i]) > max_abs] = np.nan
        print(f'Band {i + 1} ({descriptions[i]}): subtracted {offset:.3f}')

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data)
        for i, d in enumerate(descriptions, start=1):
            if d:
                dst.set_band_description(i, d)

    print(f'Corrected raster saved as {output_file}')
    return output_file



if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Subtract a constant dx, dy, dz offset from a TICP raster.')
    p.add_argument('input_file', nargs='?', default=INPUT_FILE)
    p.add_argument('--offsets', type=float, nargs=3, default=OFFSETS, metavar=('DX', 'DY', 'DZ'))
    p.add_argument('--max-abs', type=float, default=MAX_ABS)
    args = p.parse_args()

    remove_bias(args.input_file, args.offsets, args.max_abs)
