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

Values that are still larger than max_abs after the correction are set to NaN.
The output is written next to the input as <name>_norm.tif.

Installed as the `ticp-remove-bias` command:
    ticp-remove-bias results/ticp_a_to_b_150_25.tif --offsets -0.22 -0.06 -0.39
"""

import sys
import logging
import argparse
import numpy as np
import rasterio

log = logging.getLogger(__name__)



def remove_bias(input_file, offsets=None, max_abs=None):
    """ Returns the name of the corrected raster. """
    output_file = input_file.rsplit('.', 1)[0] + '_norm.tif'

    with rasterio.open(input_file) as src:
        profile = src.profile.copy()
        profile.update(dtype='float32', nodata=np.nan, compress='lzw')
        for key in ['blockxsize', 'blockysize', 'tiled']:   # let GDAL pick these again
            profile.pop(key, None)
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
        log.info(f'Band {i + 1} ({descriptions[i]}): subtracted {offset:.3f}')

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data)
        for i, d in enumerate(descriptions, start=1):
            if d:
                dst.set_band_description(i, d)

    log.info(f'Corrected raster saved as {output_file}')
    return output_file



def main(argv=None, defaults=None):
    d = dict(input_file=None, offsets=None, max_abs=3)
    d.update(defaults or {})
    p = argparse.ArgumentParser(prog='ticp-remove-bias',
                                description='Subtract a constant dx, dy, dz offset from a TICP raster.')
    p.add_argument('input_file', nargs='?' if d['input_file'] else None, default=d['input_file'])
    p.add_argument('--offsets', type=float, nargs=3, default=d['offsets'], metavar=('DX', 'DY', 'DZ'),
                   help='offsets to subtract (default: median of each band)')
    p.add_argument('--max-abs', type=float, default=d['max_abs'],
                   help='values beyond +/- this after the correction become NaN')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

    remove_bias(args.input_file, args.offsets, args.max_abs)



if __name__ == '__main__':
    main()
