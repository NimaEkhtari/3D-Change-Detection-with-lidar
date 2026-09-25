# -*- coding: utf-8 -*-
"""
@author: nekhtari

Map of a TICP result: dz as colors with block-averaged dx/dy arrows on top,
optionally also a KML overlay for Google Earth (see ticp/plotting.py).

Spyder / IDE: set the parameters below and run the file.
Terminal:     ticp-plot results/ticp_a_to_b_150_25_norm.tif --block 10 --kml
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # find ticp without installing
from ticp import plotting


''' ------------------------ Set the following parameters ------------------------ '''
INPUT_FILE = r'results/ticp_20240906_to_20240925_150_25_norm.tif'
BLOCK = 10                    # average the arrows over BLOCK x BLOCK pixels
DZ_LIMIT = 2                  # color scale of dz goes from -DZ_LIMIT to +DZ_LIMIT. None = automatic
ARROW_KEY = 2                 # length of the reference arrow in the legend. None = automatic
UNITS = 'ft'
WRITE_KML = False             # needs simplekml
''' ------------------------------------------------------------------------------ '''


if __name__ == '__main__':
    plotting.main(defaults=dict(input_file=INPUT_FILE, block=BLOCK, dz_limit=DZ_LIMIT,
                                arrow_key=ARROW_KEY, units=UNITS, kml=WRITE_KML))
