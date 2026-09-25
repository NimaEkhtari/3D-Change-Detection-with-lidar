# -*- coding: utf-8 -*-
"""
@author: nekhtari

Subtracts a constant dx, dy, dz offset from a TICP result raster
(see ticp/bias.py for the details).

Spyder / IDE: set the parameters below and run the file.
Terminal:     ticp-remove-bias results/ticp_a_to_b_150_25.tif --offsets -0.22 -0.06 -0.39
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # find ticp without installing
from ticp import bias


''' ------------------------ Set the following parameters ------------------------ '''
INPUT_FILE = r'results/ticp_20240906_to_20240925_150_25.tif'
OFFSETS = None                # [dx, dy, dz] to subtract, e.g. [-0.219, -0.065, -0.386]. None = band medians
MAX_ABS = 3                   # values beyond +/- this after the correction become NaN. None = keep all
''' ------------------------------------------------------------------------------ '''


if __name__ == '__main__':
    bias.main(defaults=dict(input_file=INPUT_FILE, offsets=OFFSETS, max_abs=MAX_ABS))
