# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:56:16 2024

@author: nekhtari

Runs the moving-window translation-only ICP between two (or more) lidar point
clouds and writes dx, dy, dz as a text file and a GeoTIFF.

From Spyder or any IDE: set the parameters below and run the file.
From a terminal:        python main.py before.laz after.laz --window 150 --step 25
                        (or, after pip install, just: ticp before.laz after.laz ...)
Values given on the command line override the ones set below.
"""

from ticp import cli


''' ------------------------ Set the following parameters ------------------------ '''
# Point clouds in time order. With more than two, every consecutive pair is
# processed (1 -> 2, 2 -> 3, ...). LAS/LAZ, or ept.json if PDAL is installed.
POINT_CLOUDS = [
    r'D:\SCE\Data\LAZ_Classified\Klondike_LAS_20240925_ModelKey_Bldgs_V2.laz',
    r'D:\SCE\Data\LAZ_Classified\Klondike_LAS_20241018_ModelKey_Bldgs_V2.laz',
]
OUTPUT_DIR = 'results'
OUTPUT_PREFIX = 'ticp_'       # outputs are named <prefix><before>_to_<after>_<window>_<step>

WINDOW_SIZE = 150             # ICP window size, in the units of the point clouds
STEP_SIZE = 25                # spacing of the windows = pixel size of the output raster
MARGIN = 3                    # extra buffer around the post-event window
CLASSES = [2, 6, 8]           # LAS classes to use. None = all points

MIN_POINTS = 20
CONVERGENCE = 0.0005
MAX_ITER = 20
OUTLIER_THRESHOLD = 3         # in multiples of the MAD of the residuals
NORMAL_KNN = 8
BOUNDS = None                 # [xmin, xmax, ymin, ymax] to process only part of the area
N_WORKERS = None              # None = all cores but one. Use 1 to debug in the IDE
CRS = None                    # e.g. 'EPSG:6424' if the point clouds have no CRS stored
PLOT = False                  # True = quick plot of the results at the end (handy in Spyder)
''' ------------------------------------------------------------------------------ '''



if __name__ == '__main__':
    cli.main(defaults=dict(
        point_clouds=POINT_CLOUDS,
        out_dir=OUTPUT_DIR,
        prefix=OUTPUT_PREFIX,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        margin=MARGIN,
        classes=CLASSES,
        min_points=MIN_POINTS,
        convergence=CONVERGENCE,
        max_iter=MAX_ITER,
        outlier_threshold=OUTLIER_THRESHOLD,
        normal_knn=NORMAL_KNN,
        bounds=BOUNDS,
        n_workers=N_WORKERS,
        crs=CRS,
        plot=PLOT,
    ))
