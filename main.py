# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:56:16 2024

@author: nekhtari

Runs the moving-window translation-only ICP between two (or more) lidar point
clouds and writes dx, dy, dz as a text file and a GeoTIFF.

From Spyder or any IDE: set the parameters below and run the file.
From a terminal:        python main.py before.laz after.laz --window 150 --step 25
                        (python main.py --help lists all options)
Values given on the command line override the ones set below.
"""

import os
import time
import argparse
import icp
import utilities


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



def parse_args():
    p = argparse.ArgumentParser(description='Moving-window translation-only ICP for 3D change detection.')
    p.add_argument('point_clouds', nargs='*', default=POINT_CLOUDS,
                   help='two or more point clouds in time order')
    p.add_argument('--out-dir', default=OUTPUT_DIR)
    p.add_argument('--prefix', default=OUTPUT_PREFIX)
    p.add_argument('--window', type=float, default=WINDOW_SIZE, help='window size')
    p.add_argument('--step', type=float, default=STEP_SIZE, help='step between windows')
    p.add_argument('--margin', type=float, default=MARGIN)
    p.add_argument('--classes', type=int, nargs='*', default=CLASSES,
                   help='LAS classes to use, e.g. --classes 2 6 (just --classes = all points)')
    p.add_argument('--min-points', type=int, default=MIN_POINTS)
    p.add_argument('--convergence', type=float, default=CONVERGENCE)
    p.add_argument('--max-iter', type=int, default=MAX_ITER)
    p.add_argument('--outlier-threshold', type=float, default=OUTLIER_THRESHOLD)
    p.add_argument('--normal-knn', type=int, default=NORMAL_KNN)
    p.add_argument('--bounds', type=float, nargs=4, default=BOUNDS,
                   metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    p.add_argument('--workers', type=int, default=N_WORKERS)
    p.add_argument('--crs', default=CRS)
    p.add_argument('--plot', action=argparse.BooleanOptionalAction, default=PLOT)
    return p.parse_args()



def fmt(v):
    """ 150.0 -> '150', 2.5 -> '2.5' for the file names """
    return f'{v:g}'



if __name__ == '__main__':
    args = parse_args()
    if len(args.point_clouds) < 2:
        raise SystemExit('Need at least two point clouds (before and after)')

    config = icp.ICPConfig(
        window_size=args.window,
        step_size=args.step,
        margin=args.margin,
        classes=args.classes or None,
        min_points=args.min_points,
        convergence=args.convergence,
        max_iter=args.max_iter,
        outlier_threshold=args.outlier_threshold,
        normal_knn=args.normal_knn,
        bounds=args.bounds,
        n_workers=args.workers,
        crs=args.crs,
    )
    os.makedirs(args.out_dir, exist_ok=True)

    events = args.point_clouds
    for i in range(len(events) - 1):
        pre_event, pos_event = events[i], events[i + 1]
        print(f'\n{pre_event}\n  -> {pos_event}')
        start_time = time.time()

        print('Reading point clouds ...')
        before, _ = utilities.read_point_cloud(pre_event, config.classes)
        after, normals = utilities.read_point_cloud(pos_event, config.classes)
        print(f'{len(before):,} pre-event and {len(after):,} post-event points')
        if normals is None:
            print('Computing normals of the post-event points ...')
            normals = utilities.compute_normals(after, config.normal_knn)

        result = icp.run_ticp(before, after, normals, config)

        name = (f'{args.prefix}{utilities.name_of(pre_event)}_to_{utilities.name_of(pos_event)}'
                f'_{fmt(config.window_size)}_{fmt(config.step_size)}')
        crs = config.crs or utilities.get_crs(pos_event)
        utilities.write_results(result, os.path.join(args.out_dir, name), config.step_size, crs)

        print(f'Elapsed time: {time.time() - start_time:.1f} seconds')
        if args.plot:
            utilities.plot_all(result)
