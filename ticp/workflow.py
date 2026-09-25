# -*- coding: utf-8 -*-
"""
@author: nekhtari

compare() runs the whole TICP workflow on two point cloud files: read them,
estimate the normals, run the moving-window ICP and (optionally) write the
results. The command line tool, main.py and the GUI all go through here.
"""

import os
import time
import logging
from dataclasses import replace

from .icp import ICPConfig, run_ticp, TICPCancelled
from . import utilities

log = logging.getLogger(__name__)



def compare(before_file, after_file, config=None, output=None, progress=None, cancel=None, **kwargs):
    """
    Measures the 3D change between two point clouds with moving-window TICP.

    Parameters
    ----------
    before_file, after_file : str
        Earlier and later point cloud (LAS/LAZ, or ept.json with PDAL installed).
    config : ICPConfig, optional
        Processing parameters. Any ICPConfig field can also be given directly as
        a keyword, e.g. compare(a, b, window_size=100, step_size=20).
    output : str, optional
        Path without extension. If given, <output>.txt and <output>.tif are written.
    progress : function progress(fraction, message), optional
        Called while the run goes on. fraction is between 0 and 1, or None while
        a step of unknown length is running (reading, normals).
    cancel : threading.Event, optional
        Set it from another thread to stop the run. TICPCancelled is raised.

    Returns
    -------
    result : dict with 'x', 'y' (window centers), the 2D grids 'dx', 'dy', 'dz',
        'rmse', 'n_points' (row 0 = south) and 'crs' (WKT or None).
    """
    config = replace(config or ICPConfig(), **kwargs)

    def report(fraction, message):
        if progress is not None:
            progress(fraction, message)
        if cancel is not None and cancel.is_set():
            raise TICPCancelled()

    start_time = time.time()
    report(None, 'Reading point clouds')
    before, _ = utilities.read_point_cloud(before_file, config.classes)
    after, normals = utilities.read_point_cloud(after_file, config.classes)
    log.info(f'{len(before):,} pre-event and {len(after):,} post-event points')

    if normals is None:
        report(None, 'Computing normals of the post-event points')
        normals = utilities.compute_normals(after, config.normal_knn)

    report(0.0, 'ICP')
    result = run_ticp(before, after, normals, config, progress=progress, cancel=cancel)
    result['crs'] = config.crs or utilities.get_crs(after_file)

    if output is not None:
        report(1.0, 'Writing results')
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        utilities.write_results(result, output, config.step_size, result['crs'])

    log.info(f'Elapsed time: {time.time() - start_time:.1f} seconds')
    if progress is not None:
        progress(1.0, 'Done')
    return result



def output_name(before_file, after_file, config, prefix=''):
    """ <prefix><before>_to_<after>_<window>_<step>, e.g. ticp_0925_to_1018_150_25 """
    return (f'{prefix}{utilities.name_of(before_file)}_to_{utilities.name_of(after_file)}'
            f'_{config.window_size:g}_{config.step_size:g}')
