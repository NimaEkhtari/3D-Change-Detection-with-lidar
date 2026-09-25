# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:54:35 2024

@author: nekhtari

Translation-only ICP (TICP) applied in a moving window over two lidar point
clouds. For every window we solve for a single 3D shift (dx, dy, dz) that best
aligns the pre-event points to the post-event surface using point-to-plane ICP.
No rotation is estimated because both clouds are already georeferenced.

The processing is done on the CPU. Columns of windows are distributed over
several processes and the point clouds are placed in shared memory, so each
process can cut its own tiles without copying the full clouds around.
"""

import os
import time
import numpy as np
from dataclasses import dataclass
from scipy.spatial import KDTree
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed



@dataclass
class ICPConfig:
    window_size: float = 150          # size of the square ICP window (point cloud units)
    step_size: float = 25             # distance between window centers = output pixel size
    margin: float = 3                 # buffer added around the post-event (fixed) window
    classes: list = None              # LAS classes to use, e.g. [2] for ground. None = all points
    min_points: int = 20              # windows with fewer points than this are skipped
    convergence: float = 0.0005       # stop when every component of the update is below this
    max_iter: int = 20                # maximum number of ICP iterations per window
    outlier_threshold: float = 3      # residuals larger than this many MADs are rejected
    normal_knn: int = 8               # neighbours used to estimate the post-event normals
    bounds: list = None               # [xmin, xmax, ymin, ymax]. None = overlap of the two clouds
    n_workers: int = None             # number of processes. None = all cores but one, 1 = no multiprocessing
    crs: str = None                   # e.g. 'EPSG:6424'. None = read from the point cloud



def transicp(moving, fixed, fixed_normal, config):
    '''
    Point-to-plane ICP that only solves for a translation.

    Parameters
    ----------
    moving : numpy array of size m x 3
        3D coordinates of the moving (pre-event) points.
    fixed : numpy array of size n x 3
        3D coordinates of the fixed (post-event) points.
    fixed_normal : numpy array of size n x 3
        Unit normal of the local plane at every fixed point.
    config : ICPConfig
        Uses min_points, convergence, max_iter and outlier_threshold.

    Returns
    -------
    icp_trans : numpy array of size 3
        Estimated shift [dx, dy, dz] that moves the moving points onto the fixed
        surface. NaN if the window could not be solved.
    RMSE : scalar
        Point-to-plane RMSE of the inliers after the last iteration.
    n_inliers : integer
        Number of points used in the last iteration.
    '''
    failed = np.array([np.nan, np.nan, np.nan]), np.nan, 0

    if len(fixed) < config.min_points or len(moving) < config.min_points:
        return failed

    # Center the point clouds (large projected coordinates hurt the precision)
    means = np.mean(fixed, axis=0)
    X1 = moving - means
    X2 = fixed - means

    kdtree = KDTree(X2)
    icp_trans = np.zeros(3)

    for count in range(config.max_iter):
        search_pts = X1 + icp_trans
        _, I = kdtree.query(search_pts, k=1)

        Normal = fixed_normal[I, :]
        misc = np.sum((search_pts - X2[I, :]) * Normal, axis=1)

        # Outlier removal based on median absolute deviation of the residuals
        median_residual = np.median(misc)
        mad = np.median(np.abs(misc - median_residual))
        inlier_mask = np.abs(misc - median_residual) < config.outlier_threshold * mad
        if np.count_nonzero(inlier_mask) < config.min_points:
            return failed

        # Least squares update using the inliers only
        A = Normal[inlier_mask]
        try:
            delcap = -np.linalg.solve(A.T @ A, A.T @ misc[inlier_mask])
        except np.linalg.LinAlgError:
            return failed

        icp_trans = icp_trans + delcap
        if np.all(np.abs(delcap) < config.convergence):
            break

    # Final RMSE for the inliers, after applying the last update
    resi = misc[inlier_mask] + A @ delcap
    RMSE = np.sqrt(np.mean(resi ** 2))

    return icp_trans, RMSE, len(resi)



''' ---------------------------------------------------------------------- '''
'''   Moving window                                                         '''
''' ---------------------------------------------------------------------- '''

# Every worker process keeps the point clouds and the grid here. In the main
# process (n_workers = 1) the same dictionary is filled directly.
_data = {}


def _set_data(before, after, normals, xs, ys, config):
    _data.update(before=before, after=after, normals=normals, xs=xs, ys=ys, config=config)



def _init_worker(specs, xs, ys, config):
    arrays = {}
    for name, (shm_name, shape, dtype) in specs.items():
        try:
            shm = shared_memory.SharedMemory(name=shm_name, track=False)   # python >= 3.13
        except TypeError:
            shm = shared_memory.SharedMemory(name=shm_name)
        _data['shm_' + name] = shm        # keep a reference, otherwise the buffer gets closed
        arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    _set_data(arrays['before'], arrays['after'], arrays['normals'], xs, ys, config)



def process_column(col):
    """
    Runs TICP for every window in one column of the grid. The clouds are sorted
    by X, so the column is cut with a binary search and each window inside the
    column only needs a mask on Y.
    """
    before, after, normals = _data['before'], _data['after'], _data['normals']
    ys, config = _data['ys'], _data['config']
    w, m = config.window_size, config.margin
    x = _data['xs'][col]

    # Pre-event (moving) points: the window itself
    lo, hi = np.searchsorted(before[:, 0], [x, x + w])
    pb = before[lo:hi]
    # Post-event (fixed) points: the window plus a margin, so the moving points
    # still find their neighbours after they are shifted
    lo, hi = np.searchsorted(after[:, 0], [x - m, x + w + m])
    pa = after[lo:hi]
    pn = normals[lo:hi]

    out = []
    for row, y in enumerate(ys):
        mask_b = (pb[:, 1] >= y) & (pb[:, 1] < y + w)
        mask_a = (pa[:, 1] >= y - m) & (pa[:, 1] < y + w + m)
        shift, rmse, n = transicp(pb[mask_b], pa[mask_a], pn[mask_a], config)
        out.append((row, col, shift[0], shift[1], shift[2], rmse, n))
    return out



def _to_shared(arr):
    shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    return shm



def make_grid(before, after, config):
    """ Lower-left corners of all windows (xs, ys). """
    if config.bounds is not None:
        x_min, x_max, y_min, y_max = config.bounds
    else:
        # Overlap of the two clouds, windows outside it cannot be solved anyway
        x_min = np.floor(max(before[:, 0].min(), after[:, 0].min()))
        x_max = np.ceil(min(before[:, 0].max(), after[:, 0].max()))
        y_min = np.floor(max(before[:, 1].min(), after[:, 1].min()))
        y_max = np.ceil(min(before[:, 1].max(), after[:, 1].max()))

    xs = np.arange(x_min, x_max - config.window_size + 1e-6, config.step_size)
    ys = np.arange(y_min, y_max - config.window_size + 1e-6, config.step_size)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError('The overlapping area is smaller than one window. '
                         'Check the input files or use a smaller window_size.')
    return xs, ys



def run_ticp(before, after, normals, config):
    """
    Moving-window TICP over two point clouds that are already in memory.

    Parameters
    ----------
    before : numpy array n x 3, pre-event points (moving)
    after : numpy array m x 3, post-event points (fixed)
    normals : numpy array m x 3, normals of the post-event points
    config : ICPConfig

    Returns
    -------
    result : dict with
        'x', 'y'  : 1D arrays with the window centers along X and Y
        'dx', 'dy', 'dz', 'rmse', 'n_points' : 2D grids of size len(y) x len(x),
                    row 0 is the southern-most row
    """
    xs, ys = make_grid(before, after, config)
    n_workers = config.n_workers or max(1, (os.cpu_count() or 2) - 1)
    print(f'{len(xs)} x {len(ys)} = {len(xs) * len(ys)} windows, using {n_workers} process(es)')

    # Sort by X so each column can be found with a binary search
    order = np.argsort(before[:, 0], kind='stable')
    before = np.ascontiguousarray(before[order])
    order = np.argsort(after[:, 0], kind='stable')
    after = np.ascontiguousarray(after[order])
    normals = np.ascontiguousarray(normals[order])

    grids = {k: np.full((len(ys), len(xs)), np.nan) for k in ['dx', 'dy', 'dz', 'rmse', 'n_points']}

    def store(rows):
        for row, col, dx, dy, dz, rmse, n in rows:
            grids['dx'][row, col] = dx
            grids['dy'][row, col] = dy
            grids['dz'][row, col] = dz
            grids['rmse'][row, col] = rmse
            grids['n_points'][row, col] = n

    st = time.time()
    last = 0
    if n_workers == 1:
        _set_data(before, after, normals, xs, ys, config)
        for col in range(len(xs)):
            store(process_column(col))
            last = _progress(col + 1, len(xs), last)
    else:
        shms = {name: _to_shared(arr) for name, arr in
                [('before', before), ('after', after), ('normals', normals)]}
        specs = {name: (shms[name].name, arr.shape, arr.dtype) for name, arr in
                 [('before', before), ('after', after), ('normals', normals)]}
        try:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                     initargs=(specs, xs, ys, config)) as executor:
                futures = [executor.submit(process_column, col) for col in range(len(xs))]
                for done, future in enumerate(as_completed(futures), start=1):
                    store(future.result())
                    last = _progress(done, len(xs), last)
        finally:
            for shm in shms.values():
                shm.close()
                shm.unlink()

    et = time.time() - st
    print(f'ICP of all windows took {int(et // 60)} min {et % 60:.1f} s')

    result = {'x': xs + config.window_size / 2, 'y': ys + config.window_size / 2}
    result.update(grids)
    return result



def _progress(done, total, last):
    """ Prints the progress roughly every 5%. """
    pct = int(100 * done / total)
    if pct >= last + 5 or done == total:
        print(f'{pct}% done')
        return pct
    return last
