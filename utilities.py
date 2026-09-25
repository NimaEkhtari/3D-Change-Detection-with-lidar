# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:02:00 2024

@author: nekhtari

Reading point clouds, estimating normals and writing the TICP results.

LAS/LAZ files are read with laspy. Anything else (for example an Entwine
ept.json) is handed to PDAL, which is optional and only needed for those formats.
"""

import os
import json
import numpy as np
from scipy.spatial import KDTree



def read_point_cloud(file, classes=None):
    """
    Returns the XYZ coordinates (n x 3) of a point cloud, optionally keeping only
    some classes. Also returns the stored normals (n x 3) if the file has
    NormalX/NormalY/NormalZ dimensions, otherwise None.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f'Point cloud not found: {file}')

    if file.lower().endswith(('.las', '.laz')):
        import laspy
        las = laspy.read(file)
        keep = np.ones(len(las.points), dtype=bool)
        if classes is not None:
            keep = np.isin(np.asarray(las.classification), classes)
        XYZ = np.column_stack([las.x, las.y, las.z])[keep]
        dims = set(las.point_format.dimension_names)
        N = None
        if {'NormalX', 'NormalY', 'NormalZ'} <= dims:
            N = np.column_stack([las['NormalX'], las['NormalY'], las['NormalZ']])[keep]

    else:
        try:
            import pdal
        except ImportError:
            raise ImportError(f'Reading {os.path.basename(file)} needs PDAL. Either convert it to '
                              'LAS/LAZ or install PDAL (conda install -c conda-forge python-pdal)')
        pipeline = [{'filename': file}]
        if classes is not None:
            limits = ','.join(f'Classification[{c}:{c}]' for c in classes)
            pipeline.append({'type': 'filters.range', 'limits': limits})
        P = pdal.Pipeline(json.dumps(pipeline))
        P.execute()
        p = P.arrays[0]
        XYZ = np.stack([p['X'], p['Y'], p['Z']], axis=1)
        N = None
        if 'NormalX' in p.dtype.names:
            N = np.stack([p['NormalX'], p['NormalY'], p['NormalZ']], axis=1)

    if len(XYZ) == 0:
        raise ValueError(f'No points left in {file} (classes = {classes})')
    return XYZ.astype(np.float64), N



def compute_normals(XYZ, knn=8, chunk=1_000_000):
    """
    Normal of the plane fitted to the knn nearest neighbours of every point
    (same idea as PDAL filters.normal). Normals are flipped to point upward.
    Done in chunks to keep the memory in check for large clouds.
    """
    tree = KDTree(XYZ)
    N = np.empty_like(XYZ)
    for i in range(0, len(XYZ), chunk):
        _, idx = tree.query(XYZ[i:i + chunk], k=knn, workers=-1)
        nbrs = XYZ[idx]
        nbrs = nbrs - nbrs.mean(axis=1, keepdims=True)
        cov = np.einsum('mki,mkj->mij', nbrs, nbrs)
        _, vecs = np.linalg.eigh(cov)           # eigenvalues come sorted ascending
        N[i:i + chunk] = vecs[:, :, 0]          # smallest eigenvalue -> normal
    N[N[:, 2] < 0] *= -1
    return N



def get_crs(file):
    """ Horizontal CRS of a point cloud as WKT, or None if it cannot be found. """
    try:
        if file.lower().endswith(('.las', '.laz')):
            import laspy
            with laspy.open(file) as f:
                crs = f.header.parse_crs()
            if crs is None:
                return None
            if crs.is_compound:
                crs = crs.sub_crs_list[0]
            return crs.to_wkt()
        if file.lower().endswith('ept.json'):
            with open(file) as f:
                return json.load(f)['srs']['wkt'] or None
    except Exception as e:
        print(f'Could not read the CRS from {file}: {e}')
    return None



def name_of(file):
    """ Short name of an input, for ept.json the name of its folder. """
    if os.path.basename(file).lower() == 'ept.json':
        return os.path.basename(os.path.dirname(os.path.abspath(file)))
    return os.path.splitext(os.path.basename(file))[0]



def write_results(result, basename, step_size, crs=None):
    """
    Writes the TICP results as
      <basename>.txt : one line per window: X Y dx dy dz rmse n_points
      <basename>.tif : GeoTIFF with bands dx, dy, dz, rmse (pixel size = step size)
    """
    import rasterio
    from rasterio.transform import from_origin

    X, Y = np.meshgrid(result['x'], result['y'])
    table = np.column_stack([X.ravel(), Y.ravel()] +
                            [result[k].ravel() for k in ['dx', 'dy', 'dz', 'rmse', 'n_points']])
    np.savetxt(basename + '.txt', table, fmt='%.3f', delimiter=' ',
               header='X Y dx dy dz rmse n_points')

    # Pixel centers are the window centers, so the raster starts half a step
    # to the left of the first center and half a step above the last one
    transform = from_origin(result['x'][0] - step_size / 2, result['y'][-1] + step_size / 2,
                            step_size, step_size)
    bands = ['dx', 'dy', 'dz', 'rmse']
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': len(bands),
        'width': len(result['x']),
        'height': len(result['y']),
        'crs': crs,
        'transform': transform,
        'nodata': np.nan,
        'compress': 'lzw',
    }
    with rasterio.open(basename + '.tif', 'w', **profile) as dst:
        for i, b in enumerate(bands, start=1):
            dst.write(np.flipud(result[b]).astype('float32'), i)   # row 0 of a raster is north
            dst.set_band_description(i, b)

    if crs is None:
        print('Warning: no CRS found, the GeoTIFF has no spatial reference. Set crs in main.py')
    print(f'Results written to {basename}.txt and {basename}.tif')



def plot_all(result, Th=None):
    """
    Quick look at the results: dx, dy, dz grids and the horizontal vectors.
    Windows with a horizontal shift larger than Th are left out.
    """
    import matplotlib.pyplot as plt

    dx, dy, dz = result['dx'].copy(), result['dy'].copy(), result['dz'].copy()
    if Th is not None:
        bad = np.hypot(dx, dy) > Th
        dx[bad] = dy[bad] = dz[bad] = np.nan

    extent = [result['x'][0], result['x'][-1], result['y'][0], result['y'][-1]]
    fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, grid, title in zip(axs.flat, [dx, dy, dz], ['dx', 'dy', 'dz']):
        lim = np.nanpercentile(np.abs(grid), 98) if np.any(np.isfinite(grid)) else 1
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-lim, vmax=lim, origin='lower', extent=extent)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    X, Y = np.meshgrid(result['x'], result['y'])
    axs[1, 1].quiver(X, Y, dx, dy, angles='xy', headwidth=2.5, headlength=4)
    axs[1, 1].set_title('Horizontal displacement')
    axs[1, 1].set_aspect('equal')
    plt.tight_layout()
    plt.show()
