# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 00:14:04 2024

@author: nekhtari
"""

import os
import numpy as np
import pdal
import json
import time
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed






class icp_configs:
    def __init__(self, conf):
        self.bounds              = conf.get('bounds')
        self.classes             = conf.get('classes')
        self.window_size         = conf.get('window_size')
        self.step_size           = conf.get('step_size')
        self.threshold           = conf.get('threshold')
        self.margin              = conf.get('margin')
        self.min_points          = conf.get('min_points')
        self.Tconverge           = conf.get('Tconverge')
        self.Tmax_iter           = conf.get('Tmax_iter')
        self.outlier_threshold   = conf.get('outlier_threshold') #points larger than this many times the RMSE removed
        self.method              = conf.get('method')
        self.has_normal_post     = conf.get('has_normal_post')
        self.null                = conf.get('null')
        self.output_basename     = conf.get('output_basename')
        
        

        
def transicp(moving, fixed, fixed_normal, config, x, y):
    '''
    Parameters
    ----------
    moving : numpy array of size m x 3
        3D coordinates of moving point cloud points.
    fixed : numpy array of size n x 3
        3D coordinates of fixed point cloud points.
    fixed_normal : numpy array of size n x 3
        3D normal vector associated to local plane fitted to point in the fixed point cloud.
    Tconverge : scalar value
        ICP convergence threshold.
    Tmax_iter : integer value
        Maximum number of iterations of the ICP algorithm.
    outlier_threshold : scalar
        Threshold for outlier removal based on median absolute deviation (MAD).

    Returns
    -------
    icp_trans : list
        Estimated 3D shift vector.
    RMSE : scalar
        RMSE of the ICP algorithm.
    Max : scalar
        Maximum value of residual.

    '''
    null = config.null
    Tconverge = config.Tconverge
    Tmax_iter = config.Tmax_iter
    min_points = config.min_points
    outlier_threshold = config.outlier_threshold
    
    
    if (fixed is None) or (moving is None):
        return [null, null, null], null, null, x, y
    
    if ((len(fixed) < min_points) | (len(moving) < min_points)):
        return [null, null, null], null, null, x, y
    
    
    loop = True
    icp_trans = np.array([0, 0, 0])
    count = 0
    
    # Center the point clouds
    means = np.mean(fixed, axis=0)
    X1 = moving - means
    X2 = fixed - means
       
    kdtree = KDTree(X2)
    flag = False
    while loop:
        # Point indexing setup, updated each iteration
        search_pts = X1 + icp_trans
        D, I = kdtree.query(search_pts, k=1)
        
        Normal = fixed_normal[I, :]
        Vec_Diff = (search_pts - X2[I, :])
        misc = np.sum(Vec_Diff * Normal, axis=1)

        # Outlier removal based on residuals
        median_residual = np.median(misc)
        mad = np.median(np.abs(misc - median_residual))
        inlier_mask = np.abs(misc - median_residual) < outlier_threshold * mad

        # Only use inliers for ICP updates
        A = Normal[inlier_mask]
        misc_inliers = misc[inlier_mask]

        AT = A.transpose()
        ATA = AT.dot(A)
        
        # Attempt inversion and handle potential singular matrix
        try:
            N = np.linalg.inv(ATA)
            U = AT.dot(misc_inliers)
            delcap = -N.dot(U)
            
            icp_trans = icp_trans + delcap
            
            count += 1
            if np.all(np.abs(delcap) < Tconverge) or count > Tmax_iter:
                loop = False
                
        except np.linalg.LinAlgError:
            flag = True
            print("Singular matrix encountered during inversion. Exiting loop with NaN values.")
            icp_trans = np.array([np.nan, np.nan, np.nan])
            RMSE = np.nan
            Max = np.nan
            break  # Exit the loop
            

    if not flag:
        # Compute final RMSE for inliers
        Vec_Diff2 = (search_pts[inlier_mask] - X2[I[inlier_mask]])
        resi = np.sum(Vec_Diff2 * Normal[inlier_mask], axis=1)
        wsquared = np.sum(resi ** 2)
        RMSE = np.sqrt(wsquared / len(Vec_Diff2))
        Max = np.amax(np.abs(misc))

    return icp_trans, RMSE, Max, x, y





def run_transicp_parallel(blocks):

    conf = blocks[0][3]
    window_size = conf.window_size

    # Variables to hold ICP vector origins and displacements
    X, Y = [], []
    dx, dy, dz = [], [], []


    # Process each block with transicp in parallel
    results = [None] * len(blocks)
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(transicp, *block): idx for idx, block in enumerate(blocks)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()  # Place each result in the correct index

        # Store results from transicp
        for res in results:
            dx.append(res[0][0])
            dy.append(res[0][1])
            dz.append(res[0][2])
            X.append(res[3] + window_size / 2)
            Y.append(res[4] + window_size / 2)


    # return results
    res = np.stack([X, Y, dx, dy, dz], axis=1)
    return res



def get_ept_file(file, get_normal, calc_normal, cl):
    C = ['Classification[{}:{}]'.format(c, c) for c in cl]
    classes = ','.join(C)
    
    if file.endswith('json'):
        reader = 'readers.ept'
    elif file.endswith('laz'):
        reader = 'readers.las'


    if calc_normal:
        pipeline = [
            {
                'type':reader,
                'filename':file
            },
            {
                "type":"filters.range",
                "limits":classes
            },
            {
                "type": "filters.normal",  # Compute normals if missing
                "knn": 8                    # Number of neighbors for normal estimation
            }
        ] 
    
    else:
        pipeline = [
            {
                'type':reader,
                'filename':file
            },
            {
                "type":"filters.range",
                "limits":classes
            }
        ]


    P = pdal.Pipeline(json.dumps(pipeline))
    P.execute()
    
    try:
        p = P.arrays[0]
        XYZ = np.stack([p['X'], p['Y'], p['Z']], axis = 1)
        if get_normal:
            N   = np.stack([p['NormalX'], p['NormalY'], p['NormalZ']], axis = 1)
            return XYZ, N
        else:
            return XYZ
    except:
        return None, None
    
    
    
    

def run_all(before_file, after_file, config):
    """
    Extracts tiles for window-based ICP from two LAS files and saves them as .npy files.
    
    Parameters:
    - before_file: Path to the pre-event LAS file.
    - after_file: Path to the post-event LAS file.
    - window_size: Size of the tile in units.
    - step_size: Step size for moving the tile across the area.
    - output_dir: Directory to save the tiles. If None, a temporary folder is created.
    
    Returns:
    - List of file paths for the saved tiles.
    """
    
    RES = []
    if config.method == 'translation_only':
        # ICP window and step sizes
        window_size = config.window_size
        step_size = config.step_size
        margin = config.margin
        # bounds = config.bounds
        

    st = time.time()
    print("Reading point cloud files...")
    # Load points from the LAS files
    points_after, N_after = get_ept_file(after_file,  get_normal=True,  calc_normal=True, cl=classes)
    points_before =         get_ept_file(before_file, get_normal=False, calc_normal=False, cl=classes)
    et = time.time()
    print(f"Time to read both las files and compute Normals for Post event points was {np.round(et - st, 3)} seconds")


    # Determine the bounding box of the area
    all_points = np.vstack((points_before, points_after))
    x_min, y_min = np.round(np.min(all_points[:, :2], axis=0))
    x_max, y_max = np.round(np.max(all_points[:, :2], axis=0))

    # Create a grid of tile top-left corners
    x_coords = np.arange(x_min, x_max - window_size, step_size)
    y_coords = np.arange(y_min, y_max - window_size, step_size)
    limit = 300
    batch_num = 0
    total = len(x_coords) * len(y_coords)
    total_batches = int(np.ceil(total / limit))
    
    # Dictionary to store tiles
    tiles = []
    st = time.time()
    # Loop over each tile and extract points
    for x in x_coords:
        # Define the bounds of the tile
        x_max_tile = x + window_size
        mask_before1 = (points_before[:, 0] >= x) & (points_before[:, 0] < x_max_tile)
        mask_after1 = (points_after[:, 0] >= (x - margin)) & (points_after[:, 0] < (x_max_tile + margin))
        
        pb = points_before[mask_before1]
        pa = points_after[mask_after1]
        pn = N_after[mask_after1]
        if pb.shape[0] == 0 or pa.shape[0] == 0:
            continue

        for y in y_coords:

                
            # Define the bounds of the tile
            y_max_tile = y + window_size

            # Find points within the tile for "before" points
            mask_before = (pb[:, 1] >= y) & (pb[:, 1] < y_max_tile)
            tile_before = pb[mask_before]

            # Find points within the tile for "after" points.
            # Add the margin here so the after points cover larger area
            mask_after = (pa[:, 1] >= (y - margin)) & (pa[:, 1] < (y_max_tile + margin))
            tile_after = pa[mask_after]
            N_tile_after = pn[mask_after]

            tiles.append([tile_before, tile_after, N_tile_after, config, x - (step_size / 2), y + (step_size / 2)])
            
            if len(tiles) == limit:
                batch_num += 1
                print(f"processing batch {batch_num} / {total_batches}")
                res = run_transicp_parallel(tiles)
                for r in res:
                    RES.append(r)
                tiles = []
            
   
    # Final batch processing for any leftover tiles
    if tiles:
        batch_num += 1
        print(f"processing final batch #{batch_num} with {len(tiles)} tiles")
        res = run_transicp_parallel(tiles)
        for r in res:
            RES.append(r)

    et = time.time()
    print(f"Time to process all tiles was {(et - st) // 60} minutes and {np.round(np.remainder((et - st), 60), 2)} seconds")
    
    
    return RES



''' ---------------------------------------------------------------------- '''


if __name__ == "__main__":
    pre_event = r'D:\Working\SCE\Landslide\Data\LAZ_Classified\Klondike_LAS_20240906_ModelKey_Bldgs_V2.laz'
    pos_event = r'D:\Working\SCE\Landslide\Data\LAZ_Classified\Klondike_LAS_20240925_ModelKey_Bldgs_V2.laz'
    # pre_event = r'D:\Working\SCE\Landslide\Data\RollingHills\Data\test_0925.laz'
    # pos_event = r'D:\Working\SCE\Landslide\Data\RollingHills\Data\test_1018.laz'
    
    
    operation = 'translation_only'
    classes = [6, 8]
    # import utilities
    # bounds_pre, has_normals_pre = utilities.get_metadata(pre_event)
    # bounds_pos, has_normals_pos = utilities.get_metadata(pos_event)
    
    # bounds = []
    # bounds.append(int(min(bounds_pre[0], bounds_pos[0])))
    # bounds.append(int(max(bounds_pre[1], bounds_pos[1])))
    # bounds.append(int(min(bounds_pre[2], bounds_pos[2])))
    # bounds.append(int(max(bounds_pre[3], bounds_pos[3])))
    
    
    
    if operation == 'translation_only':
        configs = {
        'classes': classes,
        'bounds' : None,
        'method' : 'translation_only',
        'threshold' : 20,
        'window_size' : 150,
        'step_size' : 25,
        'margin': 3,
        'min_points' : 20,
        'Tconverge' : 0.0005,
        'Tmax_iter' : 20,
        'outlier_threshold': 3,
        'has_normal_post' : False, #has_normals_pos,
        'null': -99,
        'output_basename' : 'test_sep6_sep25_' #'trans_icp_results_0906_0925¥'
        }
    
    config = icp_configs(configs)
    
    
    a = run_all(pre_event, pos_event, config)
    sname = f'{config.output_basename}_{config.window_size}_{config.step_size}.txt'
    np.savetxt(sname, np.array(a), delimiter=' ', fmt='%.3f')







    import rasterio
    from rasterio.transform import from_origin
    
    A = np.array(a)
    x_unique = np.unique(A[:, 0])
    y_unique = np.unique(A[:, 1])

    # Create grids for the rasters
    nrows, ncols = len(y_unique), len(x_unique)
    dx = np.full((nrows, ncols), np.nan)
    dy = np.full((nrows, ncols), np.nan)
    dz = np.full((nrows, ncols), np.nan)

    # Populate the grids
    for x, y, val1, val2, val3 in zip(A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]):
        row = np.where(y_unique == y)[0][0]
        col = np.where(x_unique == x)[0][0]
        dx[row, col] = val1
        dy[row, col] = val2
        dz[row, col] = val3
        
    def write_disp_rasters(dx, dy, dz, output_file, transform, crs="EPSG:6424"):

        # Set up metadata for the GeoTIFF
        height, width = dx.shape
        metadata = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 3,  # Three bands for dx, dy, dz
            'width': width,
            'height': height,
            'crs': crs,  # Set CRS to EPSG:6424
            'transform': transform  # Affine transform
        }

        # Write the three bands to a GeoTIFF
        with rasterio.open(output_file, 'w', **metadata) as dst:
            dst.write(dx, 1)  # Write dx as the first band
            dst.write(dy, 2)  # Write dy as the second band
            dst.write(dz, 3)  # Write dz as the third band

        print(f"Raster written to {output_file} with CRS {crs}")

    # Example usage
    # Assume the disp matrix and the upper-left corner coordinates and pixel size are known

    transform = from_origin(min(A[:, 0]), max(A[:, 1]), configs['step_size'], configs['step_size'])  # Replace with your actual top-left coordinates and pixel size

    filename = f'{configs['output_basename']}_{configs['window_size']}_{configs['step_size']}.tif'
    write_disp_rasters(np.flipud(dx), np.flipud(dy), np.flipud(dz), filename, transform)















