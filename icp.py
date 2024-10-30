# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:54:35 2024

@author: nekhtari
"""

import os
import numpy as np
# import faiss
import pdal
import json
import argparse
import math
import utilities
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed



''' ********************************************************************** '''
''' ********************************************************************** '''
''' ********************************************************************** '''
''' ********************************************************************** '''

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
        RMSE = math.sqrt(wsquared / len(Vec_Diff2))
        Max = np.amax(np.abs(misc))

    return icp_trans, RMSE, Max, x, y




def extract_tile(x, y, tile_size, margin, B, A, N):
    """
    Extracts a tile from a matrix M of 3D coordinates with a margin around it.
    
    Parameters:
    - x, y: Bottom-right corner of the tile (smallest x and y coordinates of the tile).
    - tile_size: Size of the tile in units.
    - margin: Additional margin to add around the tile.
    - M: Numpy array of shape (1000, 3) representing 3D coordinates (x, y, z).
    
    Returns:
    - A numpy array of points within the specified tile and margin.
    """
    
    # Define boundaries for filtering
    x_min = x - margin
    x_max = x + tile_size + margin
    y_min = y - margin
    y_max = y + tile_size + margin
    
    # Apply logical indexing to select points within the boundary
    b = B[(B[:, 0] >= x_min) & (B[:, 0] <= x_max) & (B[:, 1] >= y_min) & (B[:, 1] <= y_max)]
    mask = (A[:, 0] >= x_min) & (A[:, 0] <= x_max) & (A[:, 1] >= y_min) & (A[:, 1] <= y_max)
    a = A[mask]
    n = N[mask]
    
    return b, a, n




def run_transicp_parallel(pre_event, pos_event, config):
    cpu_cores = os.cpu_count() * 20 # Use actual core count

    if config.method == 'translation_only':
        # ICP window and step sizes
        window_size = config.window_size
        step_size = config.step_size
        margin = config.margin
        bounds = config.bounds

        calc_normal = not config.has_normal_post
        classes = config.classes

        # Variables to hold ICP vector origins and displacements
        X, Y = [], []
        dx, dy, dz = [], [], []
        DX, DY, DZ = [], [], []
        dxr, dyr, dzr = [], [], []

    # Load point cloud data
    A, N = get_ept_file(pos_event, get_normal=True, calc_normal=calc_normal, cl=classes)
    B = get_ept_file(pre_event, get_normal=False, calc_normal=calc_normal, cl=classes)
    
    # Generate coordinate grid
    x_min, x_max, y_min, y_max = bounds
    xx = np.arange(x_min, x_max, step_size)
    yy = np.arange(y_min, y_max, step_size)
    llx, lly = np.meshgrid(xx, yy)
    x = llx.ravel()
    y = lly.ravel()
    
    total_steps = len(x)
    n_batches = int(np.ceil(total_steps / cpu_cores))

    # Loop over processes and prepare tiles in parallel
    for i in range(n_batches):
        block_indices = range(i * cpu_cores, min((i + 1) * cpu_cores, total_steps))
        blocks = []

        # Parallel extraction of tiles
        with ThreadPoolExecutor() as executor:
            future_blocks = {executor.submit(extract_tile, x[j], y[j], step_size, margin, B, A, N): j for j in block_indices}
            for future in as_completed(future_blocks):
                j = future_blocks[future]
                Xb, Xa, Na = future.result()
                blocks.append([Xb, Xa, Na, config, x[j], y[j]])

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
                dxr.append(res[0][0])
                dyr.append(res[0][1])
                dzr.append(res[0][2])
                X.append(res[3] + window_size / 2)
                Y.append(res[4] + window_size / 2)

                if len(X) % len(xx) == 0:  # Per row completion
                    DX.append(dxr)
                    DY.append(dyr)
                    DZ.append(dzr)
                    dxr, dyr, dzr = [], [], []

        # Progress update
        progress = ((i + 1) / n_batches) * 100
        print(f'{progress:.2f}% done')

    # Save results after processing all tiles
    res = np.stack([X, Y, dx, dy, dz], axis=1)
    sname = f'{config.output_basename}_{window_size}_{step_size}.txt'
    np.savetxt(sname, res, delimiter=' ', fmt='%.3f')
    return res, np.stack([DX, DY, DZ], axis=2)






def run_transicp_parallel_old2(pre_event, pos_event, config):
    cpu_cores = os.cpu_count() * 10
    
    if config.method == 'translation_only':
            
        # ICP window and step sizes
        window_size = config.window_size
        step_size = config.step_size
        margin = config.margin
        bounds = config.bounds
        

        calc_normal = not config.has_normal_post
        classes = config.classes
        
        
        # Variables to hold the ICP vector origins (X, Y) and displacements (dx, dy)
        X, Y = [], []
        dx, dy, dz = [], [], []
        DX, DY, DZ = [], [], []
        dxr, dyr, dzr = [], [], []
        RMSE, fitn = [], []
    

    # Load point cloud data
    A, N = get_ept_file(pos_event, get_normal=True, calc_normal=calc_normal, cl=classes)
    B = get_ept_file(pre_event, get_normal=False, calc_normal=calc_normal, cl=classes)
    
    
    # Generate coordinate grid
    x_min, x_max, y_min, y_max = bounds
    xx = np.arange(x_min, x_max, step_size)
    yy = np.arange(y_min, y_max, step_size)
    llx, lly = np.meshgrid(xx, yy)
    x = llx.ravel()
    y = lly.ravel()
    
    total_x_steps = len(xx)
    total_y_steps = len(yy)
    n_process = np.ceil((total_x_steps * total_y_steps) / cpu_cores)
    blocks = []
    per_row = 0
    
    
    for i in range(int(n_process)):

        for j in range(cpu_cores*i, cpu_cores*(i+1)):
            Xb, Xa, Na = extract_tile(x[j], y[j], step_size, margin, B, A, N)
            blocks.append([Xb, Xa, Na, config, x[j], y[j]])
            
        if len(blocks) == cpu_cores:
            results = [None] * len(blocks)

            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(transicp, *block): idx for idx, block in enumerate(blocks)}
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()  # Place each result in the correct index
                
                for res in results:
                    dx.append(res[0][0])
                    dy.append(res[0][1])
                    dz.append(res[0][2])
                    dxr.append(res[0][0])
                    dyr.append(res[0][1])
                    dzr.append(res[0][2])
                    X.append(res[3] + window_size/2)
                    Y.append(res[4] + window_size/2)
                    per_row += 1
                    if per_row == total_x_steps:
                        DX.append(dxr)
                        DY.append(dyr)
                        DZ.append(dzr)
                        dxr, dyr, dzr = [], [], []
                        per_row = 0
                    
            blocks = []
            results.clear()


            progress = (i / n_process) * 100
            print('{:.2f}% done'.format(progress))
    
    
    
    res = np.stack([X, Y, dx, dy, dz], axis = 1)
    sname = '{0}_{1}_{2}.txt'.format(config.output_basename, window_size, step_size)
    np.savetxt(sname, res, delimiter=' ', fmt='%.3f')
    return res, np.stack([DX, DY, DZ], axis = 2)





def run_transicp_parallel_old(pre_event, pos_event, config):
    cpu_cores = os.cpu_count()
    
    if config.method == 'translation_only':
            
        # ICP window and step sizes
        window_size = config.window_size
        step_size = config.step_size
        margin = config.margin
        bounds = config.bounds
        

        calc_normal = not config.has_normal_post
        classes = config.classes
        
        
        # Variables to hold the ICP vector origins (X, Y) and displacements (dx, dy)
        X, Y = [], []
        dx, dy, dz = [], [], []
        DX, DY, DZ = [], [], []
        dxr, dyr, dzr = [], [], []
        RMSE, fitn = [], []
    
    total_x_steps = ((bounds[1] - bounds[0]) // step_size) + 1
    total_y_steps = ((bounds[3] - bounds[2]) // step_size) + 1
    n_process = np.ceil((total_x_steps * total_y_steps) / cpu_cores)
    i = 0
    blocks = []
    per_row = 0
    
    for y in range(bounds[2], bounds[3], step_size):
        for x in range(bounds[0], bounds[1], step_size):
            
            Xa, Na = get_pos_event(pos_event, x, y, margin, window_size, calc_normal, classes)
            Xb = get_pre_event(pre_event, x, y, window_size, classes)

            
            if len(blocks) < cpu_cores:
                blocks.append([Xb, Xa, Na, config, x, y])
            if len(blocks) == cpu_cores:
                results = [None] * len(blocks)
                i += 1
                with ThreadPoolExecutor() as executor:
                    # futures = [executor.submit(transicp, *block) for block in blocks]
                    futures = {executor.submit(transicp, *block): idx for idx, block in enumerate(blocks)}
                    for future in as_completed(futures):
                        # results.append(future.result())
                        idx = futures[future]
                        results[idx] = future.result()  # Place each result in the correct index
                    
                    for res in results:
                        dx.append(res[0][0])
                        dy.append(res[0][1])
                        dz.append(res[0][2])
                        dxr.append(res[0][0])
                        dyr.append(res[0][1])
                        dzr.append(res[0][2])
                        X.append(res[3] + window_size/2)
                        Y.append(res[4] + window_size/2)
                        per_row += 1
                        if per_row == total_x_steps:
                            DX.append(dxr)
                            DY.append(dyr)
                            DZ.append(dzr)
                            dxr, dyr, dzr = [], [], []
                            per_row = 0
                        
                blocks = []


                progress = (i / n_process) * 100
                print('{:.2f}% done'.format(progress))
    
    
    
    res = np.stack([X, Y, dx, dy, dz], axis = 1)
    sname = '{0}_{1}_{2}.txt'.format(config.output_basename, window_size, step_size)
    np.savetxt(sname, res, delimiter=' ', fmt='%.3f')
    return res, np.stack([DX, DY, DZ], axis = 2)





def run_transicp(pre_event, pos_event, config):
    
    if config.method == 'translation_only':
            
        # ICP window and step sizes
        window_size = config.window_size
        step_size = config.step_size
        margin = config.margin
        bounds = config.bounds
        calc_normal = not config.has_normal_post
        classes = config.classes
        
        
        # Variables to hold the ICP vector origins (X, Y) and displacements (dx, dy)
        X, Y = [], []
        dx, dy, dz = [], [], []
        DX, DY, DZ = [], [], []
        RMSE, fitn = [], []
    
    
    total_y_steps = ((bounds[3] - bounds[2]) // step_size) + 1
    
    for i, y in enumerate(range(bounds[2], bounds[3], step_size)):
        dxr, dyr, dzr = [], [], []
        for x in range(bounds[0], bounds[1], step_size):
            
            Xa, Na = get_pos_event(pos_event, x, y, margin, window_size, calc_normal, classes)
            Xb = get_pre_event(pre_event, x, y, window_size, classes)

                       
            
            res1, rmse, Max = transicp(Xb, Xa, Na, config)
            dx.append(res1[0])
            dy.append(res1[1])
            dz.append(res1[2])
            dxr.append(res1[0])
            dyr.append(res1[1])
            dzr.append(res1[2])
            
            # RMSE.append(rmse)
            # fitn.append(registration_icp.fitness)
    
            X.append(x + window_size/2)
            Y.append(y + window_size/2)
                
                
        DX.append(dxr)
        DY.append(dyr)
        DZ.append(dzr)
        progress = ((i + 1) / total_y_steps) * 100
        print('{:.2f}% done'.format(progress))
    
    
    
    res = np.stack([X, Y, dx, dy, dz], axis = 1)
    sname = '{0}_{1}_{2}.txt'.format(config.output_basename, window_size, step_size)
    np.savetxt(sname, res, delimiter=' ', fmt='%.3f')
    return res, np.stack([DX, DY, DZ], axis = 2)
    
    
 
    

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
    
    
    
    
    
def get_pos_event(file, x, y, margin, window_size, calc_normal, cl):
    if file.endswith('json'):
        reader = 'readers.ept'
    elif file.endswith('laz'):
        reader = 'readers.las'
        
        
    C = ['Classification[{}:{}]'.format(c, c) for c in cl]
    classes = ','.join(C)
    if calc_normal:
        pipeline = [
            {
                'type':reader,
                'filename':file,
                'bounds':'([{},{}],[{},{}])'.format(x - margin, x + window_size + margin, y - margin, y + window_size + margin)
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
                'filename':file,
                'bounds':'([{},{}],[{},{}])'.format(x - margin, x + window_size + margin, y - margin, y + window_size + margin)
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
        N   = np.stack([p['NormalX'], p['NormalY'], p['NormalZ']], axis = 1)
        return XYZ, N
    except:
        return None, None




def get_pre_event(file, x, y, window_size, cl):
    if file.endswith('json'):
        reader = 'readers.ept'
    elif file.endswith('laz'):
        reader = 'readers.las'
        
    C = ['Classification[{}:{}]'.format(c, c) for c in cl]
    classes = ','.join(C)
    
    pipeline = [
        {
            'type':reader,
            'filename':file,
            'bounds':'([{},{}],[{},{}])'.format(x, x + window_size, y, y + window_size)
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
        return np.stack([p['X'], p['Y'], p['Z']], axis = 1)
    except:
        return None





''' -------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------- '''

def pdal_icp(pre_event, pos_event, config):
    
    X, Y = [], []
    dx, dy, dz = [], [], []
    DX, DY, DZ = [], [], []
    window_size = config.window_size
    step_size = config.step_size
    bounds = config.bounds
    
    
    for y in range(bounds[2], bounds[3], step_size):
        dxr, dyr, dzr = [], [], []
        for x in range(bounds[0], bounds[1], step_size):
    
            # PDAL pipeline with data bounds set according to the current window.
            # The first window is 'fixed'; The second window is 'moving'.
            # Note that we pad the 'fixed' window so the second window has room to
            # move within the fixed window as the ICP solution converges.
            pipeline = [
                {
                    'type':'readers.ept',
                    'filename':pos_event,
                    'bounds':'([{},{}],[{},{}])'.format(x - 2,
                                                        x + window_size + 2,
                                                        y - 2,
                                                        y + window_size + 2)
                },
                {
                    'type':'readers.ept',
                    'filename':pre_event,
                    'bounds':'([{},{}],[{},{}])'.format(x,
                                                        x + window_size,
                                                        y,
                                                        y + window_size)
                },
                {
                    'type':'filters.icp'
                }
            ]
    
            # Execute the pipeline
            p = pdal.Pipeline(json.dumps(pipeline))
            p.execute()
    
            # Capture the metadata, which contains the ICP transformation
            m = p.metadata
            t = m.get('metadata').get('filters.icp').get('transform')
    
            # Store vector origin and ICP-derived displacement
            try:
                t = [float(val) for val in t.split()]
                X.append(x + window_size/2)
                Y.append(y + window_size/2)
                dxr.append(t[3])
                dyr.append(t[7])
                dzr.append(t[11])
                
                dx.append(t[3])
                dy.append(t[7])
                dz.append(t[11])
            except:
                dxr.append(0)
                dyr.append(0)
                dzr.append(0)
    
            
        DX.append(dxr)
        DY.append(dyr)
        DZ.append(dzr)
        print('{}% done'.format(np.ceil(y / (bounds[3] - bounds[2]) * 100)))
        return (DX, DY, DZ)
    




