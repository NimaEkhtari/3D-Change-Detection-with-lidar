# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:54:35 2024

@author: nekhtari
"""


import numpy as np
# import faiss
import pdal
import json
import argparse
import math
import utilities
from scipy.spatial import KDTree



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
        self.outlier_multiplier  = conf.get('outlier_multiplier') #points larger than this many times the RMSE removed
        self.outlier_percent     = conf.get('outlier_percent') #However, we will keep 95% of the points.
        self.method              = conf.get('method')
        self.has_normal_post     = conf.get('has_normal_post')
        self.null                = conf.get('null')
        self.output_basename     = conf.get('output_basename')

        


def transicp(moving, fixed, fixed_normal, Tconverge, Tmax_iter):
    '''
    Parameters
    ----------
    moving : numpy array of size m x 3
        3D coordinates of moving point cloud points.
    fixed : numpy array of size n x 3
        3D coordinates of fixed point cloud points.
    fixed_normal : numpy array of size n x 3
        3D normal vectoe associated to local plane fitted to point in the fixed point cloud.
    Tconverge : scalar value
        ICP convergence threshold.
    Tmax_iter : integer value
        number of maximum iterations of the ICP algorithm.

    Returns
    -------
    icp_trans : list
        estimated 3D shift vector.
    RMSE : scalar value
        RMSE of the ICP algorithm.
    Max : scalar
        maximum value of residual.

    '''
    loop = True
    icp_trans = [0,0,0]
    count = 0
    wsquared = 0
    k = 1
    
    means = np.round(np.mean(fixed, axis = 0), 2)
    X1 = moving - means
    X2 = fixed - means
       
    kdtree = KDTree(X2)
    
    
    while loop == True:
        #Set Up The Point Indexing - Needs to Be Updated Every Iteration
        search_pts = X1 + icp_trans
        D, I = kdtree.query(search_pts, k = k)
        
        # cov_target = cov_fixed[I, :]
        Normal = fixed_normal[I, :]
        Vec_Diff = (search_pts - X2[I, :])
        misc = np.sum(Vec_Diff * Normal, axis = 1)

        A = Normal
        AT = A.transpose()
        ATA = AT.dot(A)
        N = np.linalg.inv(ATA)
        U = AT.dot(misc)
        delcap = -N.dot(U)
        # offset = np.linalg.norm(delcap)
        icp_trans = icp_trans + delcap

        
        count = count + 1
        if np.all(np.abs(delcap) < Tconverge) or count > Tmax_iter:
            loop = False
    
    # Compute Final RMSE of Misclosure
    wsquared = 0
    resi = 0
    XX2 = np.squeeze(X2[I, :])
    Vec_Diff2 = (search_pts - XX2)  # I think we already added it no need to add again + delcap.transpose()
    NNormal = np.squeeze(fixed_normal[I, :])
    resi = np.sum(Vec_Diff2 * NNormal, axis = 1)

    # outlier = np.sum(np.abs(resi) > 0.02)
    wsquared = np.sum(resi ** 2)
    RMSE = math.sqrt(wsquared / len(X1))
    Max = np.amax(np.fabs(misc)) 
   
    return (icp_trans, RMSE, Max)





def run_transicp(pre_event, pos_event, config):
    
    if config.method == 'translation_only':
            
        # ICP window and step sizes
        window_size = config.window_size
        step_size = config.step_size
        margin = config.margin
        bounds = config.bounds
        null = config.null
        Tconverge = config.Tconverge
        Tmax_iter = config.Tmax_iter
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

            
            if (Xa is None) or (Xb is None):
                dxr.append(null)
                dyr.append(null)
                dzr.append(null)
                continue
            if ((len(Xa) < config.min_points) | (len(Xb) < config.min_points)):
                dxr.append(null)
                dyr.append(null)
                dzr.append(null)
                continue
            
            
            res1, rmse, Max = transicp(Xb, Xa, Na, Tconverge, Tmax_iter)
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
    
    
 
    

def get_pos_event(pos_event, x, y, margin, window_size, calc_normal, cl):
    
    C = ['Classification[{}:{}]'.format(c, c) for c in cl]
    classes = ','.join(C)
    if calc_normal:
        pipeline = [
            {
                'type':'readers.ept',
                'filename':pos_event,
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
                'type':'readers.ept',
                'filename':pos_event,
                'bounds':'([{},{}],[{},{}])'.format(x - margin, x + window_size + margin, y - margin, y + window_size + margin)
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




def get_pre_event(pre_event, x, y, window_size, cl):
    C = ['Classification[{}:{}]'.format(c, c) for c in cl]
    classes = ','.join(C)
    
    pipeline = [
        {
            'type':'readers.ept',
            'filename':pre_event,
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
    




