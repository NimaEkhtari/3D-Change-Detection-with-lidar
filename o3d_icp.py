import json
import open3d as o3d
import numpy as np
import pdal
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine




# ICP window and step sizes
window_size = 40
step_size = 20
device = o3d.core.Device("CPU:0")
dtype = o3d.core.Dtype.Float32


# Variables to hold the ICP vector origins (X, Y) and displacements (dx, dy)
X = []
Y = []
dx = []
dy = []
dz = []
DX, DY, DZ = [], [], []
rmse, fitn = [], []




''' *********************************************************************** '''
'''                                  ICP settings                           '''
''' *********************************************************************** '''
# Search distance for Nearest Neighbour Search [Hybrid-Search is used].
max_correspondence_distance = 10
# Initial alignment or source to target transform.
trans_init = np.asarray(np.eye(4))

reg_type = o3d.pipelines.registration.TransformationEstimationPointToPlane()
# Convergence-Criteria for Vanilla ICP
threshold = 1.5
# Down-sampling voxel-size. When set to -1, it is not down-sampling
# voxel_size = 0.025
# Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
save_loss_log = True
''' *********************************************************************** '''
        




''' ***********************************************************************'''
''' ***********************************************************************'''
'''               Read tiles and ensure matching filenames                 '''
''' ***********************************************************************'''
''' ***********************************************************************'''

# Path to pre- and post-event indexed point clouds (EPT)
pre_event = r'.\before_ept\ept.json'
post_event = r'.\after_ept\ept.json'


# Slide a window through the analysis area

B = [0, 5100, 0, 5200]


def open3d_icp(pre_event, pos_event, config):
    window_size = config.window_size
    step_size = config.step_size
    margin = config.margin
    bounds = config.bounds
    
    
    for y in range(bounds[2], bounds[3], step_size):
        dxr, dyr, dzr = [], [], []
        for x in range(bounds[0], bounds[1], step_size):
    
            # PDAL pipeline with data bounds set according to the current window.
            # The first window is 'fixed'; The second window is 'moving'.
            # Note that we pad the 'fixed' window so the second window has room to
            # move within the fixed window as the ICP solution converges.
            pipe1 = [
                {
                    'type':'readers.ept',
                    'filename':pre_event,
                    'bounds':'([{},{}],[{},{}])'.format(x - margin,
                                                        x + window_size + margin,
                                                        y - margin,
                                                        y + window_size + margin)
                }
            ]
            pipe2 = [
                {
                    'type':'readers.ept',
                    'filename':post_event,
                    'bounds':'([{},{}],[{},{}])'.format(x,
                                                        x + window_size,
                                                        y,
                                                        y + window_size)
                }
            ]
    
            # Execute the pipeline
            p1 = pdal.Pipeline(json.dumps(pipe1))
            p1.execute()
            p2 = pdal.Pipeline(json.dumps(pipe2))
            p2.execute()
            
    
            
            if ((len(p1.arrays[0]) < 500) | (len(p2.arrays[0]) < 500)):
                dxr.append(0)
                dyr.append(0)
                dzr.append(0)
                continue
    
                
            # Create an empty point cloud
            # Use pcd.point to access the points' attributes
            # pcd = o3d.t.geometry.PointCloud(device)
    
            bef_arr = np.stack([p1.arrays[0]["X"], p1.arrays[0]["Y"], p1.arrays[0]["Z"]], axis = 1)
            aft_arr = np.stack([p2.arrays[0]["X"], p2.arrays[0]["Y"], p2.arrays[0]["Z"]], axis = 1)
            source = o3d.geometry.PointCloud()
            # source.points = o3d.utility.Vector3dVector(bef_arr)
            target = o3d.geometry.PointCloud()
            # target.points = o3d.utility.Vector3dVector(aft_arr)
            
            # source = pcd.point.positions = o3d.core.Tensor(bef_arr)
            # target = pcd.point.positions = o3d.core.Tensor(aft_arr)
            
            # Transforming points to centroid to avoid scale issues in SVD
            ac = np.concatenate((bef_arr, aft_arr), axis=0)        
            mea = np.mean(ac, axis=0, dtype=np.float64)
            b = bef_arr - mea
            # source.point['positions'] = np.float32(b)
            source.points = o3d.utility.Vector3dVector(np.float32(b))
            d = aft_arr - mea
            # target.point['positions'] = np.float32(d)
            target.points = o3d.utility.Vector3dVector(np.float32(d))
            
            
            
            # Perform ICP
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 3, max_nn = 10)) 
            reg = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, reg_type)
    
            Res = reg.transformation[0:3,3]
            dx.append(Res[0])
            dy.append(Res[1])
            dz.append(Res[2])
            dxr.append(Res[0])
            dyr.append(Res[1])
            dzr.append(Res[2])
            
            # rmse.append(registration_icp.inlier_rmse)
            # fitn.append(registration_icp.fitness)
    
            X.append(x + window_size/2)
            Y.append(y + window_size/2)
    
            
        DX.append(dxr)
        DY.append(dyr)
        DZ.append(dzr)
        print('{}% done'.format(np.ceil(y / (B[3] - B[2]) * 100)))
        
    
    
    # Plot the ICP vectors
    plt.figure()
    plt.quiver(X, Y, dx, dy, angles='xy', scale_units='xy')
    plt.axis('equal')
    plt.show()
    
    res = np.stack([X, Y, dx, dy, dz], axis = 1)
    sname = 'o3d_shifted_results_{0}_{1}.txt'.format(window_size, step_size)
    np.savetxt(sname, res, delimiter=' ', fmt='%.3f')



    res = np.stack([X, Y, dx, dy, dz], axis = 1)
    sname = '{0}_{1}_{2}.txt'.format(config.output_basename, window_size, step_size)
    np.savetxt(sname, res, delimiter=' ', fmt='%.3f')
    return res, np.stack([DX, DY, DZ], axis = 2)







def write_raster(outname, array, ul, pixel_width, pixel_height, epsg):
    
    width, height, num_bands = get_dims(array)
    crs = rasterio.crs.CRS.from_epsg(epsg)
    transform = Affine.translation(ul[0], ul[1]) * Affine.scale(pixel_width, -pixel_height)
    
        # Set raster profile
    profile = {
    'driver': 'GTiff',
    'dtype': rasterio.float64,
    'nodata': 0,
    'width': width,
    'height':height,
    'count': num_bands,
    'crs': crs,
    'transform': transform,
    'tiled': True,
    'compress': 'lzw'
    }
    
    
    with rasterio.open(outname, 'w', **profile) as dst:
        for band in range(num_bands):
            dst.write(array[:, :, band].astype(rasterio.float64), band + 1)
    
    


def get_dims(array):
    sh = array.shape
    width  = sh[1]
    height = sh[0]
    nbands = sh[2] if len(sh) == 3 else 1
    return (width, height, nbands)



res, disp = open3d_icp(pre_event, pos_event, config)



#rasterOrigin = (np.min(X), np.max(Y))  # Upper left corner
# rasterOrigin = (594400, 4151500)
rasterOrigin = [min(res[:, 0]), max(res[:, 1])]
pixelWidth = step_size
pixelHeight = step_size
newRasterfn = 'o3d_shifted_results_{0}_{1}.tif'.format(window_size, step_size)

arrays = np.stack([np.array(DX), np.array(DY), np.array(DZ)], axis=2)   # This is a list of ndarrays
write_raster(newRasterfn, np.flipud(arrays), rasterOrigin, pixelWidth, pixelHeight, 32610)     
        



from windrose import WindroseAxes

def rose_plot(x, y):
    if x.ndim > 1:
        x = x.reshape(x.shape[0] * x.shape[1])
        y = y.reshape(y.shape[0] * y.shape[1])
    magnitude = np.linalg.norm(np.vstack((x, y)), axis=0)
    direction = np.arctan2(x, y) * (180 / np.pi) # This needs to be like azimuth
    direction[np.where(direction < 0)] += 360
    # direction -= 90
    
    # direction = direction[np.where(magnitude < np.percentile(magnitude, 99))]
    # magnitude = magnitude[np.where(magnitude < np.percentile(magnitude, 99))]

    direction = direction[np.where(magnitude < 1)]
    magnitude = magnitude[np.where(magnitude < 1)] 
    
    ax = WindroseAxes.from_ax()
    ax.bar(direction, magnitude, nsector=36, bins=10, normed=True, opening=1.0, edgecolor='white')
    ax.set_legend()
    
    return magnitude, direction

ddx = np.array(dx)
ddy = np.array(dy)
magn, dire = rose_plot(ddx, ddy)

    
        
        
        
        
        
