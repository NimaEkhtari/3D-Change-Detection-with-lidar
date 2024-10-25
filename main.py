# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:56:16 2024

@author: nekhtari
"""

import numpy as np
import pdal
import json
import icp
import utilities
import matplotlib.pyplot as plt

operation = 'ticp'
# Path to pre- and post-event indexed point clouds (EPT)
pre_event = r'D:\Working\San Andreas and Calaveras\Data\new\before_ept\ept.json'
pos_event = r'D:\Working\San Andreas and Calaveras\Data\new\after_ept\ept.json'


bounds_pre, has_normals_pre = utilities.get_metadata(pre_event)
bounds_pos, has_normals_pos = utilities.get_metadata(pos_event)

bounds = []
bounds.append(int(min(bounds_pre[0], bounds_pos[0])))
bounds.append(int(max(bounds_pre[1], bounds_pos[1])))
bounds.append(int(min(bounds_pre[2], bounds_pos[2])))
bounds.append(int(max(bounds_pre[3], bounds_pos[3])))



operation = 'translation_only'



bounds = [2226, 2550, 2458, 2758]



if operation == 'translation_only':
    configs = {
    'bounds' : bounds,
    'method' : 'translation_only',
    'threshold' : 20,
    'window_size' : 100,
    'step_size' : 100,
    'margin': 5,
    'min_points' : 500,
    'Tconverge' : 0.0005,
    'Tmax_iter' : 20,
    'outlier_multiplier' : 5,
    'outlier_percent' : 0.95,
    'has_normal_post' : has_normals_pos,
    'output_basename' : 'trans_icp_results'
    }

    config = icp.icp_configs(configs)
    res, disp = icp.run_transicp(pre_event, pos_event, config)




# # Plot the ICP vectors
# plt.figure()
# plt.quiver(X, Y, dx, dy, angles='xy', scale_units='xy')
# plt.axis('equal')
# plt.show()

# res = np.stack([X, Y, dx, dy, dz], axis = 1)
# res[:, 0] = res[:, 0] + 698000
# res[:, 1] = res[:, 1] + 4005200
# np.savetxt('pdal_shifted_results_100_50.txt', res, delimiter=' ', fmt='%.3f')