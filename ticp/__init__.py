"""
TICP: moving-window translation-only ICP for 3D change detection from lidar.

    from ticp import compare
    result = compare('before.laz', 'after.laz', window_size=150, step_size=25, output='results/run1')
"""

from .icp import ICPConfig, TICPCancelled, transicp, run_ticp
from .workflow import compare
from .bias import remove_bias
from .utilities import read_point_cloud, compute_normals, write_results

__version__ = '0.1.0'
