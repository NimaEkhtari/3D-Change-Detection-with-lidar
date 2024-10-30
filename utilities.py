# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:02:00 2024

@author: nekhtari
"""

import json
import pdal



def get_metadata(file):
    if file.endswith('json'):
        reader = 'readers.ept'
    elif file.endswith('laz'):
        reader = 'readers.las'
    
    pipeline = {
        "pipeline": [
            {
                "type": reader,
                "filename": file
            },
            {
                "type":"filters.decimation",
                "step": 100
            },
            {
                "type": "filters.stats",  # Add filter to calculate statistics
                "dimensions": "X,Y"  # Specify X and Y dimensions for which stats are calculated
            }
        ]
    }
    
    pipeline_obj = pdal.Pipeline(json.dumps(pipeline))
    pipeline_obj.execute()
    
    # Directly use pipeline_obj.metadata since it's already a dict
    metadata = pipeline_obj.metadata
    
    Xmin = metadata['metadata']['filters.stats']['statistic'][0]['minimum']
    Xmax = metadata['metadata']['filters.stats']['statistic'][0]['maximum']
    Ymin = metadata['metadata']['filters.stats']['statistic'][1]['minimum']
    Ymax = metadata['metadata']['filters.stats']['statistic'][1]['maximum']
    bounds = [Xmin, Xmax, Ymin, Ymax]
    
    dimensions = [field[0] for field in pipeline_obj.arrays[0].dtype.descr]
    if 'NormalX' in dimensions:
        has_normal = True
    else:
        has_normal = False

    
    return (bounds, has_normal)




    
    
    
    
    
    
    
    
    