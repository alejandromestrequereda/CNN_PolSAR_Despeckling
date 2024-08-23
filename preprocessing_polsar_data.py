# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:19:28 2024

@author: Alejandro
"""

import numpy as np

IMG_SIZE = 64


def denormalize_C2_single_intensity(C2_array):
    
    min_val = 0.0
    max_val1 = 0.4
    max_val2 = 0.1
    
    arr_shape = C2_array.shape
    
    C2_array_denorm = np.zeros((arr_shape[0], arr_shape[1], arr_shape[2] ), dtype = np.float32)
    
    C2_array_denorm[:,:,0] = (C2_array[:,:,0]*(max_val1 - min_val))+min_val
    C2_array_denorm[:,:,1] = (C2_array[:,:,1]*(max_val2 - min_val))+min_val
    C2_array_denorm[:,:,2] = (C2_array[:,:,2]*(max_val1 - min_val))+min_val
    C2_array_denorm[:,:,3] = (C2_array[:,:,3]*(max_val1 - min_val))+min_val
    
    
    return C2_array_denorm

def normalize_C2_single_intensity(C2_array):
    
    min_val = 0.0
    max_val1 = 0.4
    max_val2 = 0.1
    
    arr_shape = C2_array.shape
    
    C2_array_norm = np.zeros((arr_shape[0], arr_shape[1], arr_shape[2] ), dtype = np.float32)
    C2_array_norm[:,:,0] = np.clip((C2_array[:,:,0] - min_val)/(max_val1 - min_val), 0.0, 1.0)
    C2_array_norm[:,:,1] = np.clip((C2_array[:,:,1] - min_val)/(max_val2 - min_val), 0.0, 1.0)
    C2_array_norm[:,:,2] = np.clip((C2_array[:,:,2] - min_val)/(max_val1 - min_val), 0.0, 1.0)
    C2_array_norm[:,:,3] = np.clip((C2_array[:,:,3] - min_val)/(max_val1 - min_val), 0.0, 1.0)
    
    
    return C2_array_norm

def normalize_C2_intensity(C2_array):
    
    min_val = 0.0
    max_val1 = 0.4
    max_val2 = 0.1
    
    arr_shape = C2_array.shape
    
    C2_array_norm = np.zeros((arr_shape[0], arr_shape[1], arr_shape[2], arr_shape[3] ), dtype = np.float32)
    C2_array_norm[:,:,:,0] = np.clip((C2_array[:,:,:,0] - min_val)/(max_val1 - min_val), 0.0, 1.0)
    C2_array_norm[:,:,:,1] = np.clip((C2_array[:,:,:,1] - min_val)/(max_val2 - min_val), 0.0, 1.0)
    C2_array_norm[:,:,:,2] = np.clip((C2_array[:,:,:,2] - min_val)/(max_val1 - min_val), 0.0, 1.0)
    C2_array_norm[:,:,:,3] = np.clip((C2_array[:,:,:,3] - min_val)/(max_val1 - min_val), 0.0, 1.0)
    
    
    return C2_array_norm


# -------------------------------------------------------------------------------------------- #

def normalize_data(array, max_v):
    
    min_val = 0.0
    max_val = max_v
    
    array_norm = np.clip((array - min_val)/(max_val - min_val), 0.0, 1.0)
    
    return array_norm
