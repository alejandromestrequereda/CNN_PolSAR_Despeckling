# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:47:02 2024

@author: Alejandro
"""

import scipy.io as sio
import numpy as np
np.object = object
np.bool = bool
np.int = int
import random
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras import backend as Ks
from keras import optimizers
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.optimizer_v2.adam import Adam
from keras.datasets import mnist
from matplotlib import pyplot as plt
import os
import scipy
from model_git import *
import time

def direct_c2_transform(Svv, Svh):
    
    nl = Svv.shape[0]
    npx = Svv.shape[1]
    
    C2_t = np.zeros((nl, npx, 4), dtype = np.float32)
    
    SI = Svv + Svh
    SQ = Svv + 1j*Svh 

    s0_VV = np.abs(Svv)**2
    s0_VH = np.abs(Svh)**2
    s0_I = np.abs(SI)**2
    s0_Q = np.abs(SQ)**2
    
    C2_t[:,:,0] = s0_VV
    C2_t[:,:,1] = s0_VH
    C2_t[:,:,2]= s0_I
    C2_t[:,:,3] = s0_Q
    
    return C2_t



def inverse_c2_transform(C2_intensity):
    s0_VV = C2_intensity[:,:,0]
    s0_VH = C2_intensity[:,:,1]
    s0_I = C2_intensity[:,:,2]
    s0_Q = C2_intensity[:,:,3]
    
    nl = C2_intensity.shape[0]
    npx = C2_intensity.shape[1]
    
    C2_complex = np.zeros((2,2, nl, npx), dtype = np.complex64)
    
    span = s0_VV + s0_VH  
    s0_Ir = 0.5*( ( s0_I - span ) + 1j*( s0_Q - span ) )
    s0_Qr = 0.5*( ( s0_I - span ) - 1j*( s0_Q - span ) )
    
    C2_complex[0,0, :, :] = s0_VV
    C2_complex[0,1, :, :] = s0_Ir
    C2_complex[1, 0, :, :] = s0_Qr
    C2_complex[1,1,:,:] = s0_VH
    
    return C2_complex


IMG_SIZE = 64

# ----  Input Directory:  TO BE CHANGED BY THE USER  ----#
dir_in = r"D:\Usuarios\Alejandro\Universidad\PolSAR2PolSAR\Murcia\test_image_git\data\\"



# ----  Load Trained Model ----#
DnCNN = DnCNN_Class(0)
myOptimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
DnCNN.DnCNN_Model(myOptimizer)
model = DnCNN.myModel
model.load_weights("DS_DnCNN_3TS_sumse.h5")

# ----  Load Dual-Pol Data ----#

VH_img_file = dir_in + 'crop_vh.npy'
VV_img_file = dir_in + 'crop_vv.npy'

Svv = np.load(VV_img_file)
Svh = np.load(VH_img_file)

[nlines, npixels] = Svv.shape

print('Constructing C2 Transformation (4 intensity bands)')
# Ci = np.zeros((nlines, npixels, 4), dtype = np.float32)

# SI = Svv + Svh
# SQ = Svv + 1j*Svh

# s0_VV = np.abs(Svv)**2.0
# s0_VH = np.abs(Svh)**2.0
# s0_I = np.abs(SI)**2
# s0_Q = np.abs(SQ)**2

# Ci is the input image with the four intensity bands to be filtered with the CNN #
# Ci[:,:,0] = s0_VV
# Ci[:,:,1] = s0_VH
# Ci[:,:,2] = s0_I
# Ci[:,:,3] = s0_Q
Ci = direct_c2_transform(Svv, Svh)

# -------------------------------  Start Filtering ------------------------------ #

# ----  Step 1: Normalization ----#
Ci_denoised = np.zeros((nlines, npixels,4), dtype = np.float32) # Filtered data #
Ci = normalize_C2_single_intensity(Ci)

 
# ----  Step 2: Apply NN model ----#
overlap = int(IMG_SIZE/3) # 1/3 overlap assumed (can be changed) #
N = IMG_SIZE  
n_win_p = int(np.ceil(npixels/overlap))   
n_win_l = int(np.ceil(nlines/overlap))

block = np.zeros((1, IMG_SIZE, IMG_SIZE, 4))

start_time = time.time()
for ix1 in range(0, n_win_l):

    print("Azimuth block: " + str(ix1) + " of " + str(n_win_l))
    
    i1 = ix1 * overlap
    i2 = i1 + IMG_SIZE
    
    if(i2 > nlines - 1):
        i2 = nlines
        i1 = nlines - IMG_SIZE
    
    for ix2 in range(0, n_win_p):
        j1 = ix2 * overlap
        j2 = j1 + IMG_SIZE
    
        if(j2 > npixels - 1):
            j2 = npixels
            j1 = npixels - IMG_SIZE
            
            
        block[0,:,:,:] = Ci[i1:i2, j1:j2,:]
        filt_block = model.predict(block)
        filt_block  = np.squeeze(filt_block)

        # Avoid Edge effect between adjacent image blocks #
        if(ix2 == 0 and ix1 == 0):
            Ci_denoised[i1:i2, 0:N, 0] = filt_block[:, 0:N, 0]
            Ci_denoised[i1:i2, 0:N, 1] = filt_block[:, 0:N, 1]
            Ci_denoised[i1:i2, 0:N, 2] = filt_block[:, 0:N, 2]
            Ci_denoised[i1:i2, 0:N, 3] = filt_block[:, 0:N, 3]
        elif(ix2 > 0 and ix1 == 0):
            Ci_denoised[i1:i2, j1+overlap:j2, 0] = filt_block[:, overlap:N, 0]
            Ci_denoised[i1:i2, j1+overlap:j2, 1] = filt_block[:, overlap:N, 1]
            Ci_denoised[i1:i2, j1+overlap:j2, 2] = filt_block[:, overlap:N, 2]
            Ci_denoised[i1:i2, j1+overlap:j2, 3] = filt_block[:, overlap:N, 3]
        elif(ix1 > 0 and ix2 == 0):
            Ci_denoised[i1+overlap:i2, j1:j2, 0] = filt_block[overlap:N, :, 0]
            Ci_denoised[i1+overlap:i2, j1:j2, 1] = filt_block[overlap:N, :, 1]
            Ci_denoised[i1+overlap:i2, j1:j2, 2] = filt_block[overlap:N, :, 2]
            Ci_denoised[i1+overlap:i2, j1:j2, 3] = filt_block[overlap:N, :, 3]
        else:
            Ci_denoised[i1+overlap:i2, j1+overlap:j2, 0] = filt_block[overlap:N, overlap:N, 0]
            Ci_denoised[i1+overlap:i2, j1+overlap:j2, 1] = filt_block[overlap:N, overlap:N, 1]
            Ci_denoised[i1+overlap:i2, j1+overlap:j2, 2] = filt_block[overlap:N, overlap:N, 2]
            Ci_denoised[i1+overlap:i2, j1+overlap:j2, 3] = filt_block[overlap:N, overlap:N, 3]

print("--- %s seconds ---" % (time.time() - start_time))

# ----  Step 3: Denormalization ----#
Ci_denoised_denorm = denormalize_C2_single_intensity(Ci_denoised)
Ci_original_denorm = denormalize_C2_single_intensity(Ci)


s0_vv_den = Ci_denoised_denorm[:,:,0]
s0_vh_den = Ci_denoised_denorm[:,:,1]
rat_den = s0_vv_den/(s0_vh_den + 1e-3)

s0_vv_or = Ci_original_denorm[:,:,0]
s0_vh_or = Ci_original_denorm[:,:,1]
rat_or = s0_vv_or/(s0_vh_or + 1e-3)

# Generate RGB image and visualize results #

s0_vv_den_norm = normalize_data(s0_vv_den, 0.3)
s0_vh_den_norm = normalize_data(s0_vh_den, 0.07)
rat_den_norm = normalize_data(rat_den, 15)  

s0_vv_or_norm = normalize_data(s0_vv_or, 0.3)
s0_vh_or_norm = normalize_data(s0_vh_or, 0.07)
rat_or_norm = normalize_data(rat_or, 15)

rgb_image_den = np.zeros((nlines, npixels,3), dtype = np.uint8)
rgb_image_den[:,:,0] = 255*s0_vv_den_norm
rgb_image_den[:,:,1] = 255*s0_vh_den_norm
rgb_image_den[:,:,2] = 255*rat_den_norm

rgb_image_or = np.zeros((nlines, npixels,3), dtype = np.uint8)
rgb_image_or[:,:,0] = 255*s0_vv_or_norm
rgb_image_or[:,:,1] = 255*s0_vh_or_norm
rgb_image_or[:,:,2] = 255*rat_or_norm

plt.figure()
plt.imshow(rgb_image_den, interpolation = 'nearest')
plt.show()

plt.figure()
plt.imshow(rgb_image_or, interpolation = 'nearest')
plt.show()


