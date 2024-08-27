# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:48:32 2024

@author: Alejandro
"""

import scipy
from preprocessing_polsar_data import *
import numpy as np
import keras
from keras import optimizers
from keras import callbacks
import tensorflow as tf
import keras.backend as K
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract, MaxPooling2D, Dropout,Conv2DTranspose, Concatenate, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizer_v2.adam import Adam
from tensorflow.keras import layers, models, regularizers


def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))/2

class DnCNN_Class:
    def __init__(self, flag_train):

        self.IMAGE_WIDTH = IMG_SIZE
        self.IMAGE_HEIGHT = IMG_SIZE
        self.CHANNELS = 4
        self.N_LAYERS = 20
        self.Filters = 64
        
          

    def DnCNN_Model(self, optim):
        tf.keras.backend.clear_session()
        layer_count = 0
        inpt = Input(shape=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.CHANNELS),name = 'input'+str(layer_count))
        # 1st layer, Conv+relu
        layer_count += 1
        x = Conv2D(filters=self.Filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
        layer_count += 1
        x = Activation('relu',name = 'relu'+str(layer_count))(x)
        # depth-2 layers, Conv+BN+relu
        for i in range(self.N_LAYERS-2):
            layer_count += 1
            x = Conv2D(filters=self.Filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
            layer_count += 1
            BatchNormalization(axis=-1, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            layer_count += 1
            x = Activation('relu',name = 'relu'+str(layer_count))(x)  
        # last layer, Conv
        layer_count += 1
        x = Conv2D(filters=self.CHANNELS, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        layer_count += 1
        x = Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise
        self.myModel = Model(inputs=inpt, outputs=x)
        self.myModel.compile(loss=sum_squared_error, metrics = [sum_squared_error], optimizer=optim)
        print("DnCNN created")
