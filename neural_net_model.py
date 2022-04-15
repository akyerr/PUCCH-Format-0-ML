import scipy.io as sio
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, ReLU, BatchNormalization, Flatten, Reshape, Conv2D

from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, Conv2DTranspose, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import zeros, reshape, mean, array, squeeze, expand_dims, abs, angle, argmax, arange, floor, ceil, ravel
from numpy.random import normal
from numpy.linalg import norm
from numpy import concatenate
import os
import datetime
from pickle import dump, load
import seaborn as sns
plt.style.use('seaborn-whitegrid')




def neural_net_model():
    # Build Neural Network and Train
    # input_signal = Input(shape=(24,), name='input_layer')
    # x = Dense(64, activation='relu')(input_signal)
    # x = Dropout(rate=0.5)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    # output_pred = Dense(4, activation='softmax')(x)
    input_signal = Input(shape=(24,), name='input_layer')
    x = Dense(128, activation='relu')(input_signal)
    x = Dropout(rate=0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    output_pred = Dense(4, activation='softmax')(x)
    model = Model(input_signal, output_pred)

    return model
