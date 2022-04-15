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



def test_comb_snr_hw(model, filename):

    data = sio.loadmat(filename)
    X_hw = data['X_hw']
    Y_hw = data['Y_hw']

    Z_hw = zeros((len(Y_hw), 1))
    for i in range(len(Y_hw)):
        if Y_hw[i] == 3:
            Z_hw[i] = 1
        elif Y_hw[i] == 6:
            Z_hw[i] = 2
        elif Y_hw[i] == 9:
            Z_hw[i] = 3
    print(f'Before one hot: {Z_hw.shape}')

    Z_hw = to_categorical(Z_hw)

    # print(type(Z_hw))
    # print(f'After one hot: {Z_hw.shape}')
    # print(Z_hw)
    x_test_hw = X_hw
    y_test_hw = Y_hw
    z_test_hw = Z_hw
    # z_test_hw = zeros((1000, 4), dtype=int)
    # for i in range(1000):
    #     z_test_hw[i, :] = array([1, 0, 0, 0])

    # print(x_test_hw.shape)
    # print(z_test_hw.shape)
    print(f'-------------------------------------Testing on HW captures----------------------------------------')
    test_loss_hw, test_acc_hw = model.evaluate(x_test_hw, z_test_hw, verbose=2)
    predictions_hw = model.predict(x_test_hw)

    recovered_CS_hw = zeros(predictions_hw.shape[0], dtype=int)
    applied_CS_hw = zeros(predictions_hw.shape[0], dtype=int)
    possible_CS = [0, 1, 2, 3]
    for i in range(predictions_hw.shape[0]):
        recovered_CS_hw[i] = possible_CS[argmax(predictions_hw[i, :])]
        applied_CS_hw[i] = possible_CS[argmax(z_test_hw[i, :])]


    conf_mat_hw = tf.math.confusion_matrix(applied_CS_hw, recovered_CS_hw)
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.6)
    sns.heatmap(conf_mat_hw, xticklabels=["0", "3", "6", "9"], yticklabels=["0", "3", "6", "9"], annot=True, fmt="g", linewidths=.5)
    plt.xlabel('Prediction (NN Classified Cyclic Shift)', fontsize=16, fontweight="bold")
    plt.ylabel('Label (Applied Cyclic Shift)', fontsize=16, fontweight="bold")
    plt.title(f'Confusion Matrix for Hardware Test Dataset (All SNRs))', fontsize=16, fontweight="bold")
    plt.savefig(f'./Plots/Conf_mtx_hw_comb_snr_dB.png', dpi=400)
