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


def test_sep_snr_sim(model, test_SNR, dataset_size, path_to_data):
    norm_tx, norm_rx = 1, 1
    num_train = int(0.75 * dataset_size)
    num_test = int(dataset_size - num_train)
    #----------------------------------------Test multiple SNRs----------------------------------------#

    final_test_loss = zeros(len(test_SNR), dtype=float)
    final_test_accuracy = zeros(len(test_SNR), dtype=float)


    for idx, snr in enumerate(test_SNR):
        print(f'-----------------------------Sim testing SNR = {snr} dB--------------------------------')

        #----------------------------------------Test sim data----------------------------------------#
        filename_test = f'{path_to_data}/Sim_data/pucch_fading_{snr}dB_{int(dataset_size/1000)}k_norm_tx_{norm_tx}_norm_rx_{norm_rx}.mat'
        data = sio.loadmat(filename_test)
        X = data['X']
        Y = data['Y']

        Z = Y
        # Z = zeros((len(Y), 1))
        # for i in range(len(Y)):
        #     if Y[i] == 3:
        #         Z[i] = 1
        #     elif Y[i] == 6:
        #         Z[i] = 2
        #     elif Y[i] == 9:
        #         Z[i] = 3

        Z = to_categorical(Z)

        x_test = X[num_train:, :]
        y_test = Y[num_train:, :]
        z_test = Z[num_train:, :]

        test_loss, test_acc = model.evaluate(x_test, z_test, verbose=2)


        final_test_loss[idx] = test_loss
        final_test_accuracy[idx] = test_acc * 100

        # predictions = model.predict(x_test)

        # recovered_CS = zeros(predictions.shape[0], dtype=int)
        # applied_CS = zeros(predictions.shape[0], dtype=int)
        # possible_CS = [0, 1, 2, 3]
        # for i in range(predictions.shape[0]):
        #     recovered_CS[i] = possible_CS[argmax(predictions[i, :])]
        #     applied_CS[i] = possible_CS[argmax(z_test[i, :])]

        # conf_mat = tf.math.confusion_matrix(applied_CS, recovered_CS)
        # plt.figure(figsize=(10, 8))
        # sns.set(font_scale=1.6)
        # sns.heatmap(conf_mat, xticklabels=["0", "3", "6", "9"], yticklabels=["0", "3", "6", "9"], annot=True, fmt="g", linewidths=.5)
        # plt.xlabel('Prediction (NN Classified Cyclic Shift)', fontsize=16, fontweight="bold")
        # plt.ylabel('Label (Applied Cyclic Shift)', fontsize=16, fontweight="bold")
        # plt.title(f'Confusion Matrix for Simulated Test Dataset at SNR = {snr}dB', fontsize=16, fontweight="bold")
        # plt.savefig(f'./Plots/Conf_mtx_sim_{snr}_dB.png', dpi=400)

    return final_test_loss, final_test_accuracy
