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


def test_sep_snr_hw(model, occupied_symb_per_slot, test_SNR, dataset_size, path_to_data):

    print(test_SNR)

    # num_train = int(0.75 * dataset_size)
    # num_test = int(dataset_size - num_train)
    #----------------------------------------Test multiple SNRs----------------------------------------#

    final_test_loss_hw = zeros(len(test_SNR), dtype=float)
    final_test_accuracy_hw = zeros(len(test_SNR), dtype=float)

    for idx, snr in enumerate(test_SNR):

        # if idx == 0:
        #     print_snr = 5
        # elif idx == 1:
        #     print_snr = 10
        # elif idx == 2:
        #     print_snr = 15
        # elif idx == 3:
        #     print_snr = 20
        # elif idx == 4:
        #     print_snr = 25
        print(f'-----------------------------HW testing SNR = {snr} dB--------------------------------')

        #----------------------------------------Test HW data----------------------------------------#
        filename_test_hw = f"{path_to_data}pucch_hw_data_{occupied_symb_per_slot}_{snr}dB_slot_13_14.mat"
        data_hw = sio.loadmat(filename_test_hw)
        X_hw = data_hw[f'x_hw_{snr}']
        Y_hw = data_hw[f'y_hw_{snr}']



        Z_hw = zeros((len(Y_hw), 1))
        for i in range(len(Y_hw)):
            if Y_hw[i] == 3:
                Z_hw[i] = 1
            elif Y_hw[i] == 6:
                Z_hw[i] = 2
            elif Y_hw[i] == 9:
                Z_hw[i] = 3
        Z_hw = to_categorical(Z_hw)


        x_test_hw = X_hw
        y_test_hw = Y_hw
        z_test_hw = Z_hw

        test_loss_hw, test_acc_hw = model.evaluate(x_test_hw, z_test_hw, verbose=2)


        final_test_loss_hw[idx] = test_loss_hw
        final_test_accuracy_hw[idx] = test_acc_hw * 100

        predictions_hw = model.predict(x_test_hw)


        recovered_CS_hw = zeros(predictions_hw.shape[0], dtype=int)
        applied_CS_hw = zeros(predictions_hw.shape[0], dtype=int)
        possible_CS = [0, 1, 2, 3]
        for i in range(predictions_hw.shape[0]):
            recovered_CS_hw[i] = possible_CS[argmax(predictions_hw[i, :])]
            applied_CS_hw[i] = possible_CS[argmax(z_test_hw[i, :])]


        conf_mat_hw = tf.math.confusion_matrix(applied_CS_hw, recovered_CS_hw)

        print(snr)
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.6)
        sns.heatmap(conf_mat_hw, xticklabels=["0", "3", "6", "9"], yticklabels=["0", "3", "6", "9"], annot=True, fmt="g", linewidths=.5)
        plt.xlabel('Prediction (NN Classified Cyclic Shift)', fontsize=16, fontweight="bold")
        plt.ylabel('Label (Applied Cyclic Shift)', fontsize=16, fontweight="bold")
        plt.title(f'Confusion Matrix for Hardware Test Dataset at SNR = {snr}dB', fontsize=16, fontweight="bold")
        plt.savefig(f'./Plots/Conf_mtx_hw_{snr}_dB.png', dpi=400)

    return final_test_loss_hw, final_test_accuracy_hw
