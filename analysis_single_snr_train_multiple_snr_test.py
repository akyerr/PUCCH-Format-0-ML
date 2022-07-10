import os
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, ReLU, BatchNormalization, Flatten, Reshape, Conv2D
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, Conv2DTranspose, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from numpy import zeros, reshape, mean, array, squeeze, expand_dims, abs, angle, argmax, arange, floor, ceil, ravel
from numpy.random import normal
from numpy.linalg import norm
from numpy import concatenate
from pickle import dump, load
import seaborn as sns
sns.set(style="white", color_codes=False)

# plt.style.use('seaborn-whitegrid')

from test_comb_snr_hw import test_comb_snr_hw
from test_sep_snr_hw import test_sep_snr_hw
from test_sep_snr_sim import test_sep_snr_sim

from neural_net_model import neural_net_model


model = neural_net_model()
model.summary()

# load weights
weights_load_path = os.path.join('.', 'Weights', 'weights_128_12_class_unnorm.h5')
model.load_weights(weights_load_path)

# compile model
optimizer = SGD(learning_rate=1e-3, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

path_to_data = f"{os.getcwd()}/Datasets_v2/Datafiles/"
dataset_size = 1_008_000

separate_sim = False
separate_hw = True


# Test separate SNRs Sim
if separate_sim:
    test_SNR = [0, 5, 10, 15, 20]
    # Test separate SNRs Sim
    final_test_loss, final_test_accuracy = test_sep_snr_sim(model, test_SNR, dataset_size, path_to_data)

# Test separate SNRs HW
if separate_hw:
    hw_test_file = f'{path_to_data}Hw_test_data/pucch_hw_data.mat'
    final_test_loss_hw, final_test_accuracy_hw = test_sep_snr_hw(model, hw_test_file)






# plt.rcParams.update(plt.rcParamsDefault)

# if separate_sim:
#     filename_fft = f"{path_to_data}F0data_23March.mat"
#     data = sio.loadmat(filename_fft)
#     fft_accuracy = data['accuracy']

#     fft_accuracy = [77.1640, 90.9560, 96.0940, 97.4300, 97.8360]

#     plt.style.use('seaborn-whitegrid')
#     plt.rcParams['axes.edgecolor'] = 'black'
#     plt.figure()
#     plt.plot(test_SNR, final_test_accuracy, 's-', label='Neural Network (simulated test dataset)', color = (0.95, 0, 0))
#     plt.plot(test_SNR, fft_accuracy, '^-', label='DFT Algorithm', color = (0, 0, 0.95))

#     plt.title('Model Accuracy', fontsize=16, fontweight="bold")
#     plt.ylabel('Accuracy (%)', fontsize=16, fontweight="bold")
#     plt.xlabel('SNR (dB)', fontsize=16, fontweight="bold")
#     plt.xticks(ticks=test_SNR, fontsize=12, fontweight='bold')
#     plt.yticks(arange(floor(min(fft_accuracy))-1, floor(max(final_test_accuracy))+3, 2), fontsize=12, fontweight='bold')
#     plt.grid(True)
#     plt.legend(loc='lower right', fontsize=12, prop=dict(weight='bold', size=12))
#     plt.savefig(f'./Plots/Accuracy_vs_SNR_single_train_multiple_test.png', dpi=400)

# if separate_hw:
#     f1 = array([0, 0, 0, 3, 333])
#     f1r = array([333, 3, 0, 0, 0])
#     f2 = array([0, 26, 97, 0, 621])
#     f2r = array([621, 0, 97, 26, 0])

#     f1 = array([0, 0, 0, 412, 1388]) # slot 14
#     f1r = array([1388, 412, 0, 0, 0])

#     f2 = array([0, 0, 97, 8, 891]) # slot 13
#     f2r = array([891, 8, 97, 26, 26])

#     fft_accuracy_hw = 100 - ((f1r + f2r)/5200)*100

#     test_SNR_hw = [1.8, 4.9, 9.5, 15.6, 21.9]
#     plt.style.use('seaborn-whitegrid')
#     plt.rcParams['axes.edgecolor'] = 'black'
#     plt.figure()
#     plt.plot(test_SNR_hw, final_test_accuracy_hw, 's-', label='Neural Network (hardware test dataset)', color = (0.95, 0, 0))
#     plt.plot(test_SNR_hw, fft_accuracy_hw, '^-', label='DFT Algorithm', color = (0, 0, 0.95))

#     plt.title('Model Accuracy', fontsize=16, fontweight="bold")
#     plt.ylabel('Accuracy (%)', fontsize=16, fontweight="bold")
#     plt.xlabel('SNR (dB)', fontsize=16, fontweight="bold")
#     plt.xticks(ticks=test_SNR_hw, fontsize=12, fontweight='bold')
#     plt.yticks(arange(floor(min(fft_accuracy_hw))-4, floor(max(final_test_accuracy_hw))+4, 4), fontsize=12, fontweight='bold')
#     plt.grid(True)
#     plt.legend(loc='lower right', fontsize=12, prop=dict(weight='bold', size=12))
#     plt.savefig(f'./Plots/Accuracy_vs_SNR_single_train_multiple_test_hw.png', dpi=400)
