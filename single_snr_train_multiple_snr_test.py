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

from neural_net_model import neural_net_model


train_SNR = 10
num_epochs = 70
dataset_size = 1_008_000
path_to_data = f"{os.getcwd()}/Datasets_v2/Datafiles/Sim_data/"

norm_tx, norm_rx = 0, 0

print(f'-------------------------------------Training SNR = {train_SNR} dB----------------------------------------')
filename_train = f'{path_to_data}pucch_fading_{train_SNR}dB_{int(dataset_size/1000)}k_norm_tx_{norm_tx}_norm_rx_{norm_rx}.mat'


data = sio.loadmat(filename_train)
X = data['X']
Y = data['Y']

# Z = zeros((len(Y), 1))
# for i in range(len(Y)):
#     if Y[i] == 3:
#         Z[i] = 1
#     elif Y[i] == 6:
#         Z[i] = 2
#     elif Y[i] == 9:
#         Z[i] = 3

Z = Y

Z = to_categorical(Z)

# Split datasets into training and testing
num_train = int(0.75 * dataset_size)
num_test = int(dataset_size - num_train)

x_train, x_test = X[0:num_train, :], X[num_train:, :]
y_train, y_test = Y[0:num_train, :], Y[num_train:, :]
z_train, z_test = Z[0:num_train, :], Z[num_train:, :]

model = neural_net_model()
model.summary()

optimizer = SGD(learning_rate=1e-3, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, z_train, epochs=num_epochs, batch_size=512, validation_split=0.3, verbose=2)

# create save folder if it doesn't exist
save_folder = './Weights'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# save weights
weights_save_path = os.path.join(save_folder, 'weights_128_12_class.h5')
model.save_weights(weights_save_path)

# plt.rcParams['axes.edgecolor'] = 'black'
# plt.figure()
# markers = list(arange(0, num_epochs, 50))
# markers[-1] = markers[-1] - 1
# plt.plot(history.history['loss'], 's-r', label='Training', markevery=markers)
# plt.plot(history.history['val_loss'], '^-b', label='Validation', markevery=markers)
# plt.title(f'Model Loss, SNR = {train_SNR} dB', fontsize=16, fontweight="bold")
# plt.ylabel('Categorical Cross Entropy Loss', fontsize=16, fontweight="bold")
# plt.xlabel('Training Epoch', fontsize=16, fontweight="bold")
# plt.grid(True)
# plt.legend(loc='upper right', prop=dict(weight='bold'))
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')

# plt.savefig(f'./Plots/Loss_{train_SNR}_dB_single_train_multiple_test.png', dpi=400)

# acc_perc = [100*a for a in history.history['accuracy']]
# val_acc_perc = [100*a for a in history.history['val_accuracy']]
# plt.figure()
# plt.plot(acc_perc, 's-r', label='Training', markevery=markers)
# plt.plot(val_acc_perc, '^-b', label='Validation', markevery=markers)
# plt.title(f'Model Accuracy, SNR = {train_SNR} dB', fontsize=16, fontweight="bold")
# plt.ylabel('Accuracy (%)', fontsize=16, fontweight="bold")
# plt.xlabel('Training Epoch', fontsize=16, fontweight="bold")
# plt.grid(True)
# plt.legend(loc='lower right', prop=dict(weight='bold'))
# plt.xticks(fontsize=12, fontweight='bold')
# # plt.yticks(arange(floor(min(acc_perc)), floor(max(val_acc_perc))+1, 2), fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# plt.savefig(f'./Plots/Accuracy_{train_SNR}_dB_single_train_multiple_test.png', dpi=400)



