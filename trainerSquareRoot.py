from keras.metrics import MeanSquaredError
from sklearn.utils import shuffle
from tensorflow import keras
import tensorflow as tf
print('Tensorflow/Keras: %s' % keras.__version__)

from keras.models import Sequential
from keras import Input
from keras.layers import Dense

import pandas as pd
print('pandas: %s' % pd.__version__)

import numpy as np
print('numpy: %s' % np.__version__)

import sklearn
print('sklearn: %s' % sklearn.__version__)
from sklearn.model_selection import train_test_split  # for splitting the data into train and test samples

import plotly
print('plotly: %s' % plotly.__version__)

import sys
import os


def print_data(expected, predicted):
    print([[expected[i], predicted[i], abs(expected[i] - predicted[i])] for i in range(len(expected))])


# main_dir = os.path.dirname(sys.path[0])

dimension = 1
samples = 1000000
labels = ['a{}'.format(i) for i in range(1, dimension + 1)] #+ \
        # ['b{}'.format(i) for i in range(1, dimension + 1)]

model2 = Sequential()
model2.add(Input(shape=(dimension,)))
model2.add(Dense(units=4*dimension, activation='relu', input_dim=dimension))
model2.add(Dense(units=16*dimension, activation='relu'))
model2.add(Dense(units=4*dimension, activation='relu'))
model2.add(Dense(units=1, activation='relu'))
model2.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError(), 'Precision'])

# model2 = keras.models.load_model('models/multiplication/2_model/')

df = pd.read_csv('csv/training/squareRoot/squareRoot_100.csv')
X = df[labels]
y = df['distance'].values
df2 = pd.read_csv('csv/validation/squareRoot/squareRoot_10000.csv')
X2 = df[labels]
y2 = df['distance'].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=0)

X_train = np.array(X)
#X_test = np.array(X_test)
y_train = np.array(y)
#y_test = np.array(y_test)
X_valid = np.array(X2)
y_valid = np.array(y2)

X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)


# es_cbk = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000, mode="min", restore_best_weights=True)
ckp_cbk = tf.keras.callbacks.ModelCheckpoint('models/squareRoot/3_model', save_best_only=True, monitor='loss')
tb_cbk = tf.keras.callbacks.TensorBoard(log_dir='models/squareRoot/3_model/logs')

model2.fit(x=X_train,  # input data
           y=y_train,  # target data
           batch_size=100,  # Number of samples per gradient update.
           epochs=100000, #Number of epochs to train the model.
           verbose=1,
           validation_data=(X_valid, y_valid),
           shuffle=True,
           validation_freq=3,
           callbacks=[ckp_cbk, tb_cbk]
           )


print("")
print('-------------------- Model Summary --------------------')
model2.summary()  # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
for layer in model2.layers:
    print("Layer: ", layer.name)  # print layer name
    print("  --Kernels (Weights): ", layer.get_weights()[0])  # kernels (weights)
    print("  --Biases: ", layer.get_weights()[1])  # biases

model2.save('models/squareRoot/squareRoot_model.h5')

# pred_labels_tr = model2.predict(X_train)
# pred_labels_te = model2.predict(X_test)
# print("")
# print('---------- Evaluation on Training Data ----------')
# # print(classification_report(y_train, pred_labels_tr))
# # print_data(y_train, pred_labels_tr)
# print_data(y_train, pred_labels_tr)
# print("")
#
# print('---------- Evaluation on Test Data ----------')
# # print(classification_report(y_test, pred_labels_te))
# print_data(y_test, pred_labels_te)
# print("")


