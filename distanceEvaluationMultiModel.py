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

import matplotlib.pyplot as plt

import sys
import os


def print_data(expected, predicted):
    print([[expected[i], predicted[i], abs(expected[i] - predicted[i])] for i in range(len(expected))])


# main_dir = os.path.dirname(sys.path[0])

dimension = 3
samples = 1000000
labels = ['a{}'.format(i) for i in range(1, dimension + 1)] +\
         ['b{}'.format(i) for i in range(1, dimension + 1)]

# model2 = Sequential()
# model2.add(Input(shape=(dimension,)))
# model2.add(Dense(units=4*dimension, activation='relu', input_dim=dimension))
# model2.add(Dense(units=16*dimension, activation='relu'))
# model2.add(Dense(units=4*dimension, activation='relu'))
# model2.add(Dense(units=1, activation='relu'))
# model2.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError(), 'Precision'])

model_subtraction = keras.models.load_model('models/subtraction/1_model/')
model_multiplication = keras.models.load_model('models/multiplication/3_model')
model_square_root = keras.models.load_model('models/squareRoot/3_model')

df = pd.read_csv('csv/test/d3n1000.csv')
X = df[labels]
y = df['distance'].values


X_test = np.array(X)
y_test = np.array(y)



subtracted_layer = np.array([[model_subtraction.predict(np.array([[row[i], row[i+len(row)//2]]])).item() for i in range(len(row)//2)] for row in X_test])
print("DONE SUBTRACTION")
multiplied_layer = [[model_multiplication.predict(np.array([[num, num]])).item() for num in row] for row in subtracted_layer]
print("DONE MULTIPLICATION")
sums = [sum(row)/len(row) for row in multiplied_layer]
print("DONE SUMMING")
square_root_layer_predicted = [model_square_root.predict(np.array([[s]])).item()*(dimension**0.5) for s in sums]
print("DONE SQUAREROOTING")
square_root_layer_calculated = [(s**0.5)*(dimension**0.5) for s in sums]

absolute_error_sqm = [abs(y_test[i]-square_root_layer_predicted[i]) for i in range(len(square_root_layer_predicted))]
relative_error_sqm = [absolute_error_sqm[i]/y_test[i]*100 for i in range(len(absolute_error_sqm))]

absolute_error = [abs(y_test[i]-square_root_layer_calculated[i]) for i in range(len(square_root_layer_calculated))]
relative_error = [absolute_error[i]/y_test[i]*100 for i in range(len(absolute_error))]

plt.plot(range(len(relative_error_sqm)), relative_error_sqm)
plt.title('Relative error using model for square root')
plt.yscale('log')
plt.grid(True)
plt.show()
plt.plot(range(len(absolute_error_sqm)), absolute_error_sqm)
plt.title('Absolute error using model for square root')
plt.yscale('log')
plt.grid(True)
plt.show()
plt.plot(range(len(relative_error)), relative_error)
plt.title('Relative error')
plt.yscale('log')
plt.grid(True)
plt.show()
plt.plot( range(len(absolute_error)), absolute_error)
plt.title('Absolute error')
plt.yscale('log')
plt.grid(True)
plt.show()

f = open("evaluation/seq_w_sqrmd/averages.txt", "w")
f.write("Squaring with model:\n")
f.write("Avg absolute error: {}\n".format(sum(absolute_error_sqm)/len(absolute_error_sqm)))
f.write("Avg relative error: {}\n".format(sum(relative_error_sqm)/len(relative_error_sqm)))
f.close()

f = open("evaluation/seq_normal/averages.txt", "w")
f.write("Squaring normally:\n")
f.write("Avg absolute error: {}\n".format(sum(absolute_error)/len(absolute_error)))
f.write("Avg relative error: {}\n".format(sum(relative_error)/len(relative_error)))
f.close()
