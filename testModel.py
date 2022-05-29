from tensorflow import keras

print('Tensorflow/Keras: %s' % keras.__version__)
from keras.models import load_model

import pandas as pd
print('pandas: %s' % pd.__version__)

import numpy as np
print('numpy: %s' % np.__version__)

import sklearn
print('sklearn: %s' % sklearn.__version__)

import plotly
print('plotly: %s' % plotly.__version__)

import sys
import os


def print_data(expected, predicted):
    print([[expected[i], predicted[i][0], abs(expected[i] - predicted[i])[0]] for i in range(len(expected))])


main_dir = os.path.dirname(sys.path[0])

dimension = 1
samples = 1000
zeros = 1
#df = pd.read_csv('csv/test/d{}n{}.csv'.format(dimension, samples))
df = pd.read_csv('csv/training/multiplication/multi_10000.csv')

labels = ['a{}'.format(i) for i in range(1, dimension + 1)] + \
         ['b{}'.format(i) for i in range(1, dimension + 1)]

X = df[labels]
y = df['distance'].values

X_test = np.array(X)
y_test = np.array(y)

model2 = load_model('models/multiplication/1_model')

pred_labels_te = model2.predict(X_test)

print_data(y_test, pred_labels_te)
