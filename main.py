from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from sklearn.preprocessing import MinMaxScaler

# load the data from the dataset
(training_images, training_labels), (test_images, test_labels) = ks.datasets.fashion_mnist.load_data()
test_labels = test_labels.astype(int)
batch_size = len(training_images)
scaler = MinMaxScaler()
print('Training Images Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Images Dataset Labels: {}'.format(len(training_labels)))
print('Test Images Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Images Dataset Labels: {}'.format(len(test_labels)))

# normalize the data for better training
# training_images = scaler.fit_transform(training_images.astype(np.float64))
# test_images = scaler.fit_transform(test_images.astype(np.float64))
training_images = training_images / 255.0
test_images = test_images / 255.0

# build the model
input_data_shape = (28, 28)
hidden_activation_function = 'relu'
output_activation_function = 'softmax'
nn_model = ks.models.Sequential()
nn_model.add(ks.layers.Flatten(input_shape=input_data_shape, name='Input_layer'))
nn_model.add(ks.layers.Dense(32, activation=hidden_activation_function, name='Hidden_layer'))
nn_model.add(ks.layers.Dense(10, activation=output_activation_function, name='Output_layer'))
nn_model.summary()

# train the model
optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
metric = ['accuracy']
nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)
nn_model.fit(training_images, training_labels, epochs=20)

# evaluate the model
training_loss, training_accuracy = nn_model.evaluate(training_images, training_labels)
print('Training Data Accuracy {}'.format(round(float(training_accuracy),2)))

# test the model
test_loss, test_accuracy = nn_model.evaluate(test_images,test_labels)
print('Test Data Accuracy {}'.format(round(float(test_accuracy),2)))
