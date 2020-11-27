'''
An example for multilayer perceptron regression on Life Expectancy
'''
####This framework uses Neural Network to achieve Linear Regression. The functions included and their explanations are as follows:
# class Load_for_nn -- Load data sets, extract test sets, complete normalization, and convert to arrays
# plot_results -- Draw plot compairing the fitted values and true values and error histogram
# bulit_NN -- Use tf.keras to built Neural Network, define the structure of  Neural Network and training parameters

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, MissingIndicator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils import *

# cleaned_data_path--The path where you store the cleaned data
# label_names_list--A list of the names of the output variables
# test_size--The proportion of the test set to the data set
# dropout_size--In addition to the input layer, the proportion of neurons abandoned by each layer of neural network.Can use to reduce the risk of overfitting.
# layer_number--The number of hidden and output layers
# neurons_number--Number of neurons per layer
# learning_rate--learning rate of training
# epoches--Number of training
cleaned_data_path = 'data/Life_Expectancy_imputed_Data.csv'
label_names_list = ['Life expectancy']
test_size = 0.2
droupout_size = 0
layer_number = 2  # need>1
neurons_number = 5
learning_rate = 0.00001
epoches = 1000


class Load_for_nn:

    def __init__(self, cleaned_data_path):

        self.cleaned_data_path = cleaned_data_path

    def load_dataset(self, label_names_list, test_size):
        cleansed_dataset = pd.read_csv(self.cleaned_data_path, header=0, na_values='?', comment='\t', sep=',',
                                       skipinitialspace=True)
        dataset_x = cleansed_dataset
        for name in label_names_list:
            dataset_y = dataset_x.pop(name)
        train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, random_state=0, test_size=test_size)

        train_x_nor, test_x_nor = self.scaleFeature(train_x, test_x)
        train_y_nor, test_y_nor = self.scaleFeature(train_y, test_y)

        train_x_array = self.into_array(train_x_nor)
        test_x_array = self.into_array(test_x_nor)
        train_y_array = self.into_array(train_y_nor)
        test_y_array = self.into_array(test_y_nor)

        return train_x_array, test_x_array, train_y_array, test_y_array, train_x, train_y

    def into_array(self, df):
        if len(df.shape) == 1:
            df_array = np.array(df).reshape(df.shape[0], 1)
        else:
            df_array = np.array(df)
        return df_array

    def scaleFeature(self, train_array, test_array):

        train_rows = train_array.shape[0]
        test_rows = test_array.shape[0]

        array_min = np.min(train_array, axis=0)
        array_range = np.max(train_array, axis=0) - array_min
        norm_train_array = np.array((train_array - array_min) / array_range)
        norm_test_array = np.array((test_array - array_min) / array_range)

        return norm_train_array, norm_test_array


def bulit_NN(droupout_size, layer_number, neurons_number, learning_rate, input_num, output_num):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neurons_number, input_shape=(input_num,), activation='relu'))
    if layer_number <= 1:
        print('layer_number must be bigger than 1')
    else:
        for i in range(1, layer_number):
            model.add(tf.keras.layers.Dense(neurons_number, activation='relu'))
            model.add(tf.keras.layers.Dropout(droupout_size))
        model.add(tf.keras.layers.Dense(output_num))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])

    return model


Load_for_nn = Load_for_nn(cleaned_data_path)
train_x, test_x, train_y, test_y, original_train_x, original_trin_y = Load_for_nn.load_dataset(label_names_list,
                                                                                               test_size)
model = bulit_NN(droupout_size, layer_number, neurons_number, learning_rate, train_x.shape[1], train_y.shape[1])
# model.summary()
history = model.fit(train_x, train_y, epochs=epoches, validation_data=(test_x, test_y))

plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.savefig(join('outputs/MLP', 'loss.png'))

array_min = np.min(original_trin_y, axis=0)
array_range = np.max(original_trin_y, axis=0) - array_min
p_y_norm = model.predict(test_x)
p_y = p_y_norm * array_range + array_min
test_y = test_y * array_range + array_min
plot_results(test_y, p_y, 'outputs/MLP')
