'''
An example for multilayer perceptron regression on Life Expectancy
'''
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, MissingIndicator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils import *

# cleansed_data_path = '/content/drive/MyDrive/Life_Expectancy_Data_processed.csv'

filepath = 'data/Life_Expectancy_Data_processed .csv'
model_path = 'models/MLP/model.joblib'
test_size = 0.2

df = load_dataset(filepath)
df = df.iloc[:, 1:]  # Remove Country name

# Prepare training set and testing set

# features = df.copy()
# labels = features.pop('Life expectancy')
#
# # Remove features as desired
# features.drop(['Infant Mort', 'Adult Mort'], axis=1)

imputer = KNNImputer(missing_values=np.nan, n_neighbors=5)
imputed_data = imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed_data, columns=df.columns)

features = df_imputed.copy()
labels = features.pop('Life expectancy')

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, random_state=0, test_size=0.2)
train_features.shape, test_features.shape, train_labels.shape, test_labels.shape


# normalization
def scaleFeature(train_array, test_array):
    train_rows = train_array.shape[0]
    test_rows = test_array.shape[0]

    ### START CODE HERE ###
    array_min = np.min(train_array, axis=0)
    array_range = np.max(train_array, axis=0) - array_min
    norm_train_array = (train_array - array_min) / array_range
    norm_test_array = (test_array - array_min) / array_range
    ### END CODE HERE ###

    return norm_train_array, norm_test_array


train_features_nor, test_features_nor = scaleFeature(train_features, test_features)
train_labels_nor, test_labels_nor = scaleFeature(train_labels, test_labels)

# transform to array
train_x = np.array(train_features_nor)
train_y = np.array(train_labels_nor).reshape(len(train_labels_nor), 1)
test_x = np.array(test_features_nor)
test_y = np.array(test_labels_nor).reshape(len(test_labels_nor), 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(5, input_shape=(17,), activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mse')
history = model.fit(train_x, train_y, epochs=1000, validation_data=(test_x, test_y))

plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()

array_min = np.min(train_labels, axis=0)
array_range = np.max(train_labels, axis=0) - array_min

array_min

p_y_norm = model.predict(test_x)



def plot_results(labels, preds):
    ave, ax = plt.subplots()
    ax.set_title('Actual vs Fitted Life Expectancy')
    ax.set_xlabel('Actual (Years)')
    ax.set_ylabel('Fitted (Years)')
    ax.scatter(labels, preds)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    err, ax2 = plt.subplots()
    ax2.set_title('Error of Prediction')
    ax2.set_xlabel('Error (years)')
    ax2.set_ylabel('Density')
    ax2.hist(preds - labels, bins=50, density=True)


plot_results(test_y, p_y_norm)
# plot_results(np.array(test_labels).reshape(len(test_labels),1), p_y)
