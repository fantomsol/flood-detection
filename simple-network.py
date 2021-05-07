# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:02:33 2021

@author: david
"""

from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

def makeModel():
  inputs = keras.Input(shape=(256, 256, 2), dtype='float32')
  # downsample
  x = layers.ZeroPadding2D()(inputs)
  x = layers.Conv2D(32, (3,3), strides=(2,2), activation="relu")(x)
  x = layers.MaxPooling2D((2,2))(x)
  x = layers.Dropout(0.2)(x)
  x = layers.BatchNormalization()(x)
  x = layers.ZeroPadding2D()(x)
  x = layers.Conv2D(64, (3,3), strides=(2,2), activation="relu")(x)
  x = layers.MaxPooling2D((2,2))(x)
  x = layers.Dropout(0.2)(x)
  x = layers.BatchNormalization()(x)
  x = layers.ZeroPadding2D()(x)
  x = layers.Conv2D(64, (3,3), strides=(2,2), activation="linear")(x)
  
  # upsample
  x = layers.Conv2DTranspose(64,(3, 3), strides=2, activation="relu", padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2DTranspose(64,(3, 3), strides=2, activation="relu", padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2DTranspose(64,(3, 3), strides=2, activation="relu", padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2DTranspose(64,(3, 3), strides=2, activation="relu", padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2DTranspose(32,(3, 3), strides=2, activation="relu", padding="same")(x)
  x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
  out = x
  
  return Model(inputs, out)

keras.backend.clear_session()

dataset = np.load("data.npz")
data_points = int(1e4)

vv = dataset['vv']
vh = dataset['vh']
flood_label = dataset['flood_label']

vv = np.float32(vv[0:data_points, ::] / 256)
vh = np.float32(vh[0:data_points, ::] / 256)
flood_label = np.float32(flood_label[0:data_points, ::] / 256)

x = np.concatenate((vv, vh), axis=-1)
y = flood_label

model = makeModel()
model.summary()

model.compile(loss="mse", optimizer="adam")
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,restore_best_weights=True)
                                              
history = model.fit(x, y, batch_size = 64, epochs = 100, callbacks = [callback], validation_split = 0.1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()