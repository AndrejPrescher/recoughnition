#libs

import wave
from scipy.io import wavfile
import os as os
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import IPython.display as ipd
import sys
import resampy
import librosa
from tensorflow.keras import layers


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Preprocessing 
#import sounddevice as sd
from recoughnition import preprocessing as pp

# print the TF version
print(tf.__version__)

# define parameters

lengthFrame = 1
model_vggish = hub.load('https://tfhub.dev/google/vggish/1')

#model mit conv1d
from tensorflow.keras import Input, layers

X_train_shaped = np.load('./X_train_prep.npy')
train_label = np.load('./y_train_prep.npy')

# Parameters
lr = 0.0001
batch_size = 20
drop_out_rate = 0.15
input_shape = (128,1)
act = 'relu'
kernel=2

#Conv1D Model
input_tensor = Input(shape=(input_shape))

x = layers.Conv1D(8, kernel, padding='valid', activation=act, strides=1)(input_tensor)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(16, kernel, padding='valid', activation=act, strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(32, kernel, padding='valid', activation=act, strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(64, kernel, padding='valid', activation=act, strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(128, kernel, padding='valid', activation=act, strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(256, kernel, padding='valid', activation=act, strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
output_tensor = layers.Dense(2, activation='sigmoid')(x)

model = tf.keras.Model(input_tensor, output_tensor)

model.compile(loss=keras.losses.binary_crossentropy,
             optimizer=keras.optimizers.Adam(lr=lr,),
             metrics=['accuracy'],)

model.summary()

history = model.fit(X_train_shaped, train_label,
          batch_size=batch_size, 
          epochs=400,
          verbose=1,
          shuffle = True)

model.save()
