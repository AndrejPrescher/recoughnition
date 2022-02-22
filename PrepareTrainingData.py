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

X_train = np.load('./X_train.npy')
y_train = np.load('./y_train.npy')
'''
h = np.isnan(X_train)
h = np.where(h==True)
h = h[0]
h = h[1::len(X_train.T)]

X_train_new = np.delete(X_train, h, axis=0)
y_train_new = np.delete(y_train,h)

print(np.shape(X_train_new), np.shape(y_train_new))
X_train = X_train_new
y_train = y_train_new
'''
#explore data
print('X_train: ', X_train.shape)

print('y_train: ', y_train.shape)

#further pre for conv1d

#For Conv1D add Channel

if X_train.ndim < 4:
    X_train_shaped = np.expand_dims(X_train, axis=2)
    
print(X_train_shaped.shape)

#Make Label data 'class num' -> 'One hot vector'
train_label = keras.utils.to_categorical(y_train, 2)
train_label=np.fliplr(train_label)
