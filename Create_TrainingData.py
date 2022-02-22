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

anzahl = 1000

chunk_mit= np.zeros((anzahl,128))

for i in range(anzahl):
    
    samplerate, waveform = pp.getFrame(True, lengthFrame)
    mv = model_vggish(waveform)
    chunk_mit[i]= mv
    
Label_True=np.ones(chunk_mit.shape[0])

chunk_ohne=np.zeros((anzahl,128))

for i in range(anzahl):
    
    samplerate, waveform = pp.getFrame(False, lengthFrame)
    mv = model_vggish(waveform)
    chunk_ohne[i]=mv
    
Label_False=np.zeros(chunk_ohne.shape[0])

X_train, y_train=np.concatenate((chunk_mit,chunk_ohne)),np.concatenate((Label_True,Label_False)) 

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
