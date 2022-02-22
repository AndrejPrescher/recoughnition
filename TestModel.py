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
from recoughnition import vggish_params

print(vggish_params.SAMPLE_RATE)

def sumToMono(audioData: np.ndarray):
    if len(audioData.shape) == 2:
        data = np.sum(audioData, axis=1)
        return data
    else:
        return audioData

def normaliseNdarray(audioData: np.ndarray):
    # get biggest absolute value for normalisation
    absmax = max(audioData, key=abs)
    audioData = np.divide(audioData, absmax)  # normalise values
    return audioData

def processMusic_Testfile(filename, frameLength):
    minusThreeDb = 0.7079457843841379
    sr, data = wavfile.read(filename)
    data = sumToMono(data)
    data = normaliseNdarray(data)
    data = np.int16(data/np.max(np.abs(data)) * 32767)
    #if sr != vggish_params.SAMPLE_RATE:
    #        data = resampy.resample(data, sr, vggish_params.SAMPLE_RATE)
    data = data*minusThreeDb
    frame_numbers = int(np.ceil((len(data)/vggish_params.SAMPLE_RATE)/frameLength))
    data_new = np.zeros((frame_numbers,int(frameLength*vggish_params.SAMPLE_RATE)))
    for frame_number in range(frame_numbers):
        start = int(frame_number*frameLength*vggish_params.SAMPLE_RATE)
        ende = int((frame_number+1)*frameLength*vggish_params.SAMPLE_RATE)
        if frame_number != frame_numbers-1:
            data_new[frame_number] = data[start:ende]
        else:
            data_new[frame_number] = np.concatenate((data[start:-1],np.zeros(int(frameLength*vggish_params.SAMPLE_RATE-len(data[start:-1])))))
    data_new = np.asarray(data_new)
    return vggish_params.SAMPLE_RATE, data_new

def getFrames_Testfile(filename, length: float):
    global frameLength
    frameLength = length
    srMusic, dataMusic = processMusic_Testfile(filename, frameLength)
    return srMusic, dataMusic

samplerate, file_check = getFrames_Testfile('/work/test_music/testfile_new-002.wav', lengthFrame)

print(file_check)

embeddings_file = np.zeros((len(file_check),128))

for frame_number in range(len(file_check)):
    embeddings_file[frame_number] = model_vggish(file_check[frame_number])
    
timecode=np.zeros(embeddings_file.shape[0])# array in sekunden

for i in range (embeddings_file.shape[0]):

    if len(embeddings_file[i])==128:
        
        y_prob = model.predict(embeddings_file[i:i+1])
    
        if y_prob[0][0] >0.98:
            #timecode[i]=1
            timecode[i] = y_prob[0][0]
        else:
            #timecode[i]=0
            timecode[i] = y_prob[0][0]

frame_axis = np.linspace(0,len(file_check)*frameLength/2.75625,len(timecode))
hust_axis = timecode

plt.plot(frame_axis,hust_axis)
plt.xlabel('Zeit in s')
plt.ylabel('Wahrscheinlichkeit Huster')
plt.legend()
plt.show()
