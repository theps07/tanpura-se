#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The generation of control signals from the prominent harmonics in a tanpura signal.
The control signels are encoded in WAV format and stored in the folder labeled
'controlValues'. 

@author: Puru Samal
"""
# LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from Utilities import fft_findPitch, fft_findPitchEnv, scale2wav, butter_lowpass_filter
from scipy.io.wavfile import write
import os


# Parameter Specification
Fs, sig = wavfile.read('e_tanpuraEdited.wav')
sig = butter_lowpass_filter(sig, cutoff=1000, fs=Fs)
Nfft = 8192  # FFT length
R = 1  # FFT hop size for pitch estimation
n = 7  # number of pitches to be tracked
K = 128  # Hop size for time resolution of pitch estimation

# Array to hold n-max pitch values for each block
sigSlice = int(len(sig) - Nfft - R)  # original signal
env = [fft_findPitch(sig, Fs, Nfft, R, i, n) for i in range(0, sigSlice, K)]

# Array to hold amplitude envelopes of n-max pitch values for each block
envAmp = [fft_findPitchEnv(sig, Fs, Nfft, R, i, n)
          for i in range(0, sigSlice, K)]

# Creating a dictionary for each individual frequency spectral env. Key Values = ['f0'...'f(n)']
freqs = {}
keys = ['f{0}'.format(i) for i in range(n)]
for item in keys:
    freqs[item] = np.array([freq[keys.index(item)] for freq in env])

# Creating a dictionary for each individual amplitude spectral env. Key Values = ['env_f0'...'env_f(n)']
freqEnvs = {}
keysEnv = ['env_f{0}'.format(i) for i in range(n)]
for item in keysEnv:
    freqEnvs[item] = np.array([freqEnvs[keysEnv.index(item)]
                              for freqEnvs in envAmp])

# Frequency normalization (-1,1):
minFreq = 65.0
maxFreq = 1000.0
normFreqs = {k: scale2wav(v, inMin=minFreq, inMax=maxFreq)
             for (k, v) in freqs.items()}

# Envelope normalization (-1..1)
normFreqEnvs = {k: np.power(10, np.divide(v, 20))
                for (k, v) in freqEnvs.items()}  # removing log
minEnv = min([min(normFreqEnvs[key]) for key in normFreqEnvs.keys()])
maxEnv = max([max(normFreqEnvs[key]) for key in normFreqEnvs.keys()])
normFreqEnvs = {k: scale2wav(v, inMin=minEnv, inMax=maxEnv)
                for (k, v) in normFreqEnvs.items()}

# Time domain generation for plotting
t = np.linspace(0, len(env)-1, num=len(env))


# Plotting spectral envelopes and amplitude envelopes
fig_1, axes_1 = plt.subplots(figsize=(3*n, 2*n), dpi=200, nrows=n, ncols=2)
plt.tight_layout(pad=3)

for item in keys:
    axes_1[keys.index(item)][0].set_xlabel('t')
    axes_1[keys.index(item)][0].set_ylabel('f')
    axes_1[keys.index(item)][0].plot(
        t, normFreqs[item], 'b-', linewidth=1, label=item)
    axes_1[keys.index(item)][0].grid()
    axes_1[keys.index(item)][0].legend(prop={"size": 6})

for item in keysEnv:
    axes_1[keysEnv.index(item)][1].set_xlabel('t')
    axes_1[keysEnv.index(item)][1].set_ylabel('amp')
    axes_1[keysEnv.index(item)][1].plot(
        t, normFreqEnvs[item], 'g-', linewidth=1, label=item)
    axes_1[keysEnv.index(item)][1].grid()
    axes_1[keysEnv.index(item)][1].legend(prop={"size": 6})


# Write control values to wav file
Fs_ctrl = (int((len(normFreqs['f0'])/len(sig))*Fs))-1
folder = os.path.abspath('controlValues')

for key in normFreqs.keys():
    path = os.path.join(folder, "{0}Ctrl.wav".format(key))
    write(path, Fs_ctrl, normFreqs[key].astype(np.float32))

for key in normFreqEnvs.keys():
    path = os.path.join(folder, "{0}Ctrl.wav".format(key))
    write(path, Fs_ctrl, normFreqEnvs[key].astype(np.float32))
