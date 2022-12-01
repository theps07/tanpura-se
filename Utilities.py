#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of functions to facilitate the analysis of prominent harmonics in 
a tanpura signal. 

@author: Puru Samal
"""
# LIBRARIES
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy.signal import argrelextrema
import numpy as np
from scipy.signal import butter, lfilter

# A function to wrap angle to [-pi, pi]


def normalizeAngle(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


# The Pitch Detection Algorithm


def fft_findPitch(sig, Fs, Nfft=8192, R=1, i=0, n=6):
    '''sig:  Input signal of length Nfft+R
       Fs:   Sampling Rate
       Nfft: FFT Length
       R:    FFT hop size
       i:   index
       n:   Number of partials by descending order of magnitude'''

    # Hann window of same length as FFT
    win = signal.windows.hann(Nfft)
    df = Fs/Nfft                           # Freq resolution
    dt = R/Fs                                # time diffrence between ffts

    sig1 = sig[0+i:Nfft+i]                   # 1st Block
    sig2 = sig[0+R+i:Nfft+R+i]               # 2nd block with hop size R
    sig1, sig2 = sig1*win, sig2*win          # Both blocks are windowed
    fft1, fft2 = rfft(sig1), rfft(sig2)      # fft on both signal blocks

    # Log of magnitude spectrum of both sig blocks
    X1, X2 = 20*np.log10(abs(fft1)), 20*np.log10(abs(fft2))
    # phase spectrum of both sig blocks
    Phi1, Phi2 = np.angle(fft1), np.angle(fft2)
    # bin frequencies for given FFT length
    binfreqs = rfftfreq(int(Nfft), 1/Fs)

    idx = argrelextrema(X1, np.greater)[0]   # index of bin with local maximas
    # Sort magnitude values in descensing order
    desX = np.sort(X1[idx])[::-1]

    # List of bin index corresponding to descending order of mag values
    k = [np.where(X1 == desX[i])[0][0] for i in range(n)]
    # List of estimated frequencies
    # est_freqs = [index * df for index in k]

    cor_f = []
    for item in k:
        phi1 = Phi1[item]
        phi2_t = phi1 + (2 * np.pi) / (Nfft * item * R)
        phi2 = Phi2[item]
        phi2_err = normalizeAngle(phi2-phi2_t)
        phi2_unwrap = phi2_t + phi2_err
        dphi = phi2_unwrap - phi1
        Fcorr = dphi/(2 * np.pi * dt)
        cor_f.append(Fcorr)

    return cor_f  # Return list of corrected frequencies


# Pitch Envelope algorithm


def fft_findPitchEnv(sig, Fs, Nfft=8192, R=1, i=0, n=6):
    '''sig:  Input signal of length Nfft+R
       Fs:   Sampling Rate
       Nfft: FFT Length
       R:    FFT hop size
       i:   index
       n:   Number of partials by descending order of magnitude'''
    win = signal.windows.hann(Nfft)  # Hann window of same length as FFT
    df = Fs/Nfft  # Freq resolution
    dt = R/Fs  # time diffrence between ffts

    sig1 = sig[0+i:Nfft+i]  # 1st Block
    sig2 = sig[0+R+i:Nfft+R+i]  # 2nd block with hop size R
    sig1, sig2 = sig1*win, sig2*win  # Both blocks are windowes
    fft1 = rfft(sig1)  # fft on both signal blocks

    X1 = 20*np.log10(abs(fft1))  # Log of magnitude spectrum of both sig blocks

    idx = argrelextrema(X1, np.greater)[0]  # index of bin with local maximas
    desX = np.sort(X1[idx])[::-1]  # Sort magnitude values in descensing order

    # List of bin index corresponding to descending order of mag values
    k = [np.where(X1 == desX[i])[0][0] for i in range(n)]
    est_freqs = [index * df for index in k]  # List of estimated frequencies
    ampEnvs = [desX[i] for i in range(n)]
    return ampEnvs


# Array quantization


def quantizeArray(x, quant):
    quant_x = []

    for value in x:
        idx = min(range(len(quant)), key=lambda i: abs(quant[i]-value))

        if value > max(quant):
            quantized = round(value // quant[idx]) * quant[idx]
        else:
            quantized = quant[idx]
        quant_x.append(quantized)

    return quant_x


# A lowpass filter


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Function to scale data to wav files


def scale2wav(data, inMin, inMax):
    normData = np.array([((2*(item-inMin)/(inMax-inMin))-1) for item in data])
    return normData
