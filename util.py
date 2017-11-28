#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:47:06 2017

@author: wuyiming
"""


from librosa.util import find_files
from librosa.core import stft, load, istft, resample
from librosa.output import write_wav
import network
import const as C
import numpy as np
from chainer import config
import os.path


def SaveSpectrogram(y_mix, y_vocal, y_inst, fname, original_sr=44100):
    y_mix = resample(y_mix, original_sr, C.SR)
    y_vocal = resample(y_vocal, original_sr, C.SR)
    y_inst = resample(y_inst, original_sr, C.SR)

    S_mix = np.abs(
        stft(y_mix, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_vocal = np.abs(
        stft(y_vocal, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_inst = np.abs(
        stft(y_inst, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)

    norm = S_mix.max()
    S_mix /= norm
    S_vocal /= norm
    S_inst /= norm

    np.savez(os.path.join(C.PATH_FFT, fname+".npz"),
             mix=S_mix, vocal=S_vocal, inst=S_inst)


def LoadDataset(target="vocal"):
    filelist_fft = find_files(C.PATH_FFT, ext="npz")[:200]
    Xlist = []
    Ylist = []
    for file_fft in filelist_fft:
        dat = np.load(file_fft)
        Xlist.append(dat["mix"])
        if target == "vocal":
            assert(dat["mix"].shape == dat["vocal"].shape)
            Ylist.append(dat["vocal"])
        else:
            assert(dat["mix"].shape == dat["inst"].shape)
            Ylist.append(dat["inst"])
    return Xlist, Ylist


def LoadAudio(fname):
    y, sr = load(fname, sr=C.SR)
    spec = stft(y, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))
    return mag, phase


def SaveAudio(fname, mag, phase):
    y = istft(mag*phase, hop_length=C.H, win_length=C.FFT_SIZE)
    write_wav(fname, y, C.SR, norm=True)


def ComputeMask(input_mag, unet_model="unet.model", hard=True):
    unet = network.UNet()
    unet.load(unet_model)
    config.train = False
    config.enable_backprop = False
    mask = unet(input_mag[np.newaxis, np.newaxis, 1:, :]).data[0, 0, :, :]
    mask = np.vstack((np.zeros(mask.shape[1], dtype="float32"), mask))
    if hard:
        hard_mask = np.zeros(mask.shape, dtype="float32")
        hard_mask[mask > 0.5] = 1
        return hard_mask
    else:
        return mask
