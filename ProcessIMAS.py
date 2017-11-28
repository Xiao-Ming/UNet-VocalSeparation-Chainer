#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:55:00 2017

@author: wuyiming
"""


from librosa.core import load, resample, stft
import numpy as np
from librosa.util import find_files
from librosa.effects import pitch_shift, time_stretch
import const as C
import os

PATH_IMAS = "IMASTracks"

filelist = find_files(PATH_IMAS, ext="wav")


def SavespecArg(y_mix, y_inst, fname, shift, stretch):
    Savespec(y_mix, y_inst, fname)
    for sh in shift:
        y_mix_shift = pitch_shift(y_mix, C.SR, sh)
        y_inst_shift = pitch_shift(y_inst, C.SR, sh)
        Savespec(y_mix_shift, y_inst_shift, "%s_shift%d" % (fname, sh))

        y_mix_shift = pitch_shift(y_mix, C.SR, -sh)
        y_inst_shift = pitch_shift(y_inst, C.SR, -sh)
        Savespec(y_mix_shift, y_inst_shift, "%s_shift-%d" % (fname, sh))

    for st in stretch:
        y_mix_stretch = time_stretch(y_mix, st)
        y_inst_stretch = time_stretch(y_inst, st)
        Savespec(y_mix_stretch, y_inst_stretch,
                 "%s_stretch%d" % (fname, int(st * 10)))


def Savespec(y_mix, y_inst, fname):
    S_mix = np.abs(
        stft(y_mix, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_inst = np.abs(
        stft(y_inst, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_vocal = np.maximum(0, S_mix - S_inst)
    # y_vocal = istft(S_vocal*phase, hop_length=C.H, win_length=C.FFT_SIZE)
    # write_wav(os.path.join("Audiocheck", fname+".wav"), y_vocal, C.SR)
    norm = S_mix.max()
    S_mix /= norm
    S_inst /= norm
    S_vocal /= norm
    np.savez(os.path.join(C.PATH_FFT, fname+".npz"),
             vocal=S_vocal, mix=S_mix, inst=S_inst)


for i in range(0, len(filelist), 2):
    print("Processing:" + filelist[i])
    print("Processing:" + filelist[i+1])
    y_mix, _ = load(filelist[i], sr=None)
    y_inst, sr = load(filelist[i+1], sr=None)

    y_mix = y_mix[np.nonzero(y_mix)[0][0]:]
    y_inst = y_inst[np.nonzero(y_inst)[0][0]:]

    minlength = min([y_mix.size, y_inst.size])
    y_mix = resample(y_mix[:minlength], 44100, C.SR)
    y_inst = resample(y_inst[:minlength], 44100, C.SR)
    fname = filelist[i].split("/")[-1].split(".")[0]
    Savespec(y_mix, y_inst, fname)
