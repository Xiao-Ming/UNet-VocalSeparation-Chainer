#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:49:54 2017

@author: wuyiming
"""

import numpy as np
from librosa.util import find_files
from librosa.core import load
import os.path
import util


PATH_iKala = "iKala/Wavfile"

audiolist = find_files(PATH_iKala, ext="wav")

for audiofile in audiolist:
    fname = os.path.split(audiofile)[-1]
    print("Processing: %s" % fname)
    y, _ = load(audiofile, sr=None, mono=False)
    inst = y[0, :]
    mix = y[0, :]+y[1, :]
    vocal = y[1, :]
    util.SaveSpectrogram(mix, vocal, inst, fname)


print("Constructing random mix...")

rand = np.random.randint(len(audiolist), size=40)

count = 1

for i in range(0, 40, 2):
    y1, _ = load(audiolist[rand[i]], sr=None, mono=False)
    y2, _ = load(audiolist[rand[i+1]], sr=None, mono=False)
    minlen = min([y1.shape[1], y2.shape[1]])
    inst = y1[0, :minlen]
    vocal = y2[1, :minlen]
    mix = inst+vocal

    fname = "ikala_random%02d" % count
    util.SaveSpectrogram(mix, vocal, inst, fname)
    count += 1
    print("Saved %s.npz" % fname)
