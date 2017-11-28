#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 03:45:49 2017

@author: wuyiming
"""

import numpy as np
from librosa.core import load
import util
import os

PATH_DSD_SOURCE = ["DSD100/Sources/Dev", "DSD100/Sources/Test"]
PATH_DSD_MIXTURE = ["DSD100/Mixtures/Dev", "DSD100/Mixtures/Test"]

FILE_MIX = "mixture.wav"
FILE_BASS = "bass.wav"
FILE_DRUMS = "drums.wav"
FILE_OTHER = "other.wav"
FILE_VOCAL = "vocals.wav"


list_source_dir = [os.path.join(PATH_DSD_SOURCE[0], f)
                   for f in os.listdir(PATH_DSD_SOURCE[0])]
list_source_dir.extend([os.path.join(PATH_DSD_SOURCE[1], f)
                        for f in os.listdir(PATH_DSD_SOURCE[1])])
list_source_dir = sorted(list_source_dir)

list_mix_dir = [os.path.join(PATH_DSD_MIXTURE[0], f)
                for f in os.listdir(PATH_DSD_MIXTURE[0])]
list_mix_dir.extend([os.path.join(PATH_DSD_MIXTURE[1], f)
                     for f in os.listdir(PATH_DSD_MIXTURE[1])])
list_mix_dir = sorted(list_mix_dir)


for mix_dir, source_dir in zip(list_mix_dir,  list_source_dir):
    assert(mix_dir.split("/")[-1] == source_dir.split("/")[-1])
    fname = mix_dir.split("/")[-1]
    print("Processing: " + fname)
    y_mix, sr = load(os.path.join(mix_dir, FILE_MIX), sr=None)
    y_vocal, _ = load(os.path.join(source_dir, FILE_VOCAL), sr=None)
    y_inst = sum([load(os.path.join(source_dir, f), sr=None)[0]
                  for f in [FILE_DRUMS, FILE_BASS, FILE_OTHER]])

    assert(y_mix.shape == y_vocal.shape)
    assert(y_mix.shape == y_inst.shape)

    util.SaveSpectrogram(y_mix, y_vocal, y_inst, fname)


rand_voc = np.random.randint(100, size=50)
rand_bass = np.random.randint(100, size=50)
rand_drums = np.random.randint(100, size=50)
rand_other = np.random.randint(100, size=50)

count = 1
print("Generating random mix...")
for i_voc, i_bass, i_drums, i_other in \
        zip(rand_voc, rand_bass, rand_drums, rand_other):
    y_vocal, _ = load(os.path.join(list_source_dir[i_voc], FILE_VOCAL),
                      sr=None)
    y_bass, _ = load(os.path.join(list_source_dir[i_bass], FILE_BASS),
                     sr=None)
    y_drums, _ = load(os.path.join(list_source_dir[i_drums], FILE_DRUMS),
                      sr=None)
    y_other, _ = load(os.path.join(list_source_dir[i_other], FILE_OTHER),
                      sr=None)

    minsize = min([y_vocal.size, y_bass.size, y_drums.size, y_other.size])

    y_vocal = y_vocal[:minsize]
    y_inst = y_bass[:minsize] + y_drums[:minsize] + y_other[:minsize]
    y_mix = y_vocal + y_inst

    fname = "dsd_random%02d" % count
    util.SaveSpectrogram(y_mix, y_vocal, y_inst, fname)
    print("Saved:" + fname)
    count += 1
