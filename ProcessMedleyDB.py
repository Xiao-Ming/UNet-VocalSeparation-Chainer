#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:59:08 2017

@author: wuyiming
"""

import os
from librosa.core import load
from librosa.util import find_files
import yaml
import util


PATH_MENDLEY = "MedleyDB/Audio"

metadatalist = find_files(PATH_MENDLEY, ext="yaml")

all_voctracks = []
all_insttracks = []

for metafile in metadatalist:
    print("YAML file: %s" % metafile)
    songname = metafile.split("/")[-2]
    print("song: %s" % songname)
    with open(metafile, "r+") as f:
        data = yaml.load(f)

    if data["instrumental"] != "no":
        print("Instrumental track. Skipped.")
        continue

    stem_voc = []
    stem_inst = []
    stems_path = os.path.join(PATH_MENDLEY, songname, data["stem_dir"])
    mixfilename = data["mix_filename"]
    for s in data["stems"]:
        stem = data["stems"][s]
        fname = stem["filename"]

        print(
            "stem: %s %s %s" % (fname, stem["component"], stem["instrument"]))
        if (stem["instrument"].find("male") >= 0) or \
                (stem["instrument"].find("singer") > 0):
            stem_voc.append(fname)
            all_voctracks.append(fname)
            print("Is vocal!")
        else:
            stem_inst.append(fname)
            all_insttracks.append(fname)

    print("detected vocals:")
    print(stem_voc)
    if (len(stem_voc) == 0) or (len(stem_inst) == 0):
        print("empty vocal or inst...skip")
        continue
    audio_vocal = sum([load(os.path.join(stems_path, f), sr=None, mono=True)[0]
                       for f in stem_voc])
    audio_inst = sum([load(os.path.join(stems_path, f), sr=None, mono=True)[0]
                      for f in stem_inst])
    audio_mix, _ = load(os.path.join(PATH_MENDLEY, songname, mixfilename),
                        sr=None, mono=True)
    util.SaveSpectrogram(audio_mix, audio_vocal, audio_inst, songname)
