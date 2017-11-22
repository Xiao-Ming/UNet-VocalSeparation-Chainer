#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:52:48 2017

@author: wuyiming
"""

import util
import network


"""
Code example for training U-Net
"""

"""
Xlist,Ylist = util.LoadDataset(target="vocal")
print("Dataset loaded.")
network.TrainUNet(Xlist,Ylist,savefile="unet.model",epoch=30)
"""


"""
Code example for performing vocal separation with U-Net
"""
fname = "original_mix.wav"
mag,phase = util.LoadAudio(fname)
start = 2048
end = 2048+1024

mask = util.ComputeMask(mag[:,start:end],unet_model="unet.model",hard=False)

util.SaveAudio("vocal-%s" % fname,mag[:,start:end]*mask,phase[:,start:end])
util.SaveAudio("inst-%s" % fname,mag[:,start:end]*(1-mask),phase[:,start:end])
util.SaveAudio("orig-%s" % fname,mag[:,start:end],phase[:,start:end])
