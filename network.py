#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 00:27:08 2017

@author: wuyiming
"""

from chainer import Chain, serializers, optimizers, cuda, config
import chainer.links as L
import chainer.functions as F
import numpy as np
import const


cp = cuda.cupy


class UNet(Chain):
    def __init__(self):
        super(UNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 16, 4, 2, 1)
            self.norm1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, 4, 2, 1)
            self.norm2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, 4, 2, 1)
            self.norm3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, 128, 4, 2, 1)
            self.norm4 = L.BatchNormalization(128)
            self.conv5 = L.Convolution2D(128, 256, 4, 2, 1)
            self.norm5 = L.BatchNormalization(256)
            self.conv6 = L.Convolution2D(256, 512, 4, 2, 1)
            self.norm6 = L.BatchNormalization(512)
            self.deconv1 = L.Deconvolution2D(512, 256, 4, 2, 1)
            self.denorm1 = L.BatchNormalization(256)
            self.deconv2 = L.Deconvolution2D(512, 128, 4, 2, 1)
            self.denorm2 = L.BatchNormalization(128)
            self.deconv3 = L.Deconvolution2D(256, 64, 4, 2, 1)
            self.denorm3 = L.BatchNormalization(64)
            self.deconv4 = L.Deconvolution2D(128, 32, 4, 2, 1)
            self.denorm4 = L.BatchNormalization(32)
            self.deconv5 = L.Deconvolution2D(64, 16, 4, 2, 1)
            self.denorm5 = L.BatchNormalization(16)
            self.deconv6 = L.Deconvolution2D(32, 1, 4, 2, 1)

    def __call__(self, X):

        h1 = F.leaky_relu(self.norm1(self.conv1(X)))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)))
        h5 = F.leaky_relu(self.norm5(self.conv5(h4)))
        h6 = F.leaky_relu(self.norm6(self.conv6(h5)))
        dh = F.relu(F.dropout(self.denorm1(self.deconv1(h6))))
        dh = F.relu(F.dropout(self.denorm2(self.deconv2(F.concat((dh, h5))))))
        dh = F.relu(F.dropout(self.denorm3(self.deconv3(F.concat((dh, h4))))))
        dh = F.relu(self.denorm4(self.deconv4(F.concat((dh, h3)))))
        dh = F.relu(self.denorm5(self.deconv5(F.concat((dh, h2)))))
        dh = F.sigmoid(self.deconv6(F.concat((dh, h1))))
        return dh

    def load(self, fname="unet.model"):
        serializers.load_npz(fname, self)

    def save(self, fname="unet.model"):
        serializers.save_npz(fname, self)


class UNetTrainmodel(Chain):
    def __init__(self, unet):
        super(UNetTrainmodel, self).__init__()
        with self.init_scope():
            self.unet = unet

    def __call__(self, X, Y):
        O = self.unet(X)
        self.loss = F.mean_absolute_error(X*O, Y)
        return self.loss


def TrainUNet(Xlist, Ylist, epoch=40, savefile="unet.model"):
    assert(len(Xlist) == len(Ylist))
    unet = UNet()
    model = UNetTrainmodel(unet)
    model.to_gpu(0)
    opt = optimizers.Adam()
    opt.setup(model)
    config.train = True
    config.enable_backprop = True
    itemcnt = len(Xlist)
    itemlength = [x.shape[1] for x in Xlist]
    subepoch = sum(itemlength) // const.PATCH_LENGTH // const.BATCH_SIZE * 4
    for ep in range(epoch):
        sum_loss = 0.0
        for subep in range(subepoch):
            X = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                         dtype="float32")
            Y = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                         dtype="float32")
            idx_item = np.random.randint(0, itemcnt, const.BATCH_SIZE)
            for i in range(const.BATCH_SIZE):
                randidx = np.random.randint(
                    itemlength[idx_item[i]]-const.PATCH_LENGTH-1)
                X[i, 0, :, :] = \
                    Xlist[idx_item[i]][1:, randidx:randidx+const.PATCH_LENGTH]
                Y[i, 0, :, :] = \
                    Ylist[idx_item[i]][1:, randidx:randidx+const.PATCH_LENGTH]
            opt.update(model, cp.asarray(X), cp.asarray(Y))
            sum_loss += model.loss.data * const.BATCH_SIZE

        print("epoch: %d/%d  loss=%.3f" % (ep+1, epoch, sum_loss))

    unet.save(savefile)
