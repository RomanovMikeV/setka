import numpy
import torch
import skimage.io
import skimage.transform
import skimage.color
import os
import torchvision.datasets
import torchvision.transforms as transforms

import scorch.base

#import scorch

from keras.datasets import mnist
import random
import numpy
import torch
import scorch.base

class DataSet(scorch.base.DataSet):
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(list(x_train.shape) + [1,])
        x_test = x_test.reshape(list(x_test.shape) + [1,])

        train_reorder = numpy.random.permutation(len(x_train))

        x_train = x_train[train_reorder]
        y_train = y_train[train_reorder]

        n_valid = int(len(x_train) * 0.2)
        x_valid = x_train[:n_valid]
        y_valid = y_train[:n_valid]

        x_train = x_train[n_valid:]
        y_train = y_train[n_valid:]

        self.data = {
            'train': torch.from_numpy(x_train).transpose(1, 3).float(),
            'valid': torch.from_numpy(x_valid).transpose(1, 3).float(),
            'test':  torch.from_numpy(x_test).transpose(1, 3).float()
        }

        self.labels = {
            'train': torch.from_numpy(y_train).long(),
            'valid': torch.from_numpy(y_valid).long(),
            'test' : torch.from_numpy(y_test).long()
        }
