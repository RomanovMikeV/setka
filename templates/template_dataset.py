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
import scorch.scripts

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

# class DataSet(scorch.base.DataSetIndex):
#
#     # This class is not necessary, but it is recommended to use it to save
#     # memory and indexing time.
#
#     def __init__(self, path):
#
#         # Here you should make a full index of your dataset that you will use
#
#         mnist_ds = torchvision.datasets.MNIST(path, download=True)
#
#         data   = mnist_ds.train_data
#         labels = mnist_ds.train_labels
#
#         test_split = 0.1
#         valid_split = 0.1
#
#         n_test = int(data.size(0) * test_split)
#         n_valid = int(data.size(0) * valid_split)
#
#         n_train = data.size(0) - n_test - n_valid
#
#         order = numpy.random.permutation(data.size(0))
#
#         data = data[order, :, :]
#         labels = labels[order]
#
#         self.data = {
#             'train': data[n_test + n_valid:, :, :],
#             'valid': data[n_test:n_test + n_valid, :, :],
#             'test':  data[:n_test, :, :]
#         }
#
#         self.labels = {
#             'train': labels[n_test + n_valid:],
#             'valid': labels[n_test:n_test + n_valid],
#             'test':  labels[:n_test]
#         }
#
#     def get_len(self, mode='train'):
#         return len(self.data[mode])
#
#     def get_item(self, index, mode='train'):
#         return [self.data[mode][index]], [self.labels[mode][index]], str(index)


# class DataSet(scorch.base.DataSet):
#     def __init__(self, ds_index, mode='train'):
#         super(DataSet, self).__init__(ds_index, mode=mode)
#
#         self.transform = transforms.Compose([])
#
#         if self.mode == 'test':
#             self.data = self.ds_index.test_data
#             self.labels = self.ds_index.test_labels
#             self.order = numpy.arange(len(self.ds_index.test_data))
#         elif self.mode == 'valid':
#             self.data = self.ds_index.valid_data
#             self.labels = self.ds_index.valid_labels
#             self.order = numpy.arange(len(self.ds_index.valid_data))
#         else:
#             self.data = self.ds_index.train_data
#             self.labels = self.ds_index.train_labels
#             self.order = numpy.arange(len(self.ds_index.train_data))
#
#
#     def __len__(self):
#         if self.mode == 'test':
#             return self.data.size(0)
#
#         elif self.mode == 'valid':
#             return self.data.size(0)
#
#         else:
#             return self.data.size(0)
#
#
#     def __getitem__(self, index):
#         img = None
#         target = None
#         id = None
#
#         id = self.order[index]
#
#         img = self.transform(self.data[id, :, :])
#         target = self.labels[id]
#         id = 'train/' + str(id)
#
#         img = img.unsqueeze(0).float()
#
#         img = img.float()
#
#         return [img], [target], id
#
#     def shuffle(self):
#
#         # This is the rule how to shuffle your dataset from epoch to epoch
#
#         if self.mode == 'train':
#             new_order = numpy.arange(len(self.data))
#             numpy.random.shuffle(new_order)
#             self.train_order = new_order
#
#         else:
#             pass
