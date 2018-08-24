import numpy
import torch
import skimage.io
import skimage.transform
import skimage.color
import os
import torchvision.datasets
import torchvision.transforms as transforms

class DataSetIndex():
    def __init__(self, path):
        mnist_ds = torchvision.datasets.MNIST(path, download=True)

        data   = mnist_ds.train_data
        labels = mnist_ds.train_labels

        test_split = 0.1
        valid_split = 0.1

        n_test = int(data.size(0) * test_split)
        n_valid = int(data.size(0) * valid_split)

        n_train = data.size(0) - n_test - n_valid

        order = numpy.random.permutation(data.size(0))

        data = data[order, :, :]
        labels = labels[order]

        #print('OK')
        self.test_data   = data[:n_test, :, :]
        self.test_labels = labels[:n_test]

        self.valid_data   = data[n_test:n_test + n_valid, :, :]
        self.valid_labels = labels[n_test:n_test + n_valid]

        self.train_data = data[n_test + n_valid:, :, :]
        self.train_labels = labels[n_test + n_valid:]

    def shuffle(self):
        new_order = numpy.arange(len(self.train_data))
        numpy.random.shuffle(new_order)

        self.train_data = self.train_data[new_order]
        self.train_labels = self.train_labels[new_order]


class DataSet():
    def __init__(self, ds_index, mode='train'):
        self.ds_index = ds_index
        self.mode = mode

        self.train_transform = transforms.Compose([])
        self.valid_transform = transforms.Compose([])
        self.test_transform = transforms.Compose([])

    def __len__(self):
        if self.mode == 'test':
            return self.ds_index.test_data.size(0)

        elif self.mode == 'valid':
            return self.ds_index.valid_data.size(0)

        else:
            return self.ds_index.train_data.size(0)


    def __getitem__(self, index):
        img = None
        target = None

        if self.mode == 'test':
            img = self.test_transform(self.ds_index.test_data[index, :, :])
            target = self.ds_index.test_labels[index]

        elif self.mode == 'valid':
            img = self.valid_transform(self.ds_index.valid_data[index, :, :])
            target = self.ds_index.valid_labels[index]

        else:
            img = self.train_transform(self.ds_index.train_data[index, :, :])
            target = self.ds_index.train_labels[index]

        img = img.unsqueeze(0)

        img = img.float()

        return [img], [target]
