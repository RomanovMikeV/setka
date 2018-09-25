import numpy
import torch
import skimage.io
import skimage.transform
import skimage.color
import os
import torchvision.datasets
import torchvision.transforms as transforms

class DataSetIndex():

    # This class is not necessary, but it is recommended to use it to save
    # memory and indexing time.

    def __init__(self, path):

        # Here you should make a full index of your dataset that you will use

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

        self.test_data   = data[:n_test, :, :]
        self.test_labels = labels[:n_test]

        self.valid_data   = data[n_test:n_test + n_valid, :, :]
        self.valid_labels = labels[n_test:n_test + n_valid]

        self.train_data = data[n_test + n_valid:, :, :]
        self.train_labels = labels[n_test + n_valid:]

    def shuffle(self):
        new_order = numpy.arange(len(self.train_data))
        numpy.random.shuffle(new_order)
        self.train_order = new_order


class DataSet():
    def __init__(self, ds_index, mode='train'):
        self.ds_index = ds_index
        self.mode = mode

        self.transform = transforms.Compose([])

        if self.mode == 'test':
            self.data = self.ds_index.test_data
            self.labels = self.ds_index.test_labels
            self.order = numpy.arange(len(self.ds_index.test_data))
        elif self.mode == 'valid':
            self.data = self.ds_index.valid_data
            self.labels = self.ds_index.valid_labels
            self.order = numpy.arange(len(self.ds_index.valid_data))
        else:
            self.data = self.ds_index.train_data
            self.labels = self.ds_index.train_labels
            self.order = numpy.arange(len(self.ds_index.train_data))


    def __len__(self):
        if self.mode == 'test':
            return self.data.size(0)

        elif self.mode == 'valid':
            return self.data.size(0)

        else:
            return self.data.size(0)


    def __getitem__(self, index):
        img = None
        target = None
        id = None

        id = self.order[index]

        img = self.transform(self.data[id, :, :])
        target = self.labels[id]
        id = 'train/' + str(id)

        img = img.unsqueeze(0).float()

        img = img.float()

        return [img], [target], id

    def shuffle(self):

        # This is the rule how to shuffle your dataset from epoch to epoch

        if self.mode == 'train':
            new_order = numpy.arange(len(self.data))
            numpy.random.shuffle(new_order)
            self.train_order = new_order

        else:
            pass
