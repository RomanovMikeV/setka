import torch
import scorch
import torchvision.datasets
import scorch.base
import os
import numpy

class MNIST(scorch.base.DataSet):
    def __init__(self, val_split=0.1):
        ds_path = '~/datasets/MNIST'

        if not os.path.exists(ds_path):
            os.makedirs(ds_path)

        self.train_ds = torchvision.datasets.MNIST(ds_path,
                                   download=True,
                                   train=True)

        self.test_ds = torchvision.datasets.MNIST(ds_path,
                                   download=True,
                                   train=False)

        self.n_valid = int(len(self.train_ds) * val_split)
        self.order = numpy.arange(len(self.train_ds))


    def getlen(self, subset):
        if subset == 'train':
            return len(self.train_ds) - self.n_valid
        elif subset == 'valid':
            return self.n_valid
        elif subset == 'test':
            return len(self.test_ds)

    def getitem(self, subset, idx):
        if subset == 'train':
            data = self.train_ds[self.order[self.n_valid + idx]]
        elif subset == 'valid':
            data = self.train_ds[self.order[idx]]
        elif subset == 'test':
            data = self.test_ds[idx]

        return [torch.Tensor(numpy.array(data[0])).unsqueeze(0)], [data[1]], subset + "_" + str(idx)
