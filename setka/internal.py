import numpy
import torch
from torch._six import string_classes, int_classes
import collections
import re

class DataSetWrapper():
    '''
    This is the wrapper for a dataset class.

    This class should have the following methods:
    __init__, __len__, __getitem__.
    '''

    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

        self.order = numpy.arange(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        real_index = self.order[index]
        dataset_res = self.dataset[real_index]

        return dataset_res, str(self.name) + '_' + str(real_index)

    def shuffle(self):
        self.order = numpy.random.permutation(len(self.dataset))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
