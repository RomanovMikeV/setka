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
	'''
	Collect all the data about the dataset
	into one index object.
	'''
        pass


    def shuffle(self):
	'''
	This method is called after one training epoch
	in order to force shuffling of the training
	dataset. This, shuffle the training dataset
	here.
	'''
	pass

class DataSet():
    def __init__(self, ds_index, mode='train'):
        self.ds_index = ds_index
        self.mode = mode

	    # Add your transforms here
        self.train_transform = transforms.Compose([])
        self.valid_transform = transforms.Compose([])
        self.test_transform = transforms.Compose([])

    def __len__(self):
        '''
        Get the lenght of the dataset
        '''
        if self.mode == 'test':
            pass

        elif self.mode == 'valid':
            pass

        else:
            pass


    def __getitem__(self, index):
        img = None
        target = None

        if self.mode == 'test':
            img =
            target =

        elif self.mode == 'valid':
            img =
            target =

        else:
            img =
            target =

        return [img], [target]
