import numpy
import torch
import skimage.io
import skimage.transform
import skimage.color
import utils
import os

class DataSetIndex():
    def __init__(self, path):
        pass

class DataSet():
    def __init__(self, ds_index, mode='train'):
        self.ds_index = ds_index
    
    def __len__(self):
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
            pass
        
        elif self.mode == 'valid':
            pass
        
        else:
            pass
        
        return img, target