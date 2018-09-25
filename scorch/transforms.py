from __future__ import division

import numpy
import skimage.transform
import torch
import random

import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import skimage.transform
import numpy


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def reset(self):
        for index in range(len(self.transforms)):
            self.transforms[index].reset()

    def __call__(self, tensor):
        result = tensor

        for index in range(len(self.transforms)):
            result = self.transforms[index](result)

        return result


class ToTensor():
    def __init__(self):
        pass

    def reset(self):
        pass

    def __call__(self, input):
        return torch.FloatTensor(input)


class RandomHFlip():
    def __init__(self, p=0.5):
        self.p = p
        self.flip = self.get_params(self.p)

    @staticmethod
    def get_params(p):
        return random.random() > p

    def reset(self):
        self.flip = self.get_params(self.p)

    def __call__(self, tensor):
        if self.flip:
            img = tensor.numpy()[:, :, ::-1]
        else:
            img = tensor.numpy()
        return torch.FloatTensor(numpy.array(img))


class RandomVFlip():
    def __init__(self, p=0.5):
        self.p = p
        self.flip = self.get_params(self.p)

    @staticmethod
    def get_params(p):
        return random.random() > p

    def reset(self):
        self.flip = self.get_params(self.p)

    def __call__(self, tensor):
        if self.flip:
            img = tensor.numpy()[:, ::-1, :]
        else:
            img = tensor.numpy()
        return torch.FloatTensor(numpy.array(img))


class Scale():
    def __init__(self, shape=[224, 224]):
        self.shape=shape

    def reset(self):
        pass

    def __call__(self, tensor):
        img = tensor.numpy().swapaxes(0, 1).swapaxes(1, 2)
        img_max = img.max()
        img_min = img.min()

        res = (img - img_min) / (img_max - img_min + 1.0e-8)

        res = skimage.transform.resize(res, self.shape,
                                       mode='reflect').swapaxes(1, 2).swapaxes(0, 1)

        res = res * (img_max - img_min) + img_min
        return torch.FloatTensor(res)

class CenterCrop():
    def __init__(self, size=[224, 224]):
        self.size = size

    def reset(self):
        pass

    def __call__(self, tensor):
        res = tensor.clone()

        for index in range(len(self.size)):
            start = int((tensor.shape[index + 1] - self.size[index]) * 0.5)
            res = res.transpose(index + 1, 0)[start:start + self.size[index]].transpose(0, index + 1)

        return res

class RandomCrop():
    def __init__(self, size=[224, 224]):
        self.size=size

        self.pad = self.get_params(self.size)


    @staticmethod
    def get_params(size):
        pad = []
        for index in range(len(size)):
            pad.append(int(random.random()))

        return pad


    def reset(self):
        self.pad = self.get_params(self.size)


    def __call__(self, tensor):
        res = tensor.clone()

        for index in range(len(self.size)):
            start = int((tensor.shape[index + 1] - self.size[index]) * self.pad[index])
            res = res.transpose(index + 1, 0)[start:start + self.size[index]].transpose(0, index + 1)

        return res


class RandomFixedCrop():
    def __init__(self, max_size=1.0, min_size=0.5):
        self.max_size = max_size
        self.min_size = min_size

        self.size, self.x_pad, self.y_pad = self.get_params(self.min_size, self.max_size)


    @staticmethod
    def get_params(min_size, max_size):
        size = random.random() * (min_size - max_size) + min_size
        pad_limit = 1.0 - size
        x_pad = random.random() * pad_limit
        y_pad = random.random() * pad_limit

        return size, x_pad, y_pad


    def reset(self):
        self.size, self.x_pad, self.y_pad = self.get_params(self.min_size, self.max_size)


    def __call__(self, tensor):
        x_start = int(self.x_pad * tensor.size(1))
        y_start = int(self.y_pad * tensor.size(2))

        x_size = int(self.size * tensor.size(1))
        y_size = int(self.size * tensor.size(2))

        return tensor[:, x_start:x_start+x_size, y_start:y_start+y_size]


class RandomJitter():
    def __init__(self, x_pad_limit=0.05, y_pad_limit=0.05):
        self.x_pad = x_pad_limit
        self.y_pad = y_pad_limit
        self.left, self.right, self.top, self.bottom = self.get_params(x_pad_limit, y_pad_limit)

    @staticmethod
    def get_params(x_pad, y_pad):
        left_pad = random.random() * x_pad
        right_pad = random.random() * x_pad
        top_pad = random.random() * y_pad
        bottom_pad = random.random() * y_pad

        return left_pad, right_pad, top_pad, bottom_pad


    def reset(self):
        self.left, self.right, self.top, self.bottom = self.get_params(x_pad_limit, y_pad_limit)


    def __call__(self, tensor):
        top_pad = int(tensor.size(1) * self.top)
        bottom_pad = tensor.size(1) - int(tensor.size(1) * self.bottom)
        left_pad = int(tensor.size(2) * self.left)
        right_pad = tensor.size(2) - int(tensor.size(2) * self.right)

        return tensor[:, top_pad:bottom_pad, left_pad:right_pad]


class RandomRotation(object):
    def __init__(self, angles=[-180.0, 180.0]):
        self.angles = angles

        self.angle = self.get_params(self.angles[0], self.angles[1])

    @staticmethod
    def get_params(min_angle, max_angle):
        angle = random.random() * (max_angle - min_angle) + min_angle
        return angle

    def reset(self):
        self.angle = self.get_params(self.angles[0], self.angles[1])

    def __call__(self, image):
        numpy.random.seed()
        res = image.numpy()

        image_max = res.max()
        image_min = res.min()

        res = (res - image_min) / (image_max - image_min + 1.0e-8)

        res = res.swapaxes(0, 1).swapaxes(1, 2)

        res = skimage.transform.rotate(res, self.angle)
        res = res.swapaxes(1, 2).swapaxes(0, 1)
        res = res * (image_max - image_min) + image_min
        res = torch.FloatTensor(res)

        return res

class RandomXYFlip:
    def __init__(self, p=0.5):
        self.p = 0.5
        self.flag = get_params(self.p)

    def reset(self):
        self.flag = get_params(self.p)

    @staticmethod
    def get_params(p):
        return random.random() > p

    def __call__(self, tensor):
        if self.flip:
            return tensor.transpose(1, 2)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def reset(self):
        pass

    def __call__(self, tensor):
        return normalize(tensor, self.mean, self.std)
#def g_noise():
#    pass

#def p_noise():
#    pass

#def rot():
#    pass
