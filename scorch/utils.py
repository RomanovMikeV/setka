import numpy
import torch

def torch2numpy(image):
    res = image.clone()
    res = res.swapaxes(0, 1).swapaxes(1, 2)
    return res


def numpy2torch(image):
    res = image.clone()
    res = res.swapaxes(2, 1).swapaxes(1, 0)
    return res
