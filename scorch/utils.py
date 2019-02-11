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


def denorm(image, int_image=False,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]):
    res = image.clone()
    std = numpy.array(std)
    mean = numpy.array(mean)

    if int_image:
        std *= 255.0
        mean *= 255.0

    res = res * std + mean
    return res
