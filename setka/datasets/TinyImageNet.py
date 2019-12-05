import os
import urllib.request
import zipfile
import collections
from PIL import Image
import numpy
import imgaug as ia
import imgaug.augmenters as iaa
import torch
import setka
import setka.base


class TinyImageNet(setka.base.DataSet):
    def __init__(self, root='~/data',
                 transforms={
                     'train': iaa.Sequential([
                         iaa.Fliplr(0.5),
                         iaa.Sometimes(0.5, iaa.Affine(
                             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                             rotate=(-10, 10),
                             shear=(-5, 5),
                             order=[1],
                             cval=(0, 255),
                             mode=ia.ALL
                         )),
                         iaa.CropToFixedSize(64, 64)
                     ]),
                     'valid': iaa.Sequential([
                         iaa.Fliplr(0.5),
                         iaa.Sometimes(0.5, iaa.Affine(
                             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                             rotate=(-10, 10),
                             shear=(-5, 5),
                             order=[1],
                             cval=(0, 255),
                             mode=ia.ALL
                         )),
                         iaa.CropToFixedSize(64, 64)
                     ]),
                     'test': iaa.Sequential([])
                 }):

        data_path = os.path.join(root, 'TinyImagenet')
        self.transforms = transforms

        if not os.path.exists(data_path):
            zip_path = os.path.join(root, 'TinyImagenet.zip')
            if not os.path.exists(root):
                os.makedirs(data_path)

            urllib.request.urlretrieve(
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
                zip_path)

            zip_ref = zipfile.ZipFile(zip_path, 'r')
            zip_ref.extractall(data_path)
            zip_ref.close()

            os.remove(zip_path)

        self.data_path = os.path.join(data_path, 'tiny-imagenet-200')

        train_path = os.path.join(self.data_path, 'train')
        valid_path = os.path.join(self.data_path, 'val')
        test_path = os.path.join(self.data_path, 'test')

        classnames = list(set(os.listdir(train_path)))

        self.classes = {}
        self.train_images = []

        for classname in sorted(classnames):
            self.classes[classname] = len(self.classes)

            class_path = os.path.join(train_path, classname, 'images')
            for image in os.listdir(class_path):
                self.train_images.append([classname, image])

        self.valid_images = []
        for line in open(os.path.join(valid_path, 'val_annotations.txt')):
            line = line.split('\t')
            self.valid_images.append([line[1], line[0]])

        self.test_images = []
        for image in os.listdir(os.path.join(test_path, 'images')):
            self.test_images.append(image)

        self.subsets = ['train', 'valid', 'test']


    def getlen(self, subset):
        if subset == 'train':
            return len(self.train_images)
        elif subset == 'valid':
            return len(self.valid_images)
        elif subset == 'test':
            return len(self.test_images)

    def getitem(self, subset, index):
        if subset == 'train':
            classname, image = self.train_images[index]
            img = Image.open(os.path.join(self.data_path, 'train', classname, 'images', image))
            img = numpy.array(img)
            img = self.transforms['train'].augment_image(img)
            img = (img / 255.0) - 0.5
            img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

            return {'image': img, 'class': self.classes[classname]}

        elif subset == 'valid':
            classname, image = self.valid_images[index]
            img = Image.open(os.path.join(self.data_path, 'val', 'images', image))
            img = numpy.array(img)
            img = self.transforms['valid'].augment_image(img)
            img = (img / 255.0) - 0.5
            img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

            return {'image': img, 'class': self.classes[classname]}

        elif subset == 'test':
            image = self.test_images[index]
            img = Image.open(os.path.join(self.data_path, 'test', 'images', image))
            img = numpy.array(img)
            img = self.transforms['test'].augment_image(img)
            img = (img / 255.0) - 0.5
            img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

            return {'image': img}