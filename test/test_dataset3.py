import setka
import random
import torch
import numpy
random.seed(0)

import torchvision.datasets
import torchvision.transforms

class CIFAR10(setka.base.Dataset):
    def __init__(self):
        super().__init__()

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data = torchvision.datasets.CIFAR10(
            './cifar10_data',
            train=False,
            download=True,
            transform=train_transforms)

        test_data = torchvision.datasets.CIFAR10(
            './cifar10_data',
            train=False,
            download=True,
            transform=test_transforms)

        self.subsets = {
            'train': train_data,
            'valid': test_data,
            'test': test_data
        }

    def getitem(self, subset, index):
        data, label = self.subsets[subset][index]
        return data, label, 'text', torch.ones(random.randint(1, 10)), numpy.array([0, 1])

    def getlen(self, subset):
        return len(self.subsets[subset])