Master branch:
[![Build Status](https://travis-ci.com/RomanovMikeV/setka.svg?branch=master)](https://travis-ci.com/RomanovMikeV/setka)
[![codecov](https://codecov.io/gh/RomanovMikeV/setka/branch/master/graph/badge.svg)](https://codecov.io/gh/RomanovMikeV/setka)

Dev branch:
[![Build Status](https://travis-ci.com/RomanovMikeV/setka.svg?branch=dev)](https://travis-ci.com/RomanovMikeV/setka)
[![codecov](https://codecov.io/gh/RomanovMikeV/setka/branch/dev/graph/badge.svg)](https://codecov.io/gh/RomanovMikeV/setka)

# Setka: pipeline builder for Neural Network training.

Setka is a powerful and flexible tool for neural network training
with accent on fast prototyping and reproducibility. Includes
modules for logging the training process, common tricks
and visualisation.

## Overview

The network training is now as simple as:

* You build (or load) the model
* You build (or load) the dataset
* You wrap the dataset with a provided wrapper for convenience
* You define a pipeline with available (or yours) Setka modules
* You trigger train
* You enjoy

## Installation

To install this package, use
```
pip install git+http://github.com/RomanovMikeV/setka
```

## Example

1) Define the dataset:
```python
import setka.base
import torchvision.transforms
import torchvision.datasets

class CIFAR10(setka.base.DataSet):
    def __init__(self,
                 root='~/datasets'):

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

        self.train_data = torchvision.datasets.CIFAR10(
            '~/datasets', train=True, download=True,
            transform=train_transforms)
        self.test_data = torchvision.datasets.CIFAR10(
            '~/datasets', train=False, download=True,
            transform=test_transforms)

        self.n_valid = int(0.05 * len(self.train_data))

        self.subsets = ['train', 'valid', 'test']

    def getlen(self, subset):
        if subset == 'train':
            return len(self.train_data) - self.n_valid
        elif subset == 'valid':
            return self.n_valid
        elif subset == 'test':
            return len(self.test_data)

    def getitem(self, subset, index):
        if subset == 'train':
            image, label = self.train_data[self.n_valid + index]
            return {'image': image, 'label': label}
        elif subset == 'valid':
            image, label = self.train_data[index]
            return {'image': image, 'label': label}
        elif subset == 'test':
            image, label = self.test_data[index]
            return {'image': image, 'label': label}

```
2) Define your model:
```python
import torch.nn

class SimpleModel(torch.nn.Module):
    def __init__(self, channels, input_channels=3, n_classes=10):
        super().__init__()

        modules = []

        in_c = input_channels
        for out_c in channels:
            modules.append(torch.nn.Conv2d(in_c, out_c, 3, padding=1))
            modules.append(torch.nn.BatchNorm2d(out_c))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.MaxPool2d(2))

            in_c = out_c

        self.encoder = torch.nn.Sequential(*modules)
        self.decoder = torch.nn.Linear(in_c, n_classes)

    def __call__(self, input):
        x = input['image']
        # print(x.shape)
        # print(self.encoder)
        x = self.encoder(x).mean(dim=-1).mean(dim=-1)
        x = self.decoder(x)

        return x
```

3) Define your pipeline and train:
```python
import setka.pipes


def loss(pred, input):
    return torch.nn.functional.cross_entropy(pred, input['label'])


def acc(pred, input):
    return (input['label'] == pred.argmax(dim=1)).float().sum() / float(pred.size(0))



ds = CIFAR10()
model = SimpleModel(channels=[8, 16, 32, 64])

trainer = setka.base.Trainer(
    pipes=[
        setka.pipes.DataSetHandler(ds, 32, workers=4, timeit=True,
                                   shuffle={'train': True, 'valid': True, 'test': False},
                                   epoch_schedule=[
                                       {'mode': 'train', 'subset': 'train'},
                                       {'mode': 'valid', 'subset': 'train', 'n_iterations': 100},
                                       {'mode': 'valid', 'subset': 'valid'},
                                       {'mode': 'valid', 'subset': 'test'}]),
        setka.pipes.ModelHandler(model),
        setka.pipes.LossHandler(loss),
        setka.pipes.ComputeMetrics([loss, acc]),
        setka.pipes.ProgressBar(),
        setka.pipes.OneStepOptimizers([setka.base.OptimizerSwitch(model, torch.optim.Adam, lr=3.0e-2)]),
        setka.pipes.TuneOptimizersOnPlateau('acc', max_mode=True, subset='valid', lr_factor=0.3, reset_optimizer=True),
        setka.pipes.MakeCheckpoints('acc', max_mode=True)
    ]
)


trainer.run_train(10)
```


## If you need more functionality

You may define your own pipes without hustle. Here is an
example of pipe that prints when the trainer performs callbacks 
"before_batch"

```python
import setka.base

class StatusPrinter(setka.pipes.Pipe):
    def __init__(self):
        super().__init__()
    
    def before_batch(self):
        print("In before batch")
```
