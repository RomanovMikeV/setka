from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import keras
import random

import numpy

import torch

import scorch.base
import scorch.callbacks
#import scorch.scripts

import matplotlib.pyplot as plt

numpy.random.seed(0)
random.seed(0)

# Computing result with keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(list(x_train.shape) + [1,])
x_test = x_test.reshape(list(x_test.shape) + [1,])

train_reorder = numpy.random.permutation(len(x_train))

x_train = x_train[train_reorder]
y_train = y_train[train_reorder]

n_valid = int(len(x_train) * 0.2)

x_valid = x_train[:n_valid]
y_valid = y_train[:n_valid]

x_train = x_train[n_valid:]
y_train = y_train[n_valid:]

labels_train = numpy.zeros([len(y_train), 10])
labels_valid = numpy.zeros([len(y_valid), 10])
labels_test = numpy.zeros([len(y_test), 10])

labels_train[numpy.arange(len(y_train)), y_train] = 1
labels_valid[numpy.arange(len(y_valid)), y_valid] = 1
labels_test[numpy.arange(len(y_test)), y_test] = 1


# Computing result with scorch

class DataSet(scorch.base.DataSet):
    def __init__(self):
        super(DataSet, self).__init__()
        self.data = {
            'train': torch.from_numpy(x_train).transpose(1, 3).float(),
            'valid': torch.from_numpy(x_valid).transpose(1, 3).float(),
            'test':  torch.from_numpy(x_test).transpose(1, 3).float()
        }

        self.labels = {
            'train': torch.from_numpy(y_train).long(),
            'valid': torch.from_numpy(y_valid).long(),
            'test' : torch.from_numpy(y_test).long()
        }


class Network(scorch.base.Network):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 8, 5, padding=2)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, padding=2)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 56)
        self.fc3 = torch.nn.Linear(56, 10)

    def forward(self, input):
        res = input[0]
        res = self.pool1(torch.relu(self.conv1(res)))
        res = self.pool2(torch.relu(self.conv2(res)))

        res = res.view([res.size(0), -1])

        res = torch.relu(self.fc1(res))
        res = torch.relu(self.fc2(res))
        res = self.fc3(res)

        return [res]

net = Network()

def criterion(input, target):
    return torch.nn.functional.cross_entropy(input[0], target[0])

def loss(input, target):
    return criterion(input, target)

def accuracy(preds, target):
    return (preds[0].argmax(dim=1) == target[0]).float().sum(), len(preds[0])

def visualize(one_input, one_target, one_output):
    res = {'figures': []}

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(one_input[0][0])

    res['figures'] = {'input': fig}
    res['images'] = {'zero_img': numpy.zeros([3, 100, 100]).astype('float64')}
    res['texts'] = {'random_text': 'Little brown fox jumped over the lazy dog. Съешь ещё этих мягких французских булок да выпей яду.'}
    res['audios'] = {'random_noise': numpy.random.randn(1000)}

    plt.close()

    return res

def processing_placeholder(inp):
    return inp

def cycle(progress):
    if progress < 0.5:
        return 0.0
    else:
        return 1.0

trainer = scorch.base.Trainer(net,
                  criterion=criterion,
                  optimizers=[
                    scorch.base.OptimizerSwitch(net.fc3, torch.optim.Adam, lr=3.0e-5, is_active=True),
                    scorch.base.OptimizerSwitch(net.fc2, torch.optim.Adam, lr=3.0e-5, is_active=False),
                    scorch.base.OptimizerSwitch(net.fc1, torch.optim.Adam, lr=3.0e-5, is_active=False),
                    scorch.base.OptimizerSwitch(net.conv2, torch.optim.Adam, lr=3.0e-5, is_active=False),
                    scorch.base.OptimizerSwitch(net.conv1, torch.optim.Adam, lr=3.0e-5, is_active=False)],
                  callbacks=[
                             scorch.callbacks.ShuffleDataset(shuffle_valid=True),
                             scorch.callbacks.ComputeMetrics(),
                             scorch.callbacks.ComputeMetrics(
                                metrics=[accuracy, loss],
                                divide_first=[True, False]),
                             scorch.callbacks.MakeCheckpoints('accuracy'),
                             scorch.callbacks.SaveResult(),
                             scorch.callbacks.SaveResult(f=processing_placeholder),
                             scorch.callbacks.WriteToTensorboard(f=visualize),
                             scorch.callbacks.Logger(f=visualize),
                             scorch.callbacks.ReduceLROnPlateau('accuracy', max_mode=True),
                             scorch.callbacks.UnfreezeOnPlateau('accuracy', max_mode=True),
                             scorch.callbacks.CyclicLR(cycle=cycle)],
                  seed=1,
                  silent=False)
dataset = DataSet()

for epoch in range(10):
    trainer.train_one_epoch(dataset,
                            num_workers=2,
                            max_iterations=10,
                            batch_size=32)

    trainer.validate_one_epoch(dataset,
                               subset='valid',
                               num_workers=2,
                               max_iterations=10,
                               batch_size=32)


trainer = scorch.base.Trainer(net,
                  criterion=criterion,
                  optimizers=[
                    scorch.base.OptimizerSwitch(net.fc3, torch.optim.Adam, lr=3.0e-5, is_active=True),
                    scorch.base.OptimizerSwitch(net.fc2, torch.optim.Adam, lr=3.0e-5, is_active=False),
                    scorch.base.OptimizerSwitch(net.fc1, torch.optim.Adam, lr=3.0e-5, is_active=False),
                    scorch.base.OptimizerSwitch(net.conv2, torch.optim.Adam, lr=3.0e-5, is_active=False),
                    scorch.base.OptimizerSwitch(net.conv1, torch.optim.Adam, lr=3.0e-5, is_active=False)],
                  callbacks=[scorch.callbacks.ComputeMetrics(
                                metrics=[accuracy, loss]),
                             scorch.callbacks.MakeCheckpoints('accuracy'),
                             scorch.callbacks.SaveResult(),
                             scorch.callbacks.WriteToTensorboard(f=visualize),
                             scorch.callbacks.Logger(f=visualize),
                             scorch.callbacks.ReduceLROnPlateau('accuracy', max_mode=True),
                             scorch.callbacks.UnfreezeOnPlateau('accuracy', max_mode=True),
                             scorch.callbacks.CyclicLR(cycle=cycle)],
                  seed=1,
                  silent=False)

dataset = DataSet()

for epoch in range(10):
    trainer.train_one_epoch(dataset,
                            num_workers=2,
                            max_iterations=10,
                            batch_size=32)

    trainer.validate_one_epoch(dataset,
                               subset='valid',
                               num_workers=2,
                               max_iterations=10,
                               batch_size=32)


# res = trainer.train(
#              dataset,
#              batch_size=32,
#              num_workers=2,
#              validate_on_train=True,
#              max_train_iterations=2,
#              max_valid_iterations=2,
#              max_test_iterations=2,
#              solo_test=True,
#              epochs=2)
for index in range(2):
    trainer.train_one_epoch(dataset,
                            batch_size=32,
                            num_workers=2,
                            max_iterations=2)

    trainer.validate_one_epoch(dataset,
                               subset='train',
                               batch_size=32,
                               num_workers=2,
                               max_iterations=2)

    trainer.validate_one_epoch(dataset,
                               subset='valid',
                               batch_size=32,
                               num_workers=2,
                               max_iterations=2)

trainer.predict(dataset,
                subset='test',
                batch_size=32,
                max_iterations=2)

trainer.predict(dataset,
                subset='valid',
                batch_size=32,
                max_iterations=2)

trainer.save('checkpoint')

trainer.load('./checkpoints/checkpoint_latest.pth.tar')
