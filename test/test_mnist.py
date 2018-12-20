from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import keras
import random

import numpy

import torch

import scorch.base
#import scorch.scripts

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

# Keras model definition

model = Sequential()

model.add(Conv2D(64, 5, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(128, 5, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=3.0e-5),
              metrics=['accuracy'])

model.fit(x_train, labels_train, epochs=1, batch_size=32)


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

        self.conv1 = torch.nn.Conv2d(1, 64, 5, padding=2)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.fc1 = torch.nn.Linear(6272, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, input):
        res = input[0]
        res = self.pool1(torch.relu(self.conv1(res)))
        res = self.pool2(torch.relu(self.conv2(res)))

        res = res.view([res.size(0), -1])

        res = torch.relu(self.fc1(res))
        res = torch.relu(self.fc2(res))
        res = self.fc3(res)

        return [res]


class Trainer(scorch.base.Trainer):
    def __init__(self, model):
        super(Trainer, self).__init__(model)

        self.criterion_f = torch.nn.CrossEntropyLoss(reduction='sum')
        self.optimizers = [
            scorch.base.OptimizerSwitch(self.model, torch.optim.Adam, lr=3.0e-5)]

    def criterion(self, preds, target):
        return self.criterion_f(preds[0], target[0])

    def metrics(self, preds, target):
        acc = (preds[0].argmax(dim=1) == target[0]).float().mean()
        return {'main': acc}

    def process_result(self, inputs, preds, id):
        return {id: preds[0] > 0.5}


net = Network()
trainer = Trainer(net)
dataset = DataSet()

for epoch in range(1):
    trainer.train_one_epoch(dataset, batch_size=32)
    trainer.validate_one_epoch(dataset, batch_size=32)
    #for i, t, id in trainer.test_one_epoch(dataset, max_iterations=10):
    #    pass

res = model.fit(x_train, labels_train, epochs=5, batch_size=32)
res = trainer.train(dataset, batch_size=32,
             #max_train_iterations=2,
             #max_valid_iterations=2,
             max_test_iterations=2,
             epochs=5)

#trainer.predict(dataset,
#            batch_size=32,
#            max_iterations=2)

#trainer.save('./checkpoints/checkpoint')

#trainer.load('./checkpoints/checkpoint.pth.tar')
