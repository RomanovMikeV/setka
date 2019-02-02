[![Build Status](https://travis-ci.com/RomanovMikeV/scorch.svg?branch=master)](https://travis-ci.com/RomanovMikeV/scorch)

# Scorch: utilities for network training with PyTorch

Scorch is a powerful and flexible tool for neural network training and fast
prototyping.

## Prerequisites

Before using scorch, please install:

[PyTorch](https://pytorch.org/) for your
system.

tensorflow  with
```
conda install tensorflow
```

tensorboardX with
```
pip install tensorboardX
```

## Installation
To install this package, use
```
pip install git+http://github.com/RomanovMikeV/scorch
```

## Usage

Below is comparison with Keras:

Model training with Scorch:
```python
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

        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.conv4 = torch.nn.Conv2d(256, 512, 3, padding=2)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.pool4 = torch.nn.MaxPool2d(2)
        self.conv5 = torch.nn.Conv2d(512, 1024, 3, padding=2)
        self.bn5 = torch.nn.BatchNorm2d(1024)
        self.pool5 = torch.nn.MaxPool2d(2)

        self.bn_last = torch.nn.BatchNorm1d(1024)
        self.fc = torch.nn.Linear(1024, 10)

    def forward(self, input):
        res = input[0]

        res = self.pool1(self.bn1(torch.relu(self.conv1(res))))
        res = self.pool2(self.bn2(torch.relu(self.conv2(res))))
        res = self.pool3(self.bn3(torch.relu(self.conv3(res))))
        res = self.pool4(self.bn4(torch.relu(self.conv4(res))))
        res = self.pool5(self.bn5(torch.relu(self.conv5(res))))

        res = res.mean(dim=3).mean(dim=2)

        res = self.fc(self.bn_last(res))

        return [res]


def criterion(preds, target):
    return torch.nn.CrossEntropyLoss()(preds[0], target[0][:, 0])

def accuracy(preds, target):
    return (preds[0].argmax(dim=1) == target[0][:, 0]).float().mean()

net = Network()
trainer = scorch.base.Trainer(net,
                  optimizers=[
                      scorch.base.OptimizerSwitch(net, torch.optim.Adam, lr=3.0e-4)],
                  callbacks=[
                    scorch.callbacks.ComputeMetrics(
                        metrics={'main': accuracy, 'loss': criterion}),
                    scorch.callbacks.MakeCheckpoints(),
                    scorch.callbacks.SaveResult(),
                    scorch.callbacks.WriteToTensorboard()],
                  criterion=criterion,
                  use_cuda=False, silent=False)
dataset = DataSet()

start = time.time()
trainer.train_one_epoch(dataset, batch_size=32, num_workers=4)
print('One epoch time:', time.time() - start)

trainer.validate_one_epoch(dataset, batch_size=32)

print('One epoch loss:', trainer._loss.item())
print('One epoch accuracy:', trainer._valid_metrics['main'].item())


trainer.train(dataset, batch_size=32,
             num_workers=4,
             epochs=2)

print('3 epochs loss:', trainer._loss.item())
print('3 epochs accuracy:', trainer._valid_metrics['main'].item())
```

Keras model training:
```python
model = Sequential()

model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(512, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(1024, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=3.0e-4),
              metrics=['accuracy'])

start = time.time()

model.fit(x_train, labels_train, epochs=1, batch_size=32, verbose=1)
print("Epoch time:", time.time() - start)

scores = model.evaluate(x_test, labels_test, verbose=0)
print('One epoch loss:', scores[0])
print('One epoch acc :', scores[1])

keras_one_epoch_loss = scores[0]
keras_one_epoch_acc = scores[1]

res = model.fit(x_train, labels_train, epochs=2, batch_size=32, verbose=1)

scores = model.evaluate(x_test, labels_test, verbose=1)
keras_3_epochs_loss = scores[0]
keras_3_epochs_acc = scores[1]
print('3 epochs loss:', scores[0])
print('3 epochs acc :', scores[1])
```


## Bash interfaces

Soon will be available


## Defining the Model

Network should be defined as follows:

```python
class Network(scorch.base.Network):
    def __init__(self):
        super().__init__()

        # Layers for your network should be defined here.

    def forward(self, input):

        # Your code here, compute res1 and res2.

        return [res1, res2]
```


## Defining the DataSet

Dataset should be defined as follows:
```python
class DataSet(scorch.base.DataSet):
    def __init__(self):
        '''
        Definition of your dataset elements
        '''
        pass

    def get_len(self, mode='train'):
        '''
        Function to get length of the subset specified with 'mode'.
        '''
        return len(self.data[mode])

    def get_item(self, index, mode='train'):
        '''
        Function to get an item from from the subset of the dataset specified
        with 'mode'.
        '''
        return [datum], [target], id
```

## Callbacks engine

You can use as many callbacks available in callbacks collection or you may
specify your own callbacks to extend the functionality of the package.
