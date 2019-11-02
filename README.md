Master branch:
[![Build Status](https://travis-ci.com/RomanovMikeV/setka.svg?branch=master)](https://travis-ci.com/RomanovMikeV/setka)
[![codecov](https://codecov.io/gh/RomanovMikeV/setka/branch/master/graph/badge.svg)](https://codecov.io/gh/RomanovMikeV/setka)

Dev branch:
[![Build Status](https://travis-ci.com/RomanovMikeV/setka.svg?branch=dev)](https://travis-ci.com/RomanovMikeV/setka)
[![codecov](https://codecov.io/gh/RomanovMikeV/setka/branch/dev/graph/badge.svg)](https://codecov.io/gh/RomanovMikeV/setka)



# setka: utilities for network training with PyTorch

setka is a powerful and flexible tool for neural network training and fast
prototyping.

## Installation
To install this package, use
```
pip install git+http://github.com/RomanovMikeV/setka
```

## Usage

Model training with setka:
```python

# Define your dataset
class Iris(setka.base.DataSet):
    def __init__(self, valid_split=0.1, test_split=0.1):
        super()
        data = sklearn.datasets.load_iris()

        X = data['data']
        y = data['target']

        n_valid = int(y.size * valid_split)
        n_test =  int(y.size * test_split)

        order = numpy.random.permutation(y.size)

        X = X[order, :]
        y = y[order]

        self.data = {
            'valid': X[:n_valid, :],
            'test': X[n_valid:n_valid + n_test, :],
            'train': X[n_valid + n_test:, :]
        }

        self.targets = {
            'valid': y[:n_valid],
            'test': y[n_valid:n_valid + n_test],
            'train': y[n_valid + n_test:]
        }

    # Define a method that gets the length of the subset by subset keyword
    def getlen(self, subset):
        return len(self.targets[subset])

    # Define a method that gets the item by the subset keyword and index of the item
    def getitem(self, subset, index):
        features = torch.Tensor(self.data[subset][index])
        class_id = torch.Tensor(self.targets[subset][index:index+1])
	
	# Method should return three items: list of features, list of targets and string ID of a sample
        return [features], [class_id], subset + "_" + str(index)


# Define your network (in the same way as it is done in PyTorch)
class IrisNet(setka.base.Network):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 100)
        self.fc2 = torch.nn.Linear(100, 3)

    def forward(self, x):
	# Network should return a list of outputs
        return [self.fc2(self.fc1(x[0]))]


ds = Iris()
model = IrisNet()

# Define your loss
def loss(pred, targ):
    return torch.nn.functional.cross_entropy(pred[0], targ[0][:, 0].long())

# Define your metrics
def accuracy(pred, targ):
    predicted = pred[0].argmax(dim=1)
    return (predicted == targ[0][:, 0].long()).sum(), predicted.numel()

# Define your trainer
trainer = setka.base.Trainer(model,
                              optimizers=[setka.base.OptimizerSwitch(model, torch.optim.Adam, lr=3.0e-3)],
                              criterion=loss,
                              callbacks=[
                                setka.callbacks.ComputeMetrics(metrics=[loss, accuracy]),
                                setka.callbacks.ReduceLROnPlateau(metric='loss'),
                                setka.callbacks.ExponentialWeightAveraging(),
                                setka.callbacks.WriteToTensorboard(name=name),
                                setka.callbacks.Logger(name=name)
                              ])

# You are ready to train
for index in range(100):
    trainer.train_one_epoch(ds, subset='train')
    trainer.validate_one_epoch(ds, subset='train')
    trainer.validate_one_epoch(ds, subset='valid')

trainer.predict_one_epoch(ds, subset='test')
```

## Defining the Model

Network should be defined as follows:

```python
class Network(setka.base.Network):
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
class DataSet(setka.base.DataSet):
    def __init__(self):
        '''
        Definition of your dataset elements
        '''
        pass

    def getlen(self, subset):
        '''
        Function to get length of the subset specified with 'mode'.
        '''
        return len(self.data[subset])

    def getitem(self, subset, index):
        '''
        Function to get an item from from the subset of the dataset specified
        with 'mode'.
        '''
        return [datum], [target], id
```

## Callbacks engine

You can use as many callbacks available in callbacks collection or you may
specify your own callbacks to extend the functionality of the package.
