
# Scorch: utilities for network training with PyTorch

This package is a set of codes that may be reused quite often
for different networks trainings.

## Prerequisites

You will need openmpi installed before running or installing this package.
The easiest way to get the openmpi on linux machine without root is to use 
[LinuxBrew](http://linuxbrew.sh/).

After the LinuxBrew is installed, run this:
```
brew install openmpi
```

After the success you are ready to install this package.

It is also recommended that you install [PyTorch](https://pytorch.org/) for your
system before you install scorch.

Also you will need to install tensorflow in order to be able to use tensorboardX.

Both are easy to install using Conda.

## Installation
To install this package, use 
```
pip install git+http://github.com/RomanovMikeV/scorch
```
To use notebooks for testing the model and the dataset you
will need Jupyter Notebook or JupyterLab installed.

## Usage

Here is the minimal command to run to train the model specified in MODEL_FILE with a
dataset specified in DATASET_FILE, the data is located in the DATASET_PATH.

```
scorch-train --model MODEL_FILE --dataset DATASET_FILE --dataset-path DATASET_PATH
```

Here is a list of parameters of the script (it will be soon updated):

To see the full list of parameters, please use
```
scorch-train --help
```

This scripy relies on the Hororvod to parallelize the model training for several devices.
You can use mpi calls as follows (to run the script on 4 devices of the localhost in 
parallel):
```
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    scorch-train SCORCH_TRAIN_PARAMETERS
```

To run on several workstations, do as follows (16 processes in total,
4 on server1, 4 on server2, 4 on server3, 4 on server4):
```
mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    scorch-train SCORCH_TRAIN_PARAMETERS
```

## Model module syntax

The syntax for the model file is the following:

```python
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        pass
     
    def forward(self, input):
        return [output1, output2]
        
    def __call__(self, input):
        return self.forward(input)
    
class Socket:
    def __init__(self, model):
        self.model = model
    
    def criterion(self, pred, target):
        pass
    
    def metrics(self, pred, target):
        pass
```

The requirements are as follows:
* Network should have (obviously) the constructor
* Network should have forward function which:
  * Takes as input a **list** of inputs.
  * Outputs a **list** of outputs.
* There should be also the ```__call__``` function specified that is a proxy for the forward function.
* There should be a ```Socket``` class defined in order to specify how to handle the model, it should contain:
  * ```criterion``` method that takes as inputs a **list** of tensors with predictions and a **list** of tensors with targets. The output should be a number.
  * ```metrics``` method that specifies the metrics which are of the interest for your experiment. It should take as inputs a list of tensors with predictions and a list of tensors with targets and return a list of metrics which.
  * ```torch.nn.ModuleList``` of ```trainable_modules```. Only the modules that were specified here will be trained.

The reason there are lists everywhere is the following: the network may have more than one input and more than one output. We have to deal with this fact smart enough to reuse the code. Thus, the best way to do things is to pass the values of interests in lists.

## Dataset module syntax

Here is a syntax for the Dataset module:

```python
class DataSetIndex():
    def __init__(self, path):
        pass

class DataSet():
    def __init__(self, ds_index, mode='train'):
        self.ds_index = ds_index
    
    def __len__(self):
        if self.mode == 'test':
            pass
        
        elif self.mode == 'valid':
            pass
        
        else:
            pass
        
        
    def __getitem__(self, index):
        img = None
        target = None
        
        if self.mode == 'test':
            pass
        
        elif self.mode == 'valid':
            pass
        
        else:
            pass
        
        return [img1, img2], [target1, target2]
```

The dataset script should have at least the class DataSet which should have the following specified:

* ```__init__```, the constructor that defines all three parts of the dataset. The mode of the dataset should be defined here.
* ```__len__``` function that returns the length of the dataset
* ```__getitem__``` function that returns a list of input tensors and a list of target tensors

Although it is enough to have only the DataSet specified, it is recommended to specify also the DataSetIndex class that contains the information about the whole dataset. It is recommended to share one instance of the DataSetIndex between all the instances of the DataSet with different modes to avoid doubling or tripling the memory used to store this index and also to avoid collecting the dataset index several times.
