
# Utilities for network training with PyTorch

This package is a set of codes that may be reused quite often
for different networks trainings.

## Prerequisites

You will need Python 3 to use this package.

You will need the following packages installed:

* pytorch
* numpy
* scikit-image
* scikit-learn

To use the notebooks for testing the model and the dataset you
will need Jupyter Notebook or JupyterLab installed.

## Usage

Here is the minimal command to run to train the model specified in MODEL_FILE with a
dataset specified in DATASET_FILE, the data is located in the DATASET_PATH.

```
python train.py --model MODEL_FILE --dataset DATASET_FILE --dataset-path DATASET_PATH
```

Here is a list of parameters of the script (it will be soon updated):

```
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size to train or validate your model
  -w WORKERS, --workers WORKERS
                        Number of workers in a dataloader
  --pretraining         Pretraining mode
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -d DUMP_PERIOD, --dump-period DUMP_PERIOD
                        Dump period
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to perform
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Checkpoint to load from
  --use-cuda            Use cuda for training
  --validate-on-train   Flag showing that you want to perform validation on training dataset 
                        along with the validation on the validation set
                        
  --model MODEL         File with a model specification
  --dataset DATASET     File with a dataset sepcification
  --max-train-iterations MAX_TRAIN_ITERATIONS
                        Maximum training iterations
  --max-valid-iterations MAX_VALID_ITERATIONS
                        Maximum validation iterations
  -dp DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset
  -v VERBOSITY, --verbosity VERBOSITY
                        -1 for no output, 0 for epoch output, positive number
                        is printout frequency during the training
  -cp CHECKPOINT_PREFIX, --checkpoint-prefix CHECKPOINT_PREFIX
                        Prefix to the checkpoint name

```

## Model module syntax

The syntax for the model file is the following:

```
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

The reason there are lists everywhere is the following: the network may have more than one input and more than one output. We have to deal with this fact smart enough to reuse the code. Thus, the best way to do things is to pass the values of interests in lists.

## Dataset module syntax

Here is a syntax for the Dataset module:

```
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
        
        return img, target
```

The dataset script should have at least the class DataSet which should have the following specified:

* ```__init__```, the constructor that defines all three parts of the dataset. The mode of the dataset should be defined here.
* ```__len__``` function that returns the length of the dataset
* ```__getitem__``` function that returns a list of input tensors and a list of target tensors

Although it is enough to have only the DataSet specified, it is recommended to specify also the DataSetIndex class that contains the information about the dataset's data. It is recommended to share one instance of the DataSetIndex between all the instances of the DataSet with different modes to avoid doubling or tripling the memory used to store this index and also to avoid collecting the dataset index several times.
