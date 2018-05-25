
# Utilities for network training with PyTorch

This package is a set of codes that may be reused quite often
for different networks trainings.

## Prerequisites

You will need Python 3 to use this package.

You will need the following packages installed:
* pytorch
* numpy

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

## Dataset module syntax

