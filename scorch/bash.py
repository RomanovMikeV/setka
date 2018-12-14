import torch
import torch.utils.data.distributed
import torchvision
import skimage.transform
import sys
import argparse
import time
import importlib.util
import horovod.torch as hvd
import gc
from tensorboardX import SummaryWriter
import tensorboardX
import scorch.data
import numpy
import random

import torchvision.transforms as transform

from . import utils


def training():

    ## Training parameters

    parser = argparse.ArgumentParser(
        description='Script to train a specified model with a specified dataset.')
    parser.add_argument('-b','--batch-size',
                        help='Batch size', default=8, type=int)
    parser.add_argument('-w','--workers',
                        help='Number of workers in a dataloader',
                        default=0, type=int)
    parser.add_argument('-d', '--dump-period',
                        help='Dump period', default=1, type=int)
    parser.add_argument('-e', '--epochs',
                        help='Number of epochs to perform',
                        default=1000, type=int)
    parser.add_argument('-c', '--checkpoint',
                        help='Checkpoint to load from', default=None)
    parser.add_argument('--use-cuda',
                        help='Use cuda for training',
                        action='store_true')
    parser.add_argument('--validate-on-train',
                        help='Validate on train',
                        action='store_true')
    parser.add_argument('--model',
                        help='File with a model specifications',
                        required=True)
    parser.add_argument('--dataset',
                        help='File with a dataset specification',
                        required=True)
    parser.add_argument('--trainer',
                        help='File with trainer specification',
                        required=True)
    parser.add_argument('--max-train-iterations',
                        help='Maximum training iterations',
                        default=None, type=int)
    parser.add_argument('--max-valid-iterations',
                        help='Maximum validation iterations',
                        default=None, type=int)
    parser.add_argument('--max-test-iterations',
                        help='Maximum test iterations',
                        default=None, type=int)
    parser.add_argument('-s', '--silent',
                        help='do not print the status of learning',
                        action='store_true')
    parser.add_argument('-cn', '--checkpoint-name',
                        help='Prefix to the checkpoint name',
                        default='checkpoint')
    parser.add_argument(
        '--model-args',
        help=('Model arguments which will be used during training. ' +
              "The syntax is the same as the syntax of the python's dicts except for braces. " +
              "Example: --model-args \"'hidden_neurons':20\". " +
              "Note that here you may specify keyword parameters of your " +
              "dataset specified in DATASET_FILE"), default='', type=str)
    parser.add_argument(
        '--dataset-args',
        help=('Dataset arguments which will be used during training' +
              'Syntax is the same as the syntax of the model arguments.' +
              "Note that here you may specify keyword parameters of your " +
              "model specified in MODEL_FILE"), default='', type=str)
    parser.add_argument('--new-optimizer',
                        help='Use new optimizer when loading from the checkpoint',
                        action='store_true')
    parser.add_argument('--seed',
                        help='Seed for random number generators',
                        default=0, type=int)
    parser.add_argument('--solo-test',
                        help='This argument switches test to the mode' +
                        'when the test is performed on one device with' +
                        'batch_size=1',
                        action='store_true')
    parser.add_argument('--deterministic-cuda',
                        help='Use deterministic CUDA backend (slower by ~10\%)',
                        action='store_true')
    args = vars(parser.parse_args())

    # Importing a model from a specified file
    spec = importlib.util.spec_from_file_location("model", args['model'])
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    # Importing a dataset from a specified file
    spec = importlib.util.spec_from_file_location("dataset", args['dataset'])
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    # Importing a trainer from a specified file
    spec = importlib.util.spec_from_file_location("trainer", args['trainer'])
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    ## Calling training function

    model = model.Network()
    dataset = dataset.DataSet()
    trainer = trainer.Trainer(model,
                              use_cuda=args['use_cuda'],
                              seed=args['seed'],
                              deterministic_cuda=args['deterministic_cuda'])

    if args['checkpoint'] != None:
        trainer.load(args['checkpoint'])


    trainer.train(
              dataset,
              batch_size=args['batch_size'],
              num_workers=args['workers'],
              dump_period=args['dump_period'],
              epochs=args['epochs'],
              validate_on_train=args['validate_on_train'],
              max_train_iterations=args['max_train_iterations'],
              max_valid_iterations=args['max_valid_iterations'],
              max_test_iterations=args['max_test_iterations'],
              silent=args['silent'],
              checkpoint_name=args['checkpoint_name'],
              solo_test=args['solo_test'])


def testing():

    ## Training parameters

    parser = argparse.ArgumentParser(
        description='Script to train a specified model with a specified dataset.')
    parser.add_argument('-b','--batch-size', help='Batch size', default=1, type=int)
    parser.add_argument('-w','--workers', help='Number of workers in a dataloader', default=0, type=int)
    parser.add_argument('-c', '--checkpoint', help='Checkpoint to load from', default=None)
    parser.add_argument('--use-cuda', help='Use cuda for training', action='store_true')
    parser.add_argument('--model', help='File with a model specifications', required=True)
    parser.add_argument('--dataset', help='File with a dataset specification', required=True)
    parser.add_argument('--trainer', help='File with trainer specification', required=True)
    parser.add_argument('--max-test-iterations', help='Maximum test iterations', default=-1, type=int)
    parser.add_argument('-s', '--silent', help='do not print the status of learning',
                        action='store_true')
    parser.add_argument('--model-args', help='Model arguments which will be used during training', default='', type=str)
    parser.add_argument('--dataset-args', help='Dataset arguments which will be used during training', default='', type=str)
    parser.add_argument('--seed', help='Seed for random number generators', default=0, type=int)
    parser.add_argument('--deterministic-cuda', help='Use deterministic cuda backend', action='store_true')
    parser.add_argument('--prefix', help='Prefix for the results folder', default='')
    args = vars(parser.parse_args())

    # Importing a model from a specified file
    spec = importlib.util.spec_from_file_location("model", args['model'])
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    # Importing a dataset from a specified file
    spec = importlib.util.spec_from_file_location("dataset", args['dataset'])
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    # Importing a trainer from a specified file
    spec = importlib.util.spec_from_file_location("trainer", args['trainer'])
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    ## Calling inference function

    model = model.Network()
    dataset = dataset.DataSet()
    trainer = trainer.Trainer(model)

    if args['checkpoint'] != None:
        trainer.load(args['checkpoint'])

    trainer.predict(dataset,
                 batch_size=args['batch_size'],
                 num_workers=args['workers'],
                 max_iterations=args['max_test_iterations'],
                 silent=args['silent'],
                 prefix=args['prefix'])
