import torch
import torchvision
import skimage.transform
import sys
import argparse
import time
import importlib.util

import torchvision.transforms as transform

sys.path.append('../src')

import trainer
import utils
import scripts

## Training parameters

parser = argparse.ArgumentParser(
    description='Script to train a specified model with a specified dataset.')
parser.add_argument('-b','--batch-size', help='Batch size', default=32, type=int)
parser.add_argument('-w','--workers', help='Number of workers in a dataloader', default=4, type=int)
parser.add_argument('--pretraining', help='Pretraining mode', action='store_true')
parser.add_argument('-lr', '--learning-rate', help='Learning rate', default=3.0e-4, type=float)
parser.add_argument('-d', '--dump-period', help='Dump period', default=1, type=int)
parser.add_argument('-e', '--epochs', help='Number of epochs to perform', default=1000, type=int)
parser.add_argument('-c', '--checkpoint', help='Checkpoint to load from', default=None)
parser.add_argument('--use-cuda', help='Use cuda for training', action='store_true')
parser.add_argument('--validate-on-train', help='Validate on train', action='store_true')
parser.add_argument('--model', help='File with a model specifications', required=True)
parser.add_argument('--dataset', help='File with a dataset sepcification', required=True)
parser.add_argument('--max-train-iterations', help='Maximum training iterations', default=-1, type=int)
parser.add_argument('--max-valid-iterations', help='Maximum validation iterations', default=-1, type=int)
parser.add_argument('-dp', '--dataset-path', help='Path to the dataset', required=True)
parser.add_argument('-v', '--verbosity', 
                    help='-1 for no output, 0 for epoch output, positive number is printout frequency', 
                    default=-1, type=int)
parser.add_argument('-cp', '--checkpoint-prefix', help='Prefix to the checkpoint name', default='')
args = vars(parser.parse_args())

scripts.train(args['model'], args['dataset'], args['dataset_path'],
              batch_size=args['batch_size'],
              num_workers=args['workers'],
              pretraining=args['pretraining'],
              learning_rate=args['learning_rate'],
              dump_period=args['dump_period'],
              epochs=args['epochs'],
              checkpoint=args['checkpoint'],
              use_cuda=args['use_cuda'],
              validate_on_train=args['validate_on_train'],
              max_train_iterations=args['max_train_iterations'],
              max_valid_iterations = args['max_valid_iterations'],
              verbosity = args['verbosity'],
              checkpoint_prefix = args['checkpoint_prefix'])
