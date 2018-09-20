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
import os

import torchvision.transforms as transform

from . import trainer
from . import utils


def test(model_source_path,
              dataset_source_path,
              dataset_path,
              batch_size=1,
              num_workers=0,
              checkpoint=None,
              use_cuda=False,
              max_test_iterations=-1,
              silent=False,
              dataset_kwargs={},
              model_kwargs={},
              seed=0):

    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Initializing Horovod
    hvd.init()

    # Enabling garbage collector
    gc.enable()

    # Importing a model from a specified file
    spec = importlib.util.spec_from_file_location("model", model_source_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    # Importing a dataset from a specified file
    spec = importlib.util.spec_from_file_location("dataset", dataset_source_path)
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    # Creating neural network instance
    net = model.Network(**model_kwargs)

    # Model CPU -> GPU
    if use_cuda:
        torch.cuda.set_device(hvd.local_rank())
        net = net.cuda()

    # Creating a Socket for a network
    socket = model.Socket(net)

    # Preparing Socket for Horovod
    hvd.broadcast_parameters(socket.model.state_dict(), root_rank=0)

    # Creating the datasets
    ds_index = dataset.DataSetIndex(dataset_path, **dataset_kwargs)

    test_dataset = dataset.DataSet(
        ds_index, mode='test')

    # Preparing the datasets for Horovod
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    ## Creating dataloaders based on datasets
    test_loader = scorch.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         drop_last=False,
                                         pin_memory=False,
                                         collate_fn=utils.default_collate,
                                         sampler=test_sampler)

    # Shut up if you're not the first process
    if hvd.rank() != 0:
        silent = True

    # Creating a trainer
    my_trainer = trainer.Trainer(socket,
                                 silent=silent,
                                 use_cuda=use_cuda,
                                 max_test_iterations=max_test_iterations)

    best_metrics = None

    if checkpoint is not None:
        my_trainer = trainer.load_from_checkpoint(checkpoint,
                                                  socket,
                                                  silent=silent,
                                                  use_cuda=use_cuda,
                                                  max_train_iterations=max_train_iterations,
                                                  max_valid_iterations=max_valid_iterations,
                                                  max_test_iterations=max_test_iterations,
                                                  new_optimizer=new_optimizer)
        best_metrics = my_trainer.metrics

    if not os.path.exists('results'):
        os.mkdir('results')

    batch_index = 0
    for input, output, test_id in my_trainer.test(test_loader):
        result = {}
        for index in range(len(test_id)):
            result[test_id[index]] = socket.process_result(input[index], output[index])

        torch.save(result,
            os.path.join('results',
                str(batch_index) + '_' + str(hvd.rank()) + '.pth.tar'))

        batch_index += 1


    gc.collect()


def testing():

    ## Training parameters

    parser = argparse.ArgumentParser(
        description='Script to train a specified model with a specified dataset.')
    parser.add_argument('-b','--batch-size', help='Batch size', default=1, type=int)
    parser.add_argument('-w','--workers', help='Number of workers in a dataloader', default=0, type=int)
    parser.add_argument('-c', '--checkpoint', help='Checkpoint to load from', default=None)
    parser.add_argument('--use-cuda', help='Use cuda for training', action='store_true')
    parser.add_argument('--model', help='File with a model specifications', required=True)
    parser.add_argument('--dataset', help='File with a dataset sepcification', required=True)
    parser.add_argument('--max-test-iterations', help='Maximum test iterations', default=-1, type=int)
    parser.add_argument('-dp', '--dataset-path', help='Path to the dataset', required=True)
    parser.add_argument('-s', '--silent', help='do not print the status of learning',
                        action='store_true')
    parser.add_argument('--model-args', help='Model arguments which will be used during training', default='', type=str)
    parser.add_argument('--dataset-args', help='Dataset arguments which will be used during training', default='', type=str)
    parser.add_argument('--seed', help='Seed for random number generators', default=0, type=int)
    args = vars(parser.parse_args())

    ## Calling inference function

    test(args['model'], args['dataset'], args['dataset_path'],
              batch_size=args['batch_size'],
              num_workers=args['workers'],
              checkpoint=args['checkpoint'],
              use_cuda=args['use_cuda'],
              max_test_iterations=args['max_test_iterations'],
              silent=args['silent'],
              dataset_kwargs=eval('{' + args['dataset_args'] + '}'),
              model_kwargs=eval('{' + args['model_args'] + '}'),
              seed=args['seed'])

if __name__ == '__main__':
    testing()
