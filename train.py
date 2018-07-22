import torch
import torchvision
import skimage.transform
import sys
import argparse
import time
import importlib.util

import torchvision.transforms as transform

import trainer
import utils


def train(model_source_path,
          dataset_source_path,
          dataset_path,
          batch_size=32,
          num_workers=4,
          pretraining=False,
          learning_rate=3.0e-4,
          dump_period=1,
          epochs=1000,
          checkpoint=None,
          use_cuda=False,
          validate_on_train=False,
          max_train_iterations=-1,
          max_valid_iterations=-1,
          verbosity=-1,
          checkpoint_prefix=''):

    spec = importlib.util.spec_from_file_location("model", model_source_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    spec = importlib.util.spec_from_file_location("dataset", dataset_source_path)
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    ## Creating neural network and a socket for it

    net = model.Network()
    net = torch.nn.parallel.DataParallel(net).eval()

    if use_cuda:
        net = net.cuda()

    socket = model.Socket(net)


    # Creating the datasets

    ds_index = dataset.DataSetIndex(dataset_path)
    train_dataset = dataset.DataSet(
        ds_index, mode='train')

    valid_dataset = dataset.DataSet(
        ds_index, mode='valid')

    ## Creating dataloaders from the datasets

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=False,
                                               collate_fn=utils.default_collate)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               pin_memory=False,
                                               collate_fn=utils.default_collate)



    my_trainer = trainer.Trainer(socket,
                                 verbosity=verbosity,
                                 use_cuda=use_cuda,
                                 max_train_iterations=max_train_iterations,
                                 max_valid_iterations=max_valid_iterations)
    best_metrics = None

    if checkpoint is not None:
        my_trainer = trainer.load_from_checkpoint(checkpoint,
                                                  socket,
                                                  verbosity=verbosity,
                                                  use_cuda=use_cuda,
                                                  max_train_iterations=max_train_iterations,
                                                  max_valid_iterations=max_valid_iterations)
        best_metrics = my_trainer.metrics


    for index in range(epochs):
        loss = my_trainer.train(train_loader)
        metrics = []
        if validate_on_train:
            for metric in my_trainer.validate(train_loader):
                metrics.append(metric)

        for metric in my_trainer.validate(valid_loader):
            metrics.append(metric)

        if index % dump_period == 0:
            is_best = False
            if best_metrics == None:
                is_best = True
            else:
                if best_metrics[0] < my_trainer.metrics[0]:
                    is_best = True
            if is_best:
                best_metrics = my_trainer.metrics

            my_trainer.make_checkpoint(checkpoint_prefix, info=metrics, is_best=is_best)


if __name__ == '__main__':

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

    ## Calling training function

    train(args['model'], args['dataset'], args['dataset_path'],
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
