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

from . import trainer
from . import utils

def show(tb_writer, to_show, epoch):

    type_writers = {
        'images': tb_writer.add_image,
        'texts': tb_writer.add_text,
        'audios': tb_writer.add_audio,
        'figures': (lambda x, y, z: tb_writer.add_figure(x, y, z)),
        'graphs': tb_writer.add_graph,
        'embeddings': tb_writer.add_embedding}

    for type in type_writers:
        if type in to_show:
            for desc in to_show[type]:
                type_writers[type](desc, to_show[type][desc], str(epoch))


def train(model_source_path,
          dataset_source_path,
          batch_size=4,
          num_workers=0,
          dump_period=1,
          epochs=1000,
          checkpoint=None,
          use_cuda=False,
          validate_on_train=False,
          max_train_iterations=-1,
          max_valid_iterations=-1,
          max_test_iterations=-1,
          silent=False,
          checkpoint_prefix='',
          dataset_kwargs={},
          model_kwargs={},
          new_optimizer=False,
          seed=0):

    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # Initializing Horovod
    hvd.init()

    # Enabling garbage collector
    gc.enable()

    # Creating tensorboard writer
    if hvd.rank() == 0:
        tb_writer = SummaryWriter()

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

    # Preparing Socket for Horovod training
    socket.optimizer = hvd.DistributedOptimizer(
        socket.optimizer,
        named_parameters=socket.model.named_parameters())
    hvd.broadcast_parameters(socket.model.state_dict(), root_rank=0)

    # Creating the datasets
    ds_index = dataset.DataSetIndex(**dataset_kwargs)

    train_dataset = dataset.DataSet(
        ds_index, mode='train')

    valid_dataset = dataset.DataSet(
        ds_index, mode='valid')

    test_dataset = dataset.DataSet(
        ds_index, mode='test')

    # Preparing the datasets for Horovod
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    ## Creating dataloaders based on datasets
    train_loader = scorch.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=False,
                                               collate_fn=utils.default_collate,
                                               sampler=train_sampler)

    valid_loader = scorch.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               pin_memory=False,
                                               collate_fn=utils.default_collate,
                                               sampler=valid_sampler)

    test_loader = scorch.data.DataLoader(test_dataset,
                                               batch_size=1,
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
                                 max_train_iterations=max_train_iterations,
                                 max_valid_iterations=max_valid_iterations,
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

    # Training cycle
    for epoch_index in range(epochs):

        # Training
        gc.enable()
        loss = my_trainer.train(train_loader)
        gc.collect()

        train_metrics = {}

        # Validation on training subuset
        if validate_on_train:
            train_metrics = my_trainer.validate(train_loader)
            gc.collect()

        # Validation on validation subset
        valid_metrics = my_trainer.validate(valid_loader)
        gc.collect()

        # Processing resulting metrics
        for metric_name in valid_metrics:
            metric_data = {'valid': valid_metrics[metric_name]}

            if metric_name in train_metrics:
                metric_data['train'] = train_metrics[metric_name]

            if hvd.rank() == 0:
                tb_writer.add_scalars(
                    'metrics/' + metric_name + '/' + checkpoint_prefix,
                    metric_data, my_trainer.epoch)


            # Running tests for visualization
        if hvd.rank() == 0:
            inputs = []
            outputs = []
            test_ids = []

        for input, output, test_id in my_trainer.test(test_loader):
            if hvd.rank() == 0:
                for item_index in range(len(test_id)):
                    for input_index in range(len(input)):
                        if len(inputs) <= input_index:
                            inputs.append([])
                        inputs[input_index].append(
                            input[input_index][item_index].unsqueeze(0))

                    for output_index in range(len(output)):
                        if len(outputs) <= output_index:
                            outputs.append([])
                        outputs[output_index].append(
                            output[output_index][item_index].unsqueeze(0))

                    test_ids.append(test_id[item_index])


        for index in range(len(inputs)):
            inputs[index] = torch.cat(inputs[index], dim=0)

        for index in range(len(outputs)):
            outputs[index] = torch.cat(outputs[index], dim=0)

        try:
            show(tb_writer,
                 my_trainer.socket.visualize(inputs, outputs, test_ids),
                 my_trainer.epoch)

        except AttributeError:
            if hvd.rank() == 0:
                print('Visualization is not implemented. Skipping.')



        gc.collect()

        # Updating Learning Rate if needed
        if socket.scheduler is not None and 'main' in valid_metrics:
            socket.scheduler.step(valid_metrics['main'])

        gc.collect()

        # Dumping the model
        if epoch_index % dump_period == 0 and hvd.rank() == 0:
            is_best = False
            if best_metrics == None:
                is_best = True
            else:
                if best_metrics['main'] < my_trainer.metrics['main']:
                    is_best = True
            if is_best:
                best_metrics = my_trainer.metrics

            my_trainer.make_checkpoint(checkpoint_prefix,
                                       info=valid_metrics,
                                       is_best=is_best)
        gc.collect()


def training():

    ## Training parameters

    parser = argparse.ArgumentParser(
        description='Script to train a specified model with a specified dataset.')
    parser.add_argument('-b','--batch-size', help='Batch size', default=8, type=int)
    parser.add_argument('-w','--workers', help='Number of workers in a dataloader', default=0, type=int)
    parser.add_argument('-d', '--dump-period', help='Dump period', default=1, type=int)
    parser.add_argument('-e', '--epochs', help='Number of epochs to perform', default=1000, type=int)
    parser.add_argument('-c', '--checkpoint', help='Checkpoint to load from', default=None)
    parser.add_argument('--use-cuda', help='Use cuda for training', action='store_true')
    parser.add_argument('--validate-on-train', help='Validate on train', action='store_true')
    parser.add_argument('--model', help='File with a model specifications', required=True)
    parser.add_argument('--dataset', help='File with a dataset sepcification', required=True)
    parser.add_argument('--max-train-iterations', help='Maximum training iterations', default=-1, type=int)
    parser.add_argument('--max-valid-iterations', help='Maximum validation iterations', default=-1, type=int)
    parser.add_argument('--max-test-iterations', help='Maximum test iterations', default=-1, type=int)
    parser.add_argument('-s', '--silent', help='do not print the status of learning',
                        action='store_true')
    parser.add_argument('-cp', '--checkpoint-prefix', help='Prefix to the checkpoint name', default='')
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
    parser.add_argument('--new-optimizer', help='Use new optimizer when loading from the checkpoint', action='store_true')
    parser.add_argument('--seed', help='Seed for random number generators', default=0, type=int)
    args = vars(parser.parse_args())

    ## Calling training function

    train(args['model'], args['dataset'],
              batch_size=args['batch_size'],
              num_workers=args['workers'],
              dump_period=args['dump_period'],
              epochs=args['epochs'],
              checkpoint=args['checkpoint'],
              use_cuda=args['use_cuda'],
              validate_on_train=args['validate_on_train'],
              max_train_iterations=args['max_train_iterations'],
              max_valid_iterations=args['max_valid_iterations'],
              max_test_iterations=args['max_test_iterations'],
              silent=args['silent'],
              checkpoint_prefix=args['checkpoint_prefix'],
              dataset_kwargs=eval('{' + args['dataset_args'] + '}'),
              model_kwargs=eval('{' + args['model_args'] + '}'),
              new_optimizer=args['new_optimizer'],
              seed=args['seed'])

if __name__ == '__main__':
    training()
