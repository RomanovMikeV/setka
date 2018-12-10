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

from . import internal
from . import trainer

def train(NetworkClass,
          SocketClass,
          DataSetClass,
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
          seed=0,
          solo_test=False,
          deterministic_cuda=False):

    #numpy.random.seed(seed)
    #random.seed(seed)
    #torch.manual_seed(seed)

    if torch.cuda.is_available:
    #    torch.cuda.manual_seed(seed)
    #    torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.deterministic = False

    # Initializing Horovod
    hvd.init()

    # Enabling garbage collector
    gc.enable()

    # Creating tensorboard writer
    if hvd.rank() == 0:
        tb_writer = SummaryWriter()

    # Creating neural network instance
    net = NetworkClass(**model_kwargs)

    # Model CPU -> GPU
    if use_cuda:
        torch.cuda.set_device(hvd.local_rank())
        net = net.cuda()

    # Creating a Socket for a network
    socket = SocketClass(net)

    # Preparing Socket for Horovod training
    for opt_index in range(len(socket.optimizers)):
        socket.optimizers[opt_index].optimizer = hvd.DistributedOptimizer(
            socket.optimizers[opt_index].optimizer)

    hvd.broadcast_parameters(socket.model.state_dict(), root_rank=0)

    # Creating the datasets
    ds_index = DataSetClass(**dataset_kwargs)

    train_dataset = internal.DataSetWrapper(
        ds_index, mode='train')

    valid_dataset = internal.DataSetWrapper(
        ds_index, mode='valid')

    test_dataset = internal.DataSetWrapper(
        ds_index, mode='test')

    # Preparing the datasets for Horovod
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    if solo_test:
        test_batch_size = 1
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    else:
        test_batch_size = batch_size
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    ## Creating dataloaders based on datasets
    train_loader = scorch.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=False,
                                               collate_fn=internal.default_collate,
                                               sampler=train_sampler)

    valid_loader = scorch.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               pin_memory=False,
                                               collate_fn=internal.default_collate,
                                               sampler=valid_sampler)

    test_loader = scorch.data.DataLoader(test_dataset,
                                               batch_size=test_batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               pin_memory=False,
                                               collate_fn=internal.default_collate,
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
        best_metrics = my_trainer.socket.metrics_valid

    # Validation before training

    # Validation on training subuset
    if validate_on_train:
        train_metrics = my_trainer.validate(train_loader)
        my_trainer.socket.metrics_train = train_metrics
        gc.collect()

    # Validation on validation subset
    valid_metrics = my_trainer.validate(valid_loader)
    gc.collect()
    my_trainer.socket.metrics_valid = valid_metrics

    if hasattr(socket, 'scheduling'):
        socket.scheduling()

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
            my_trainer.socket.metrics_train = train_metrics
            gc.collect()

        # Validation on validation subset
        valid_metrics = my_trainer.validate(valid_loader)
        gc.collect()
        my_trainer.socket.metrics_valid = valid_metrics

        # Updating Learning Rate if needed

        if hasattr(socket, 'scheduling'):
            socket.scheduling()

        gc.collect()


        # Processing resulting metrics
        for metric_name in valid_metrics:
            metric_data = {'valid': valid_metrics[metric_name]}

            if metric_name in train_metrics:
                metric_data['train'] = train_metrics[metric_name]

            if hvd.rank() == 0:
                tb_writer.add_scalars(
                    'metrics/' + metric_name + '/' + checkpoint_prefix,
                    metric_data, my_trainer.socket.epoch)


            # Running tests for visualization
        if hvd.rank() == 0:
            inputs = []
            outputs = []
            test_ids = []


        for input, output, test_id in my_trainer.test(test_loader,
                                                      solo_test=solo_test):
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

        if hvd.rank() == 0:
            for index in range(len(inputs)):
                inputs[index] = torch.cat(inputs[index], dim=0)

            for index in range(len(outputs)):
                outputs[index] = torch.cat(outputs[index], dim=0)

            for index in range(len(test_ids)):
                one_input = []
                one_output = []

                for input_index in range(len(inputs)):
                    one_input.append(inputs[input_index][index])

                for output_index in range(len(outputs)):
                    one_output.append(outputs[output_index][index])

                test_id = test_ids[index]

                if hasattr(my_trainer.socket, 'visualize'):
                    internal.show(
                        tb_writer,
                        my_trainer.socket.visualize(
                            one_input, one_output, test_id),
                        my_trainer.socket.epoch)

                #except AttributeError:
                #    pass
                    # if hvd.rank() == 0:
                        # print('Visualization is not implemented. Skipping.')

            del inputs
            del outputs



        gc.collect()

        # Dumping the model
        if epoch_index % dump_period == 0 and hvd.rank() == 0:
            is_best = False
            if best_metrics is None:
                is_best = True
            else:
                if best_metrics['main'] < my_trainer.socket.metrics_valid['main']:
                    is_best = True
            if is_best:
                best_metrics = my_trainer.socket.metrics_valid

            my_trainer.make_checkpoint(checkpoint_prefix,
                                       info=valid_metrics,
                                       is_best=is_best)
        gc.collect()

    return my_trainer.socket.metrics_valid


def test(NetworkClass,
         SocketClass,
         DataSetClass,
         batch_size=1,
         num_workers=0,
         checkpoint=None,
         use_cuda=False,
         max_test_iterations=-1,
         silent=False,
         dataset_kwargs={},
         model_kwargs={},
         seed=0,
         deterministic_cuda=False,
         prefix=''):

    if len(prefix) > 0:
        prefix += '_'

    #numpy.random.seed(seed)
    #random.seed(seed)
    #torch.manual_seed(seed)

    if torch.cuda.is_available:
        #torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.deterministic = False

    # Initializing Horovod
    hvd.init()

    # Enabling garbage collector
    gc.enable()

    # Creating neural network instance
    net = NetworkClass(**model_kwargs)

    # Model CPU -> GPU
    if use_cuda:
        torch.cuda.set_device(hvd.local_rank())
        net = net.cuda()

    # Creating a Socket for a network
    socket = SocketClass(net)

    # Preparing Socket for Horovod
    hvd.broadcast_parameters(socket.model.state_dict(), root_rank=0)

    # Creating the datasets
    ds_index = DataSetClass(**dataset_kwargs)

    test_dataset = internal.DataSetWrapper(
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
                                         collate_fn=internal.default_collate,
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
                                                  use_cuda=use_cuda)
        best_metrics = my_trainer.socket.metrics

    if not os.path.exists(prefix + 'results'):
        os.mkdir(prefix + 'results')

    batch_index = 0

    for inputs, outputs, test_ids in my_trainer.test(test_loader):
        result = {}

        for index in range(len(test_ids)):
            one_input = []
            one_output = []
            test_id = test_ids[index]

            for input_index in range(len(inputs)):
                one_input.append(inputs[input_index][index])

            for output_index in range(len(outputs)):
                one_output.append(outputs[output_index][index])

            if hasattr(socket, 'process_result'):
                result[test_id] = socket.process_result(one_input,
                                                        one_output,
                                                        test_id)
            else:
                result[test_id] = one_output

        torch.save(result,
            os.path.join(prefix + 'results',
                str(hvd.rank()) + '_' + str(batch_index) + '.pth.tar'))

        batch_index += 1


    gc.collect()
