import numpy
import os
import random
import torch
import shutil
#import horovod.torch as hvd
import gc
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm

from . import internal
from . import data


class OptimizerSwitch():
    '''
    This class contains optimizer and the module that this optimizer should
    optimize.
    '''
    def __init__(self, train_module, optimizer, is_active=True, **kwargs):
        self.optimizer = optimizer(train_module.parameters(), **kwargs)
        self.module = train_module
        self.active = is_active


class DataSet():
    '''
    Dataset index class, where all the information about the dataset is
    collected. This class should have only the constructor, where the dataset
    structure is described. It is recommended to store the class info here
    instead of the DataSet class as in the other case the amount of memery
    consumed by dataset info may triple.

    You may redefine in this class the following methods:
    ```get_len```, ```get_item```.
    '''

    def __init__(self):
        '''
        Class constructor. Please overload this function definition when making
        your dataset.
        '''
        pass

    def get_len(self, mode='train'):
        '''
        Function that gets length of dataset's subset specified in ```mode```.
        By default is

        ```
        return len(self.inputs[mode])
        ```

        Args:
            mode (str): subset of the dataset to get the length of.

        Returns:
            int, number of elements in the subset.
        '''

        return len(self.data[mode])

    def get_item(self, index, mode='train'):
        '''
        Function that gets the item from the dataset's subset specified in
        ```mode```.

        By default it is:
        ```
        datum = self.data[mode][index]
        target = self.labels[mode][index]

        id = mode + "_" + str(index)

        return [datum], [target], id
        ```

        Args:
            index: index of the item to be loaded

            mode (str): subset of the dataset to get the length of.

        Returns:
            list of inputs to the neural network for the item.

            list of targets for the item.

            id (str is preferred) of the item.
        '''

        datum = self.data[mode][index]
        target = self.labels[mode][index]

        id = mode + "_" + str(index)

        return [datum], [target], id


class Network(torch.nn.Module):
    '''
    Base class for your models.

    Your networks should be a subclass of this class.
    In your models you should redefine ```__init__``` function
    to construct the model and ```forward``` function.
    '''
    def __init__(self):
        '''
        Class constructor.

        You should define your network elements here. Network elements should
        be ```torch.nn.Module``` subclasses' objects.

        Args:
            This method may also take extra keyword arguments specified with
            ```--network-args``` keyword in the bash or specified in
            ```network_args``` parameter of a function call.
        '''
        super(Network, self).__init__()


    def forward(self, input):
        '''
        Forward propagation function for your network. By default this function
        returns the input.

        Args:
            input (list): A list of batches of inputs for your Network (most
                likely, you want a list of ```torch.Tensors```).
                Note that those inputs come in batches.

        Returns:
            list: A list of outputs of Network for specified inputs.
        '''

        return input

    def __call__(self, input):
        '''
        Call should duplicate the forward propagation function
        (in some versions of pytorch the __call__ is needed, while in the
        others it is enough to have just forward function).

        Args:
            input (list): Same as ```forward``` function arguments.

        Returns:
            The same result as ```forward``` function.
        '''
        return self.forward(input)


class Trainer():
    '''
    Socket base class for your networks. The object of this class
    knows how to deal with your network. It contains such fields and methods as
    optimizers, criterion, metrics, process_result, visualize, scheduling.
    '''
    def __init__(self, model,
                 optimizers,
                 criterion,
                 callbacks=[],
                 use_cuda=False,
                 seed=0,
                 deterministic_cuda=False,
                 silent=False):
        '''
        Class constructor. This is one of the most important
        things that you have to redefine. In this function you should define:

        ```self.optimizers``` -- a list of OptimizerSwitches that will be used
        during the training procedure.

        Args:
            model (base.Network): the model for the socket.

            callbacks (list): List of scorch.callbacks to use during training.

            use_cuda (bool): whether to use cuda if available or not.

            seed (int): seed to initialize the random value generators.
            0 by default.

            deterministic_cuda (bool): whether to use deterministic CUDA backend.
            If True, the computations are slower, but deterministic.

            batch_to_metrics (int): how many batches to accumulate before
            metrics computation. Default is 1. If None -- use all batches for
            metrics computation.

            silent (bool): no outputs if True.
        '''

        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic_cuda:
                torch.backends.cudnn.deterministic = True
            else:
                torch.backends.cudnn.deterministic = False

        self._model = torch.nn.DataParallel(model)

        self._optimizers = optimizers
        self._criterion = criterion
        self._callbacks = callbacks

        self._epoch = 0
        self._iteration = 0

        self._best_metrics = None

        self._silent = silent

        self._use_cuda = use_cuda

        for callback in self._callbacks:
            callback.set_trainer(self)
            callback.on_init()


    def train(self,
              dataset,
              batch_size=4,
              num_workers=0,
              epochs=1000,
              validate_on_train=False,
              max_train_iterations=None,
              max_valid_iterations=None,
              max_test_iterations=None,
              solo_test=False):

        '''
        Train the model (full training procedure with validation and testing
        stages).

        Args:
            dataset (base.DataSet)

            batch_size (int): batch size for training and testing.

            num_workers (int): number of workers to use with torch DataLoader.

            epochs (int): number of epochs in training procedure

            validate_on_train (bool): perform validation on training subset

            max_train_iterations (int): number of iterations in training procedure

            max_valid_iterations (int): number of iterations in validation procedure

            max_test_iterations (int): number of iterations in testing procedure

            solo_test (bool): if you need to feed the test inputs one-by-one.

        '''

        # Enabling garbage collector
        gc.enable()

        # Creating tensorboard writer
        tb_writer = SummaryWriter()

        # Validation on training subuset
        if validate_on_train:
            self.validate_one_epoch(dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    max_iterations=max_valid_iterations,
                                    subset='train')
            gc.collect()

        # Validation on validation subset
        self.validate_one_epoch(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                max_iterations=max_test_iterations,
                                subset='valid')

        gc.collect()

        for callback in self._callbacks:
            callback.on_train_begin()

        # Training cycle
        for epoch_index in range(epochs):

            # Training
            gc.enable()
            self.train_one_epoch(dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 max_iterations=max_train_iterations)
            gc.collect()

            # Validation on training subuset
            if validate_on_train:
                self.validate_one_epoch(dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        max_iterations=max_valid_iterations,
                                        subset='train')
                gc.collect()

            # Validation on validation subset
            self.validate_one_epoch(dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    max_iterations=max_valid_iterations,
                                    subset='valid')

            gc.collect()

            self.predict(dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          solo_test=solo_test,
                          max_iterations=max_test_iterations)

        for callback in self._callbacks:
            callback.on_train_end()


    def train_one_epoch(self,
                        dataset,
                        batch_size=4,
                        num_workers=0,
                        max_iterations=None):

        '''
        Trains a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance

            batch_size (int): batch size to use during training

            num_workers (int): number of workers to use in torch.data.DataLoader

            max_iterations (int): maximum amount of iterations to perform during one epoch
        '''

        self._mode = 'training'
        self._subset = 'train'

        self._dataset = internal.DataSetWrapper(
            dataset, mode='train')


        train_sampler = torch.utils.data.sampler.SequentialSampler(self._dataset)

        train_loader = data.DataLoader(self._dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=True,
                                       pin_memory=True,
                                       collate_fn=internal.default_collate,
                                       sampler=train_sampler)

        gc.enable()

        self._epoch += 1

        batch_time = internal.AverageMeter()
        data_time = internal.AverageMeter()
        losses = internal.AverageMeter()

        iterator = iter(train_loader)
        n_iterations = len(train_loader)
        if max_iterations is not None:
            n_iterations = min(len(train_loader), max_iterations)

        self._model.eval()

        gc.collect()

        # Progress bar
        pbar = tqdm(
            range(n_iterations), ascii=True, disable=self._silent, ncols=0)

        pbar.set_description(
            "Train -  "
            "D ----(----)  "
            "L --------(--------)")

        end = time.time()

        avg_metrics = {}

        for callback in self._callbacks:
            callback.on_epoch_begin()

        # Iterating through the batches
        for i in pbar:

            start = time.time()
            self._input, self._target, self._ids = next(iterator)
            data_time.update(time.time() - start)

            for opt_index in range(len(self._optimizers)):
                if self._optimizers[opt_index].active:
                    self._optimizers[opt_index].module.train()

            for callback in self._callbacks:
                callback.on_batch_begin()

            # Moving tensors to CUDA device
            if self._use_cuda and torch.cuda.is_available():
                for index in range(len(self._input)):
                    self._input[index] = self._input[index].cuda()

                for index in range(len(self._target)):
                    self._target[index] = self._target[index].cuda()

            self._output = self._model.forward(self._input)
            self._loss = self._criterion(self._output, self._target)

            for opt_index in range(len(self._optimizers)):
                self._optimizers[opt_index].optimizer.zero_grad()

            self._loss.backward()

            for opt_index in range(len(self._optimizers)):
                if self._optimizers[opt_index].active:
                    self._optimizers[opt_index].optimizer.step()

            self._iteration += 1

            losses.update(self._loss.data.item())

            batch_time.update(time.time() - end)
            end = time.time()

            # Print status
            self._line = ("Train {0}  "
                         "D {data.avg:.2f}({data.val:.2f}) ".format(
                            self._epoch,
                            time=batch_time,
                            data=data_time,
                            loss=losses))

            for callback in self._callbacks:
                callback.on_batch_end()

            del self._input, self._target, self._loss, self._output

            pbar.set_description(self._line)

            for opt_index in range(len(self._optimizers)):
                if self._optimizers[opt_index].active:
                    self._optimizers[opt_index].module.eval()


        for callback in self._callbacks:
            callback.on_epoch_end()

        gc.collect()


    def validate_one_epoch(self,
                           dataset,
                           subset='valid',
                           batch_size=4,
                           num_workers=0,
                           max_iterations=None):
        '''
        Validates a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance

            subset (str): which subset of the dataset to use

            batch_size (int): batch size to use during training

            num_workers (int): number of workers to use in torch.data.DataLoader

            max_iterations (int): maximum amount of iterations to perform during one epoch
        '''

        self._mode = 'validating'
        self._subset = subset

        metrics = {}
        with torch.no_grad():

            # Creating test wrapper for the dataset
            self._dataset = internal.DataSetWrapper(
                dataset, mode=subset)

            for callback in self._callbacks:
                callback.on_epoch_begin()

            valid_sampler = torch.utils.data.sampler.SequentialSampler(self._dataset)

            # Creating dataloader
            valid_loader = data.DataLoader(self._dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=False,
                                       pin_memory=True,
                                       collate_fn=internal.default_collate,
                                       sampler=valid_sampler)

            gc.enable()

            batch_time = internal.AverageMeter()
            data_time = internal.AverageMeter()
            losses = internal.AverageMeter()

            iterator = iter(valid_loader)
            n_iterations = len(valid_loader)
            if max_iterations is not None:
                n_iterations = min(len(valid_loader), max_iterations)

            self._model.eval()

            pbar = tqdm(
                range(n_iterations), ascii=True,
                disable=self._silent, ncols=0)

            end = time.time()
            pbar.set_description(
                "Valid -  "
                "D ----(----)  "
                "L --------(--------)")

            for i in pbar:

                start = time.time()
                self._input, self._target, self._ids = next(iterator)
                data_time.update(time.time() - start)

                for callback in self._callbacks:
                    callback.on_batch_begin()

                if self._use_cuda:
                    for index in range(len(self._input)):
                        self._input[index] = self._input[index].cuda()

                if self._use_cuda:
                    for index in range(len(self._target)):
                        self._target[index] = self._target[index].cuda()

                self._output = self._model.forward(self._input)
                self._loss = self._criterion(self._output, self._target)

                losses.update(self._loss.data.item())

                batch_time.update(time.time() - end)
                end = time.time()

                self._line = ("Valid {0}  "
                        "D {data.avg:.2f}({data.val:.2f}) ".format(
                            self._epoch,
                            data=data_time,
                            loss=losses))

                for callback in self._callbacks:
                    callback.on_batch_end()

                del self._input, self._output, self._target

                pbar.set_description(self._line)

            for callback in self._callbacks:
                callback.on_epoch_end()

            gc.collect()
            gc.collect()


    def predict(self,
                dataset,
                solo_test=True,
                subset='test',
                batch_size=4,
                num_workers=0,
                max_iterations=None):

        '''
        Validates a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance

            solo_test (bool): tests images one-by-one.

            subset (str): which subset of the dataset to use

            batch_size (int): batch size to use during training

            num_workers (int): number of workers to use in torch.data.DataLoader

            max_iterations (int): maximum amount of iterations to perform during one epoch
        '''
        self._mode = 'predicting'
        self._dataset = dataset

        with torch.no_grad():

            # Creating test wrapper for the dataset
            test_dataset = internal.DataSetWrapper(
                dataset, mode=subset)

            # Creating distributed samplers for horovod
            if solo_test:
                batch_size = 1

            test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)

            # Creating dataloader
            test_loader = data.DataLoader(test_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=False,
                                       pin_memory=True,
                                       collate_fn=internal.default_collate,
                                       sampler=test_sampler)

            gc.enable()

            iterator = iter(test_loader)
            n_iterations = len(test_loader)

            if max_iterations is not None:
                n_iterations = min(max_iterations, n_iterations)

            self._model.eval()

            gc.collect()

            pbar = tqdm(range(n_iterations), ascii=True,
                        disable=self._silent, ncols=0)
            pbar.set_description("Test  ")

            for i in pbar:

                for callback in self._callbacks:
                    callback.on_batch_begin()

                self._input, self._target, self._ids = next(iterator)

                if self._use_cuda:
                    for index in range(len(self._input)):
                        self._input[index] = self._input[index].cuda()

                self._output = self._model(self._input)

                for callback in self._callbacks:
                    callback.on_batch_end()

                gc.collect()

                del self._input, self._output, self._ids, self._target


    def save(self, name='./checkpoint', info=""):

        '''
        Saves trainer to the checkpoint (stored in checkpoints directory).

        Args:
            prefix (str):
        '''

        checkpoint = {
            "epoch": self._epoch,
            "iteration": self._iteration,
            "model_state": self._model.module.cpu().state_dict(),
            "info": info,
            "metrics_val": self._val_metrics}

        self._model.module.cuda()

        if hasattr(self, '_train_metrics'):
            checkpoint['metrics_train'] = self._train_metrics

        for opt_index in range(len(self._optimizers)):
            checkpoint['optimizer_state_' + str(opt_index)] = self._optimizers[opt_index].optimizer.state_dict()
            checkpoint['optimizer_switch_' + str(opt_index)] = self._optimizers[opt_index].active

        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        torch.save(checkpoint,
                   'checkpoints/' + name + '.pth.tar')

    def load(self, checkpoint_name):
        '''
        Loads model parameters from the checkpoint.

        Args:
            checkpoint_name (str): path to the checkpoint of interest
        '''
        checkpoint = torch.load(open(checkpoint_name, 'rb'))
        #print(checkpoint['info'])
        print("Model restored from", checkpoint_name)

        self._epoch = checkpoint['epoch']
        self._iteration = checkpoint['iteration']
        self._model.module.load_state_dict(checkpoint["model_state"])
        self._val_metrics = checkpoint["metrics_val"]

        if 'metrics_train' in checkpoint:
            self._train_metrics = checkpoint["metrics_train"]
        #restored_trainer.metrics = checkpoint["metrics"]

        # if not new_optimizer:
        for opt_index in range(len(self.optimizers)):
            try:
                self._optimizers[index].optimizer.load_state_dict(
                    checkpoint["optimizer_state_" + str(opt_index)])
                self._optimizers[index].active = checkpoint[
                    "optimizer_active_" + str(opt_index)]
            except:
                print('Failed to load optimizer ' + str(opt_index) + '.')
