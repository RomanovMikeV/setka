"""
This is the base module of the package. It may also be called the
engine. It contains the core classes that are used in the scripts
such as:

* OptimizerSwitch -- wrapper to facilitate work with optimizers.
* DataSet -- a dataset class that Trainer needs.
* Network -- a network class that Trainer needs.
* Trainer -- class that performs training, validation and prediction.

"""


import numpy
import os
import random
import torch
import torch.utils.data
import shutil
import gc
import time
import copy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from . import internal


class OptimizerSwitch():
    '''
    This class contains optimizer and the module that this
    optimizer should optimize. The OptimizerSwitch tells trainer
    if it should switch the modules to the training mode and back
    to evaluation mode and if it should perform optimization for
    this module's parameters.
    '''

    def __init__(self, train_module, optimizer, is_active=True, **kwargs):
        '''
        Constructs the OptimizerSwitch instance.

        Args:
            train_module (torch.nn.Module):

            optimizer (torch.optim.Optimizer):

            is_active (bool):

            **kwargs:

        Example:
        ```
        OptimizerSwitch(
            net.params(),
            torch.optim.Adam,
            lr=3.0e-4,
            is_active=False)
        ```
        '''

        self.optimizer = optimizer(train_module.parameters(), **kwargs)
        self.module = train_module
        self.active = is_active


class DataSet():
    '''
    Dataset class, where all the information about the dataset is
    collected. It contains all the information about the dataset.
    The dataset may be split into the subsets (that are ususally
    referred to as ```train```, ```valid``` and ```test```, but
    you may make much more different modes for your needs).

    The DataSet has the following methods:
        * __init__ -- constructor, where the index collection
            is ususally performed. You need to define it.
            In the original constuctor the ```subset``` is set
            to ```train```.

        * getlen -- function that gets the size of the subset of the dataset.
            This function gets as argument the subset ID (hashable).
            You need to define this function in your class.

        * getitem -- function that retrieves the item form the dataset.
            This function gets as arguments the subset ID (hashable) and the
            ID of the element in the subset (index).
            You need to define this function in your class.


        * getitem -- function that selects

        * __len__ -- function that is called when the ```len()```
            operator is called and returns the volume of the
            current subset. Predefined, you do not need to redefine it.

        * __getitem__ -- function that gets the item of the dataset
            by its index. Predefined, you do not need to redefine it.


    '''

    def __init__(self):
        pass

    def getitem(self, subset, index):
        pass

    def getlen(self, subset):
        pass

    def __getitem__(self, *args):
        args = args[0]
        if type(args) is tuple:
            assert(len(args) <= 2)
            assert(len(args) > 0)

            return self.getitem(args[0], args[1])
        else:
            if not hasattr(self, 'subset'):
                sliced_dataset = copy.copy(self)
                sliced_dataset.subset = args
                return sliced_dataset

            else:
                return self.getitem(self.subset, args)

    def __len__(self):
        assert(hasattr(self, 'subset'))
        return self.getlen(self.subset)


class Network(torch.nn.Module):
    '''
    Base class for your models, where the element of your model are
    defined and the forward propagation.

    Your networks should be a subclass of this class.

    The model contains the following methods:

        * __init__ -- class constructor. Usually here the blocks of
            the network are specified. You will need to redefine
            this method in your network.

        * forward -- function that performs the forward propagation
            of the signal through the network. You will need to
            redefine this method in your network.

        * __call__ -- alias to forward.
    '''
    def __init__(self):
        '''
        Class constructor.

        Most likely you have to redefine it.

        You should define your network elements here.
        Network elements should be ```torch.nn.Module```
        subclasses' objects.
        '''
        super(Network, self).__init__()


    def forward(self, input):
        '''
        Forward propagation function for your network.

        Most likely you have to redefine it.

        By default this function returns the input.

        Takes inputs and computes outputs.

        To use the majority of callbacks, you have to wrap inputs
        and outputs of the network into the lists.

        Args:
            input (list): A list of batches of inputs for your Network (most
                likely, you want a list of ```torch.Tensors```).
                Note that those inputs come in batches.

        Returns:
            (list) Outputs from the Network for specified inputs.
        '''

        return input

    def __call__(self, input):
        '''
        Alias to forward.
        '''
        return self.forward(input)


class Trainer():
    '''
    Trainer can train and validate your network. It can also
    predict results.

    It contains the following methods:

    * __init__ -- class constructor, where you specify everything
        that you will need during the training procedure.

    * train_one_epoch -- performs training for one epoch

    * validate_one_epoch -- performs validation for one epoch

    * predict -- predicts results

    * save -- dumps network's and optimizer's states to the file

    * load -- loads network's and optimizer's states from the file

    '''

    @staticmethod
    def setup_environment(seed=0,
                          max_threads=4,
                          deterministic_cuda=False,
                          benchmark_cuda=True):
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        os.environ["OMP_NUM_THREADS"] = str(max_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
        os.environ["MKL_NUM_THREADS"] = str(max_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = deterministic_cuda
            torch.backends.cudnn.benchmark = benchmark_cuda


    def __init__(self, model,
                 optimizers,
                 criterion,
                 callbacks=[],
                 seed=0,
                 max_threads=4,
                 deterministic_cuda=False,
                 benchmark_cuda=True,
                 silent=False):
        '''
        Class constructor.

        Args:
            model (base.Network) -- model to train

            optimizers (list of base.OptimizerSwitch) -- optimizer
                to use during training. The optimisers whose
                is_active flag is set to True will be used in the
                optimization procedure (there may be many of them)

            criterion (function) -- criterion (loss) function that
                needs to be optimized

            callbacks (list of callbacks.Callback) -- callbacks
                that extend functionality of the trainer.

            seed (int) -- seed for random number generators to use.

            deterministic_cuda (bool) -- flag that specifies if
                the cuda should behave deterministically
                (takes more time to perform computations)

            silent (bool) -- do not show output


        Args:
            model (base.Network): the model for the socket.

            callbacks (list): List of setka.callbacks to use
                during training.

            seed (int): seed to initialize the random value
                generators. 0 by default.

            deterministic_cuda (bool): whether to use deterministic
                CUDA backend. If True, the computations are slower,
                but deterministic.

            batch_to_metrics (int): how many batches to accumulate before
            metrics computation. Default is 1. If None -- use all batches for
            metrics computation.

            silent (bool): no outputs if True.
        '''
        self.setup_environment(seed=seed,
                               max_threads=max_threads,
                               deterministic_cuda=deterministic_cuda,
                               benchmark_cuda=benchmark_cuda)

        self._model = torch.nn.DataParallel(model)

        self._optimizers = optimizers
        self._criterion = criterion
        self._callbacks = callbacks

        self._epoch = 0
        self._iteration = 0

        self._best_metrics = None

        self._silent = silent

        for callback in self._callbacks:
            callback.set_trainer(self)
            callback.on_init()


    def train_one_epoch(self,
                        dataset,
                        subset='train',
                        batch_size=4,
                        num_workers=0,
                        max_iterations=None):

        '''
        Trains a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance to train on.

            subset (hashable): identifier of the dataset's subset
                which willbe used for training.

            batch_size (int): batch size to use during training.

            num_workers (int): number of workers to use in
                torch.utils.data.DataLoader.

            max_iterations (int): maximum amount of iterations to
                perform during one epoch. If None -- training
                will be performed until the end of the subset.
        '''

        self._mode = 'training'
        self._subset = subset
        self._dataset = dataset

        self._ds_wrapper = internal.DataSetWrapper(self._dataset[subset], subset)

        for callback in self._callbacks:
            callback.on_epoch_begin()

        train_sampler = torch.utils.data.sampler.SequentialSampler(self._ds_wrapper)

        train_loader = torch.utils.data.DataLoader(self._ds_wrapper,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=True,
                                       pin_memory=True,
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
            "Train ----"
            "D ----(----)  ")

        end = time.time()

        # Iterating through the batches
        for i in pbar:
            self._progress = i / len(pbar)

            start = time.time()

            self._input, self._ids = next(iterator)

            if not isinstance(self._input, list):
                self._input = [self._input,]

            data_time.update(time.time() - start)

            for opt_index in range(len(self._optimizers)):
                if self._optimizers[opt_index].active:
                    self._optimizers[opt_index].module.train()

            for callback in self._callbacks:
                callback.on_batch_begin()

            # Moving tensors to CUDA device
            if torch.cuda.is_available():
                for index in range(len(self._input)):
                    self._input[index] = self._input[index].cuda()

            for opt_index in range(len(self._optimizers)):
                self._optimizers[opt_index].optimizer.zero_grad()
                    
            self._output = self._model.forward(self._input)
            if not isinstance(self._output, tuple):
                self._output = (self._output, )

            self._loss = self._criterion(self._output, self._input)

            self._loss.backward()

            for opt_index in range(len(self._optimizers)):
                if self._optimizers[opt_index].active:
                    self._optimizers[opt_index].optimizer.step()

            self._iteration += 1

            losses.update(self._loss.data.item())

            batch_time.update(time.time() - end)
            end = time.time()

            # Print status
            self._line = ("Train {0:4d}  "
                         "D {data.avg:.2f}({data.val:.2f}) ".format(
                            self._epoch,
                            time=batch_time,
                            data=data_time,
                            loss=losses))

            for callback in self._callbacks:
                callback.on_batch_end()

            for opt_index in range(len(self._optimizers)):
                if self._optimizers[opt_index].active:
                    self._optimizers[opt_index].module.eval()

            del self._input, self._loss, self._output

            if i == len(pbar) - 1:
                for callback in self._callbacks:
                    callback.on_epoch_end()

            pbar.set_description(self._line)




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

            subset (hashable): which subset of the dataset to use

            batch_size (int): batch size to use during training

            num_workers (int): number of workers to use in
                torch.utils.data.DataLoader

            max_iterations (int): maximum amount of iterations to
                perform during one epoch. If None -- training
                will be performed until the end of the subset.

        '''

        self._mode = 'validating'
        self._subset = subset
        self._dataset = dataset

        with torch.no_grad():

            # Creating test wrapper for the dataset
            self._ds_wrapper = internal.DataSetWrapper(dataset[subset], subset)

            for callback in self._callbacks:
                callback.on_epoch_begin()

            valid_sampler = torch.utils.data.sampler.SequentialSampler(
                self._ds_wrapper)

            # Creating dataloader
            valid_loader = torch.utils.data.DataLoader(self._ds_wrapper,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=False,
                                       pin_memory=True,
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
                "Valid ----"
                "D ----(----)")

            for i in pbar:
                self._progress = i / len(pbar)

                start = time.time()
                self._input, self._ids = next(iterator)

                if not isinstance(self._input, list):
                    self._input = [self._input,]

                data_time.update(time.time() - start)

                for callback in self._callbacks:
                    callback.on_batch_begin()

                if torch.cuda.is_available():
                    for index in range(len(self._input)):
                        self._input[index] = self._input[index].cuda()

                self._output = self._model.forward(self._input)

                if not isinstance(self._output, tuple):
                    self._output = (self._output, )

                self._loss = self._criterion(self._output, self._input)

                losses.update(self._loss.data.item())

                batch_time.update(time.time() - end)
                end = time.time()

                self._line = ("Valid {0:4d}  "
                        "D {data.avg:.2f}({data.val:.2f}) ".format(
                            self._epoch,
                            data=data_time,
                            loss=losses))

                for callback in self._callbacks:
                    callback.on_batch_end()

                del self._input, self._output

                if i == len(pbar) - 1:
                    for callback in self._callbacks:
                        callback.on_epoch_end()

                pbar.set_description(self._line)

            pbar.set_description(self._line)

            gc.collect()
            gc.collect()


    def predict(self,
                dataset,
                subset='test',
                batch_size=4,
                num_workers=0,
                max_iterations=None):

        '''
        Validates a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance

            subset (hashable): which subset of the dataset to use

            batch_size (int): batch size to use during training

            num_workers (int): number of workers to use in
                torch.utils.data.DataLoader

            max_iterations (int): maximum amount of iterations to
                perform during one epoch
        '''
        self._mode = 'predicting'
        self._dataset = dataset
        self._subset = subset

        with torch.no_grad():

            # Creating test wrapper for the dataset
            self._ds_wrapper = internal.DataSetWrapper(
                dataset[subset], subset)

            for callback in self._callbacks:
                callback.on_epoch_begin()

            test_sampler = torch.utils.data.sampler.SequentialSampler(
                self._ds_wrapper)

            # Creating dataloader
            test_loader = torch.utils.data.DataLoader(self._ds_wrapper,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=False,
                                       pin_memory=True,
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
                self._progress = i / len(pbar)

                for callback in self._callbacks:
                    callback.on_batch_begin()

                self._input, self._ids = next(iterator)
                if not isinstance(self._input, list):
                    self._input = [self._input,]

                if torch.cuda.is_available():
                    for index in range(len(self._input)):
                        self._input[index] = self._input[index].cuda()

                self._output = self._model(self._input)

                if not isinstance(self._output, tuple):
                    self._output = (self._output, )

                for callback in self._callbacks:
                    callback.on_batch_end()

                gc.collect()

                del self._input, self._output, self._ids

        for callback in self._callbacks:
            callback.on_epoch_end()


    def save(self, name='./checkpoint'):

        '''
        Saves trainer to the checkpoint (stored in checkpoints directory).

        Args:
            name (str): name of the file where the model is saved.
        '''

        checkpoint = {
            "epoch": self._epoch,
            "iteration": self._iteration,
            "model_state": self._model.module.cpu().state_dict()}

        if hasattr(self, '_metrics'):
            checkpoint['metrics'] = self._metrics

        for opt_index in range(len(self._optimizers)):
            checkpoint['optimizer_state_' + str(opt_index)] = (
                self._optimizers[opt_index].optimizer.state_dict())
            checkpoint['optimizer_switch_' + str(opt_index)] = (
                self._optimizers[opt_index].active)

        torch.save(checkpoint,
                   name)

        if torch.cuda.is_available():
            self._model.module.cuda()


    def load(self, checkpoint_name):
        '''
        Loads model parameters from the checkpoint.

        Args:
            checkpoint_name (str): path to the checkpoint
                to load.
        '''
        checkpoint = torch.load(open(checkpoint_name, 'rb'))
        print("Model restored from", checkpoint_name)

        self._epoch = checkpoint['epoch']
        self._iteration = checkpoint['iteration']
        self._model.module.load_state_dict(checkpoint["model_state"])
        if 'metrics' in checkpoint:
            self._metrics = checkpoint['metrics']

        # if not new_optimizer:
        for opt_index in range(len(self._optimizers)):
            try:
                self._optimizers[opt_index].optimizer.load_state_dict(
                    checkpoint["optimizer_state_" + str(opt_index)])
                self._optimizers[opt_index].active = checkpoint[
                    "optimizer_switch_" + str(opt_index)]
            except:
                print('Failed to load optimizer ' +
                    str(opt_index) + '.')
