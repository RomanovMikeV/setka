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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
                 use_cuda=False,
                 seed=0,
                 deterministic_cuda=True,
                 horovod=False,
                 global_metrics=False,
                 silent=False):
        '''
        Class constructor. This is one of the most important
        things that you have to redefine. In this function you should define:

        ```self.optimizers``` -- a list of OptimizerSwitches that will be used
        during the training procedure.

        Args:
            model (base.Network): the model for the socket.
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

        #self.horovod = horovod

        #if self.horovod:
        #    hvd.init()

        self.model = torch.nn.DataParallel(model)

        self.epoch = 0
        self.iteration = 0

        self.metrics_valid = {}
        self.metrics_train = {}

        self.best_metrics = None

        self.global_metrics = global_metrics

        self.silent = silent

        self.use_cuda = use_cuda




    def criterion(self, preds, target):
        '''
        Function that computes the criterion, also known as loss function
        for your network. You should define this function for your network
        in order to proceed.

        Note that the loss function should be computed using only pytorch
        operations as you should be able to do backpropagation.

        Args:
            preds: list of outputs returned by your model.

            target: list of target values for your model.

        Returns:
            value for the criterion function.
        '''
        return self.criterion_f(input, target)

    def metrics(self, preds, target):
        '''
        Optional.

        Function that computes metrics for your model.
        Note that there should be a main metric based
        on which framework will decide which checkpoint
        to mark as best.

        Arguments are the same as in the ```criterion method```.

        By default returns criterion value as main metric.

        Args:
            preds (list): list of outputs returned by your model.

            target (list): list of target values for your model.

        Returns:
            dictionary with metrics for your model.
        '''

        return {'main': self.criterion(preds, target)}

    def process_result(self, input, output, id):
        '''
        Optional.

        Function that processes the outputs of the network and forms
        the understandable result of the network. Used only during the
        testing procedure.

        The results of this function will be packed in pth.tar files
        and saved as inference results of the network.

        By default returns {id: output}.

        Args:
            input (list): List of inputs for the network for one item.

            output (list): list of outputs from the network for one item.

            id (str): an id of the sample. The string is preferred.

        Returns:
            dictionary, containing inference results of the network to be stored
            as a result. It is useful to use the id of the sample as a key.
        '''

        return {id: output}

    def visualize(self, input, output, id):
        '''
        Optional.

        Function that is used during the testing session of the epoch.
        It visualizes how the network works. You may use matplotlib for
        visualization.

        By default returns empty dict.

        Args:
            input (list): List of inputs for the network for one item.

            output (list): list of outputs from the network for one item.

            id (str): an id of the sample. The string is preferred.

        Returns:
            dictionary, containing the following fields:

            ```figures```
            (the value should be the dictionary with matplotlib figures),

            ```images``` (the value should be the dictionary with
            numpy arrays that can be treated as images),

            ```texts``` (the value should be the dictionary with strings),

            ```audios```, ```graphs``` and ```embeddings```.
        '''
        return {}


    def scheduling(self, new_epoch=False):
        '''
        Optional.

        Function that is called every iteration. You may use it to
        switch between optimizers, change learning rate and do whatever
        you want your model to do from iteration to iteration.

        Has no inputs or outputs, usually just changes parameters of training.

        Does nothing by default.
        '''
        pass


    def train(self,
              dataset,
              batch_size=4,
              num_workers=0,
              dump_period=1,
              epochs=1000,
              validate_on_train=False,
              max_train_iterations=None,
              max_valid_iterations=None,
              max_test_iterations=None,
              checkpoint_name='checkpoint',
              solo_test=False):

        '''
        Train the model (full training procedure with validation and testing
        stages).

        Args:
            batch_size (int): batch size for training and testing.

            num_workers (int): number of workers to use with torch DataLoader.

            dump_period (int): number epochs between checkpoint dumps

            epochs (int): number of epochs in training procedure

            validate_on_train (bool): perform validation on training subset

            max_train_iterations (int): number of iterations in training procedure

            max_valid_iterations (int): number of iterations in validation procedure

            max_test_iterations (int): number of iterations in testing procedure

            checkpoint_prefix (str): prefix with which the chekpoint will be saved
                for the experiment.

            solo_test (bool): if you need to feed the test inputs one-by-one.

        '''

        # Enabling garbage collector
        gc.enable()

        # Creating tensorboard writer

        #if self.get_write_flag():
        #    tb_writer = SummaryWriter()
        #else:
        #    silent = True
        tb_writer = SummaryWriter()

        # Validation on training subuset
        if validate_on_train:
            self.metrics_train = self.validate_one_epoch(dataset,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         max_iterations=max_valid_iterations,
                                                         subset='train')
            gc.collect()

        # Validation on validation subset
        self.metrics_valid = self.validate_one_epoch(dataset,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     max_iterations=max_test_iterations,
                                                     subset='valid')

        gc.collect()

        # Training cycle
        for epoch_index in range(epochs):

            self.scheduling(new_epoch=True)

            # Training
            gc.enable()
            loss = self.train_one_epoch(dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        max_iterations=max_train_iterations)
            gc.collect()

            train_metrics = {}

            # Validation on training subuset
            if validate_on_train:
                self.metrics_train = self.validate_one_epoch(dataset,
                                                             batch_size=batch_size,
                                                             num_workers=num_workers,
                                                             max_iterations=max_valid_iterations,
                                                             subset='train')
                gc.collect()

            # Validation on validation subset
            self.metrics_valid = self.validate_one_epoch(dataset,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         max_iterations=max_valid_iterations,
                                                         subset='valid')
            gc.collect()

            # Processing resulting metrics
            for metric_name in self.metrics_valid:
                metric_data = {'valid': self.metrics_valid[metric_name]}

                if metric_name in self.metrics_train:
                    metric_data['train'] = self.metrics_train[metric_name]

                #if self.get_write_flag():
                tb_writer.add_scalars(
                        'metrics/' + metric_name + '/' + checkpoint_name,
                        metric_data, self.epoch)


            # Running tests for visualization

            #if self.get_write_flag():
            inputs = []
            outputs = []
            test_ids = []


            for input, output, test_id in self.test_one_epoch(dataset,
                                                              batch_size=batch_size,
                                                              num_workers=num_workers,
                                                              solo_test=solo_test,
                                                              max_iterations=max_test_iterations):

                #if self.get_write_flag():
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

            #if self.get_write_flag():
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

                internal.show(
                        tb_writer,
                        self.visualize(
                            one_input, one_output, test_id),
                        self.epoch)

            del inputs
            del outputs



            gc.collect()

            # Dumping the model
            if epoch_index % dump_period == 0: #and self.get_write_flag():
                is_best = False
                if self.best_metrics is None:
                    is_best = True
                else:
                    if self.best_metrics['main'] < self.metrics_valid['main']:
                        is_best = True
                if is_best:
                    self.best_metrics = self.metrics_valid

                self.save(name=checkpoint_name,
                          is_best=is_best)

            gc.collect()

        return self.metrics_valid


    def train_one_epoch(self,
                        dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0,
                        max_iterations=None):

        '''
        Trains a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance

            batch_size (int): batch size to use during training

            shuffle (bool): indicates if you want to shuffle dataset before training

            num_workers (int): number of workers to use in torch.data.DataLoader

            max_iterations (int): maximum amount of iterations to perform during one epoch

        Returns:
            float, average loss value during epoch
        '''

        # Creating test wrapper for the dataset
        train_dataset = internal.DataSetWrapper(
            dataset, mode='train')

        if shuffle:
            train_dataset.shuffle()

        # Creating distributed samplers for horovod
#        if self.horovod:
#            train_sampler = torch.utils.data.distributed.DistributedSampler(
#                train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
#        else:
        train_sampler = torch.utils.data.sampler.SequentialSampler(train_dataset)

        # Creating dataloader
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=True,
                                       pin_memory=True,
                                       collate_fn=internal.default_collate,
                                       sampler=train_sampler)

        gc.enable()

        self.epoch += 1

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        iterator = iter(train_loader)
        n_iterations = len(train_loader)
        if max_iterations is not None:
            n_iterations = min(len(train_loader), max_iterations)

        self.model.eval()

        gc.collect()

        # Progress bar
        pbar = tqdm(
            range(n_iterations), ascii=True, disable=self.silent, ncols=0)

        pbar.set_description(
            "Train -  "
            "D ----(----)  "
            "L --------(--------)")

        end = time.time()

        avg_metrics = {}
        

        # Iterating through the batches
        for i in pbar:
            self.scheduling()

            start = time.time()
            input, target, ids = next(iterator)
            data_time.update(time.time() - start)
            
            for opt_index in range(len(self.optimizers)):
                if self.optimizers[opt_index].active:
                    self.optimizers[opt_index].module.train()

            # Moving tensors to CUDA device
            if self.use_cuda and torch.cuda.is_available():
                for index in range(len(input)):
                    input[index] = input[index].cuda()

                for index in range(len(target)):
                    target[index] = target[index].cuda()

            output = self.model.forward(input)
            loss = self.criterion(output, target)

            metrics = {}
            if not self.global_metrics:
                metrics = self.metrics(output, target)

                for metric in metrics:
                    if metric not in avg_metrics:
                        avg_metrics[metric] = AverageMeter()
                    avg_metrics[metric].update(metrics[metric])

            for opt_index in range(len(self.optimizers)):
                self.optimizers[opt_index].optimizer.zero_grad()

            loss.backward()

            for opt_index in range(len(self.optimizers)):
                if self.optimizers[opt_index].active:
                    self.optimizers[opt_index].optimizer.step()

            self.iteration += 1

            losses.update(loss.data.item())

            batch_time.update(time.time() - end)
            end = time.time()

            del input, target
            del loss, output

            # Print status
            line = ("Train {0}  "
                    "D {data.avg:.2f}({data.val:.2f})  "
                    "L {loss.avg:.2e}({loss.val:.2e})  ".format(
                        self.epoch,
                        time=batch_time,
                        data=data_time,
                        loss=losses))

            if len(metrics) > 0:
                line += "Metrics: " + " ".join(
                    ["{}:{:.2e}".format(x, avg_metrics[x].avg) for x in metrics])

            pbar.set_description(line)
        
        
            for opt_index in range(len(self.optimizers)):
                if self.optimizers[opt_index].active:
                    self.optimizers[opt_index].module.eval()
                    
        gc.collect()
        return losses.avg

    def validate_one_epoch(self,
                           dataset,
                           subset='valid',
                           batch_size=4,
                           shuffle=False,
                           num_workers=0,
                           max_iterations=None):
        '''
        Validates a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance

            subset (str): which subset of the dataset to use

            batch_size (int): batch size to use during training

            shuffle (bool): indicates if you want to shuffle dataset before training

            num_workers (int): number of workers to use in torch.data.DataLoader

            max_iterations (int): maximum amount of iterations to perform during one epoch

        Returns:
            dict, dictionary with metrics.
        '''

        metrics = {}
        with torch.no_grad():

            # Creating test wrapper for the dataset
            valid_dataset = internal.DataSetWrapper(
                dataset, mode=subset)

            if shuffle:
                valid_dataset.shuffle()

            # Creating distributed samplers for horovod
            #if self.horovod:
            #    valid_sampler = torch.utils.data.distributed.DistributedSampler(
            #        valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            #else:
            valid_sampler = torch.utils.data.sampler.SequentialSampler(valid_dataset)

            # Creating dataloader
            valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=False,
                                       pin_memory=True,
                                       collate_fn=internal.default_collate,
                                       sampler=valid_sampler)

            gc.enable()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            outputs = []
            targets = []

            n_items = 0

            iterator = iter(valid_loader)
            n_iterations = len(valid_loader)
            if max_iterations is not None:
                n_iterations = min(len(valid_loader), max_iterations)

            self.model.eval()

            pbar = tqdm(
                range(n_iterations), ascii=True,
                disable=self.silent, ncols=0)

            end = time.time()
            pbar.set_description(
                "Valid -  "
                "D ----(----)  "
                "L --------(--------)")

            for i in pbar:

                start = time.time()
                input, target, ids = next(iterator)
                data_time.update(time.time() - start)

                if self.use_cuda:
                    for index in range(len(input)):
                        input[index] = input[index].cuda()

                if self.use_cuda:
                    for index in range(len(target)):
                        target[index] = target[index].cuda()

                output = self.model.forward(input)
                loss = self.criterion(output, target)

                losses.update(loss.data.item())


                for index in range(len(output)):
                    output[index] = output[index].detach()

                for index in range(len(target)):
                    target[index] = target[index].detach()

                if self.use_cuda:
                    for index in range(len(output)):
                        output[index] = output[index].cpu()

                    for index in range(len(target)):
                        target[index] = target[index].cpu()

                #if self.horovod:
                #    for index in range(len(output)):
                #        output[index] = hvd.allgather(output[index])

                #if self.horovod:
                #    for index in range(len(target)):
                #        target[index] = hvd.allgather(target[index])

                batch_time.update(time.time() - end)
                end = time.time()

                line = ("Valid {0}  "
                        "D {data.avg:.2f}({data.val:.2f})  "
                        "L {loss.avg:.2e}({loss.val:.2e})  ".format(
                            self.epoch,
                            data=data_time,
                            loss=losses))

                if self.global_metrics:
                    if i == len(pbar) - 1:
                        outputs = list(zip(*outputs))
                        targets = list(zip(*targets))

                        for index in range(len(outputs)):
                            outputs[index] = torch.cat(outputs[index], dim=0)

                        for index in range(len(targets)):
                            targets[index] = torch.cat(targets[index], dim=0)

                        metrics = {}

                        metrics = self.metrics(outputs, targets)

                        line += "Metrics: " + " ".join(
                            ["{}:{:.2e}".format(x, metrics[x]) for x in metrics])

                    else:
                        outputs.append(output)
                        targets.append(target)


                else:
                    n_items += len(ids)
                    batch_metrics = self.metrics(output, target)

                    for key in batch_metrics:
                        val = batch_metrics[key]

                        if key not in metrics:
                            metrics[key] = 0
                        metrics[key] += batch_metrics[key] * len(ids)

                    line += "Metrics: " + " ".join(
                        ["{}:{:.2e}".format(x, metrics[x] / n_items) for x in metrics])

                del input
                del output, target

                pbar.set_description(line)

            del outputs, targets

            gc.collect()
            gc.collect()

        for metric in metrics:
            metrics[metric] /= n_items
            
        return metrics

    def test_one_epoch(self,
                       dataset,
                       solo_test=True,
                       subset='test',
                       batch_size=4,
                       shuffle=True,
                       num_workers=0,
                       max_iterations=None):

        '''
        Validates a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance

            solo_test (bool): tests images one-by-one.

            subset (str): which subset of the dataset to use

            batch_size (int): batch size to use during training

            shuffle (bool): indicates if you want to shuffle dataset before training

            num_workers (int): number of workers to use in torch.data.DataLoader

            max_iterations (int): maximum amount of iterations to perform during one epoch

        Returns:
            dict, dictionary with metrics.
        '''

        with torch.no_grad():
            # Creating test wrapper for the dataset
            test_dataset = internal.DataSetWrapper(
                dataset, mode=subset)

            if shuffle:
                test_dataset.shuffle()

            # Creating distributed samplers for horovod

            if solo_test:
                batch_size = 1
            #    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
            #elif self.horovod:
            #    test_sampler = torch.utils.data.distributed.DistributedSampler(
            #        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            #else:
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

            self.model.eval()

            gc.collect()

            pbar = tqdm(range(n_iterations), ascii=True,
                        disable=self.silent, ncols=0)
            pbar.set_description("Test  ")

            for i in pbar:

                input, _, id = next(iterator)

                if self.use_cuda:
                    for index in range(len(input)):
                        input[index] = input[index].cuda()

                output = self.model(input)

                #if self.horovod and not solo_test:
                #    for index in range(len(output)):
                #        output[index] = hvd.allgather(output[index])

                #if self.horovod and not solo_test:
                #    for index in range(len(input)):
                #        input[index] = hvd.allgather(input[index])

                gc.collect()

                yield input, output, id


    def predict(self,
                dataset,
                subset='test',
                batch_size=1,
                num_workers=0,
                max_iterations=None,
                prefix=''):

        '''
        Runs and saves predictions of the network.

        Args:
            dataset (base.DataSet): dataset to use for predictions.

            subset (str): subset to produce result on.

            batch_size (int): batch size to use

            num_workers (int): amount of workers for data loading

            max_iterations (int): maximal amount of iterations to perform

            prefix (str): prefix for the dir with predictions
        '''

        if len(prefix) > 0:
            prefix += '_'

        # Enabling garbage collector
        gc.enable()

        # Shut up if you're not the first process
        #if not self.get_write_flag():
        #    silent = True

        if not os.path.exists(prefix + 'results'):
            os.mkdir(prefix + 'results')

        batch_index = 0

        for inputs, outputs, test_ids in self.test_one_epoch(dataset,
                                                             subset=subset,
                                                             batch_size=batch_size,
                                                             max_iterations=max_iterations,
                                                             num_workers=num_workers,
                                                             silent=self.silent):
            result = {}

            for index in range(len(test_ids)):
                one_input = []
                one_output = []
                test_id = test_ids[index]

                for input_index in range(len(inputs)):
                    one_input.append(inputs[input_index][index])

                for output_index in range(len(outputs)):
                    one_output.append(outputs[output_index][index])

                result[test_id] = self.process_result(one_input,
                                                      one_output,
                                                      test_id)

            rank = 0
            #if self.horovod:
            #    rank = hvd.rank()

            torch.save(result,
                os.path.join(prefix + 'results',
                    str(rank) + '_' + str(batch_index) + '.pth.tar'))

            batch_index += 1


        gc.collect()

    def save(self, name='./checkpoint', is_best=False, info=""):

        '''
        Saves trainer to the checkpoint (stored in checkpoints directory).

        Args:
            prefix (str):
        '''

        checkpoint = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model_state": self.model.module.cpu().state_dict(),
            "info": info,
            "metrics_valid": self.metrics_valid}

        if hasattr(self, 'metrics_train'):
            checkpoint['metrics_train'] = self.metrics_train

        for opt_index in range(len(self.optimizers)):
            checkpoint['optimizer_state_' + str(opt_index)] = self.optimizers[opt_index].optimizer.state_dict()
            checkpoint['optimizer_switch_' + str(opt_index)] = self.optimizers[opt_index].active

        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        torch.save(checkpoint,
                   'checkpoints/' + name + '.pth.tar')

        if is_best:
            shutil.copy('checkpoints/' + name + '.pth.tar',
                        'checkpoints/' + name + '_best.pth.tar')

    def load(self, checkpoint_name):
        '''
        Loads model parameters from the checkpoint.

        Args:
            checkpoint_name (str): path to the checkpoint of interest
        '''
        checkpoint = torch.load(open(checkpoint_name, 'rb'))
        #print(checkpoint['info'])
        print("Model restored from", checkpoint_name)

        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.model.module.load_state_dict(checkpoint["model_state"])
        self.metrics_valid = checkpoint["metrics_valid"]

        if 'metrics_train' in checkpoint:
            self.metrics_train = checkpoint["metrics_train"]
        #restored_trainer.metrics = checkpoint["metrics"]

        # if not new_optimizer:
        for opt_index in range(len(self.optimizers)):
            try:
                self.optimizers[index].optimizer.load_state_dict(
                    checkpoint["optimizer_state_" + str(opt_index)])
                self.optimizers[index].active = checkpoint[
                    "optimizer_active_" + str(opt_index)]
            except:
                print('Failed to load optimizer ' + str(opt_index) + '.')
