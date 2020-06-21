import time

import numpy
import torch
import collections
import math
import datetime

from setka.pipes.Pipe import Pipe

DEFAULT_SCHEDULE = [
     {'mode': 'train', 'subset': 'train'},
     {'mode': 'valid', 'subset': 'train'},
     {'mode': 'valid', 'subset': 'valid'}
]

def fractal_order(size):
    power_2 = 1
    while power_2 * 2 < size:
        power_2 *= 2

    order = numpy.zeros(size).astype('long')

    index = 0
    while power_2 > 0:
        to_assign = numpy.arange(len(order[power_2 - 1::2*power_2])) + index
        order[power_2 - 1::2*power_2] = to_assign
        index = to_assign[-1] + 1
        power_2 //= 2
    order = order.argsort()
    return order


class DataSetWrapper:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

        self.order = fractal_order(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        real_index = self.order[index]
        dataset_res = self.dataset[real_index]

        return dataset_res, str(self.name) + '_' + str(real_index)

    def shuffle(self):
        self.order = numpy.random.permutation(len(self.dataset))
        # print('Shuffled order:', self.order[:16])


def progress_str(width, state):
    progress = width * state
    filled = int(math.floor(progress))

    if filled < width:
        remnant = str(int(math.floor((progress - filled) * 10.0)))
        return '[' + '='* filled + remnant + ' ' * (width - filled - 1) + ']'
    else:
        return '[' + '=' * width + ']'


class TimeEstimator:
    def __init__(self, eta_threshold=0.001):
        self.eta_threshold = eta_threshold
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.cur_state = 0
        self.est_finish_time = None
        return self

    def update(self, cur_state):
        self.cur_state = cur_state
        if self.cur_state >= self.eta_threshold:
            self.est_finish_time = self.start_time + (time.time() - self.start_time) / self.cur_state

    def __str__(self):
        elapsed = str(datetime.timedelta(seconds=int(time.time() - self.start_time)))
        if self.est_finish_time is not None:
            eta = str(datetime.timedelta(seconds=int(self.est_finish_time - time.time())))
        else:
            eta = '?'

        return f'[{elapsed}>{eta}]'


class DataSetHandler(Pipe):
    """
    This class is a main pipe in the learning procedure. It holds the dataset and produces batches from the dataset
    for training. Besides, it also controls the epochs launches via ```run_epoch```within training loop and batches
    passes launches via ```run_batch``` within epoch loop.

    Stores:
        self.trainer._input -- input samples
        self.trainer._ids   -- ids of the input samples

    Args:
        dataset: (setka.DataSet, required)
            setka.dataset to handle during the training procedure

        batch_size: (int or dict of ints {'train': 24, 'valid': 32, 'test': 1}, required)
            batch size used for training

        workers: (int, default 0)
            number of workers to be used in the training procedure.

        timeit: (bool, default True)
            if True -- time of new batch waiting and batch computation is tracked. In status
            string printed as 'D' and 'B'

        limits: (int or dict of ints, default {})
            number of iterations to perform within specified subset.

        shuffle: (bool or dict of bools, default {})
            whether to shuffle the dataset before epoch start.

        epoch_schedule: (list of dicts, default is [
                     {'mode': 'train', 'subset': 'train'},
                     {'mode': 'valid', 'subset': 'train'},
                     {'mode': 'valid', 'subset': 'valid'}
                 ])

            sets up modes of trainer and subsets used in these modes during the epoch. Given the default schedule:
                * firstly the Trainer is switched to the `train` mode and `train` subset of the dataset is used.
                * secondly, the Trainer is switched to the `valid` mode and `train` subset of the dataset is used.
                * thirdly, the Trainer is switched to the `valid` mode and `valid` subset of the dataset is used.
    """
    def __init__(self, dataset, batch_size, workers=0, timeit=True, limits={}, shuffle={'train': True},
                 epoch_schedule=DEFAULT_SCHEDULE):

        super(DataSetHandler, self).__init__()
        self.dataset = dataset
        self.set_priority({'before_epoch': 10, 'before_batch': 10, 'after_batch': -10, "after_epoch": -10})
        
        if isinstance(batch_size, int):
            batch_size = {'train': batch_size, 'valid': batch_size, 'test': batch_size}
            
        self.batch_size = batch_size
        
        self.workers = workers
        self.timeit = timeit
        self.limits = limits
        self.shuffle = shuffle
        self.collate_fn = None
        self.epoch_schedule = epoch_schedule
        self.time_est = TimeEstimator()

    def before_epoch(self):
        """
        Initializes new epoch: shuffles dataset, prepares dataloader, counts number of iterations in dataloader.
        """
        ds_wrapper = DataSetWrapper(self.dataset[self.trainer._subset], self.trainer._subset)
        drop_last = True if self.trainer._mode == 'train' else False

        shuffle = False
        if isinstance(self.shuffle, dict):
            if self.trainer._mode in self.shuffle:
                shuffle = self.shuffle[self.trainer._mode]
        else:
            shuffle = self.shuffle

        if shuffle:
            ds_wrapper.shuffle()

        if self.collate_fn is None:
            self.collate_fn = self.trainer.collection_op.collate_fn

        self.loader = torch.utils.data.DataLoader(
            ds_wrapper,
            batch_size=self.batch_size[self.trainer._mode],
            shuffle=False,
            num_workers=self.workers,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=self.collate_fn,
            sampler=torch.utils.data.sampler.SequentialSampler(ds_wrapper))

        self.iterator = iter(self.loader)

        if self.trainer._n_iterations is not None:
            self.trainer._n_iterations = min(self.trainer._n_iterations, len(self.loader))
        else:
            self.trainer._n_iterations = len(self.loader)

        if isinstance(self.limits, dict):
            if self.trainer._mode in self.limits:
                self.trainer._n_iterations = min(self.trainer._n_iterations, self.limits[self.trainer._mode])
        else:
            self.trainer._n_iterations = min(self.trainer._n_iterations, self.limits)
        self.time_est.reset()

    def before_batch(self):
        """
        Samples a batch from dataloader and measures the
        time used for it.
        """

        self.start_time = time.time()
        self.trainer._input, self.trainer._ids = next(self.iterator)
        self.data_time = time.time() - self.start_time

        progress = collections.OrderedDict()
        progress['Ep'] = str(self.trainer._epoch)
        if hasattr(self.trainer, '_n_epochs'):
            progress['Ep'] += '/' + str(self.trainer._n_epochs)

        percentage = float(self.trainer._epoch_iteration) / float(self.trainer._n_iterations)
        self.time_est.update(percentage)
        progress['Mode'] = self.trainer._mode
        progress['Subset'] = self.trainer._subset
        progress['Iter'] = str(self.trainer._epoch_iteration) + '/' + str(self.trainer._n_iterations)
        progress['Iter'] += ' ' + progress_str(20, percentage)
        progress['Iter'] += ' ' + str(int(percentage * 1000.0) / 10.0) + '%'
        progress['Time'] = str(self.time_est)

        self.trainer.status['Progress'] = progress

    def after_batch(self):
        """
        Releases the 'self.trainer._input' and 'self.trainer._ids'. Also
        evaluates batch time.
        """

        del self.trainer._input, self.trainer._ids

        self.batch_time = time.time() - self.start_time

        if not hasattr(self, 'avg_data_time'):
            self.avg_data_time = 0
        if not hasattr(self, 'avg_batch_time'):
            self.avg_batch_time = 0
        if not hasattr(self, 'n_iter'):
            self.n_iter = 0

        self.avg_data_time = (self.avg_data_time * self.n_iter + self.data_time)
        self.avg_batch_time = (self.avg_batch_time * self.n_iter + self.batch_time)
        self.n_iter += 1
        self.avg_data_time /= self.n_iter
        self.avg_batch_time /= self.n_iter


        if self.timeit:
            if 'Info' not in self.trainer.status:
                self.trainer.status['Time'] = collections.OrderedDict()

            self.trainer.status['Time']['D'] = self.data_time
            self.trainer.status['Time']['B'] = self.batch_time
            self.trainer.status['Time']['AvgD'] = self.avg_data_time
            self.trainer.status['Time']['AvgB'] = self.avg_batch_time
            # self.trainer.status['Time']['Elapsed'] = self.elapsed_time
            # self.trainer.status['Time']['Remaining'] = self.remaining_time

    def on_train(self):
        """
        Cycles through the epochs. For epoch details, use
        `view_epoch()`.
        """

        while True:
            if hasattr(self.trainer, '_n_epochs'):
                if self.trainer._epoch >= self.trainer._n_epochs:
                    break

            for regime in self.epoch_schedule:
                self.trainer.run_epoch(**regime)

            if hasattr(self.trainer, '_stop_train_signal'):
                if self.trainer._stop_train_signal:
                    del self.trainer._stop_train_signal
                    break

    def on_epoch(self):
        """
        Cycles through batches. For batch details, use
        `view_batch()`.
        """
        while True:
            if self.trainer._epoch_iteration >= self.trainer._n_iterations:
                break

            self.trainer.run_batch()

            if hasattr(self.trainer, '_stop_epoch_signal'):
                if self.trainer._stop_epoch_signal:
                    # del self.trainer._stop_epoch_signal
                    break

    def after_epoch(self):
        """
        Deletes dataloader.
        """
        del self.loader, self.iterator
