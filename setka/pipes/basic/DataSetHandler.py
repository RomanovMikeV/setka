import time

import numpy
import torch

import setka.base
from setka.pipes.Pipe import Pipe

DEFAULT_SCHEDULE = [
     {'mode': 'train', 'subset': 'train'},
     {'mode': 'valid', 'subset': 'train'},
     {'mode': 'valid', 'subset': 'valid'}
]


class DataSetWrapper:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

        self.order = numpy.arange(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        real_index = self.order[index]
        dataset_res = self.dataset[real_index]

        return dataset_res, str(self.name) + '_' + str(real_index)

    def shuffle(self):
        self.order = numpy.random.permutation(len(self.dataset))
        # print('Shuffled order:', self.order[:16])


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
            batch_size=self.batch_size[self._mode],
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

    def before_batch(self):
        """
        Samples a batch from dataloader and measures the
        time used for it.
        """

        self.start_time = time.time()
        self.trainer._input, self.trainer._ids = next(self.iterator)
        self.data_time = time.time() - self.start_time

    def after_batch(self):
        """
        Releases the 'self.trainer._input' and 'self.trainer._ids'. Also
        evaluates batch time.
        """

        del self.trainer._input, self.trainer._ids

        self.batch_time = time.time() - self.start_time

        if self.timeit:
            self.trainer.status['D'] = self.data_time
            self.trainer.status['B'] = self.batch_time

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
