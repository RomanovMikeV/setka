from .Callback import Callback

import setka.base
import torch
import time

class DataSetHandler(Callback):
    def __init__(self,
                 dataset,
                 batch_size,
                 workers=0,
                 timeit=True,
                 limits={},
                 shuffle={'train': True}):

        self.dataset = dataset
        self.set_priority({'on_epoch_begin':10, 'on_batch_begin': 10, 'on_batch_end': -10, "on_epoch_end": -10})
        self.batch_size = batch_size
        self.workers = workers
        self.timeit = timeit
        self.limits = limits
        self.shuffle = shuffle


    def on_epoch_begin(self):
        ds_wrapper = setka.base.DataSetWrapper(
            self.dataset[self.trainer._subset],
            self.trainer._subset)

        drop_last = False
        if self.trainer.status._mode == 'train':
            drop_last = True

        shuffle = False
        if isinstance(self.shuffle, dict):
            if self.trainer.status._mode in self.shuffle:
                shuffle = shuffle

        else:
            shuffle = self.shuffle

        if shuffle:
            ds_wrapper.shuffle()

        self.loader = torch.utils.data.DataLoader(
            ds_wrapper,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            drop_last=drop_last,
            pin_memory=True,
            sampler=torch.utils.data.sampler.SequentialSampler(ds_wrapper))

        self.iterator = iter(self.loader)

        self.trainer._n_iterations = len(self.loader)
        if isinstance(self.limits, dict):
            if self.trainer._mode in self.limits:
                self.trainer._n_iterations = min(
                    self.trainer._n_iterations,
                    self.limits[self.trainer._mode])
        else:
            self.trainer._n_iterations = min(
                self.trainer._n_iterations,
                self.limits)


    def on_batch_begin(self):
        self.start_time = time.time()
        self.trainer._input, self.trainer._ids = next(self.iterator)
        self.data_time = time.time() - self.start_time

    def on_batch_end(self):
        del self.trainer._input, self.trainer._ids

        self.batch_time = time.time() - self.start_time

        if self.timeit:
            self.trainer.status['D'] = self.data_time
            self.trainer.status['B'] = self.batch_time

    def on_epoch_end(self):
        del self.loader, self.iterator


