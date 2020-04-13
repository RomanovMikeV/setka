import datetime
import time
import os
import numpy as np

from setka.pipes.Pipe import Pipe
from setka.pipes.logging.progressbar import StageProgressBar, isnotebook
from setka.pipes.logging.progressbar import progress_str, recursive_substituion
from setka.pipes.logging.progressbar.theme import main_theme, adopt2ipython


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


class ProgressBar(Pipe):
    """
    This pipe shows progress of the training.
    """

    def __init__(self, theme=None, default_width=100):
        super(ProgressBar, self).__init__()
        self.set_priority(-100)

        self.config = theme if theme is not None else main_theme()
        if (theme is None) and isnotebook():
            self.config = adopt2ipython(self.config)
        self.last_epoch = 0
        self.pbar = None
        self.time_est = TimeEstimator()
        self.display_counter = 0
        self.default_width = 100

    def on_init(self):
        self.last_epoch = self.safe_getattr(self.trainer, '_epoch', 0)

    def get_width(self):
        try:
            return os.get_terminal_size()[0]
        except :
            return self.default_width

    @staticmethod
    def safe_getattr(obj, attr, default=None):
        if hasattr(obj, '__getitem__'):
            return obj[attr] if attr in obj else default
        return getattr(obj, attr) if hasattr(obj, attr) else default

    def collect_info(self):
        cur_iter = self.safe_getattr(self.trainer, '_epoch_iteration', 0)
        max_iter = self.safe_getattr(self.trainer, '_n_iterations', 1)
        percentage = cur_iter / max_iter

        self.time_est.update(percentage)
        if hasattr(self.trainer, '_loss_values'):
            losses = [{'loss_name': 'Loss', 'loss_value': float(self.trainer._loss)}]
            losses += [{'loss_name': name, 'loss_value': float(val)} for name, val in self.trainer._loss_values.items()]
        else:
            losses = []

        if hasattr(self.trainer, '_avg_metrics'):
            metrics = [{'metric_name': name, 'metric_value': val}
                       for name, val in self.trainer._avg_metrics.items()]
        else:
            metrics = []

        return {
            'COMMON': {},
            'cur_epoch': self.safe_getattr(self.trainer, '_epoch', 0),
            'n_epochs': self.safe_getattr(self.trainer, '_n_epochs', 0),
            'cur_stage': self.safe_getattr(self.trainer, '_mode', 'unknown'),
            'cur_subset': self.safe_getattr(self.trainer, '_subset', 'unknown'),
            'cur_iter': cur_iter,
            'max_iter': max_iter,
            'time_str': str(self.time_est),
            'percentage_completed': percentage * 100.0,
            'progress_bar': progress_str(30, percentage),
            'batch_time': self.safe_getattr(self.trainer.status, 'B', 0),
            'data_time': self.safe_getattr(self.trainer.status, 'D', 0),
            'loss': losses,
            'metrics': metrics
        }

    def before_epoch(self):
        self.display_counter += 1
        if self.last_epoch != self.safe_getattr(self.trainer, '_epoch', 0):
            # Epoch number changed, display epoch delimiter
            print('-' * self.get_width())
            print(recursive_substituion('epoch_global', self.collect_info(), self.config))
            self.last_epoch = self.safe_getattr(self.trainer, '_epoch', 0)

        self.pbar = StageProgressBar(width_function=self.get_width, config=self.config,
                                     display_id='ep{}'.format(self.display_counter))
        self.time_est.reset()

    def after_epoch(self):
        """
        Clears the progressbar, keeps the status message.
        """
        del self.pbar

    def after_batch(self):
        """
        Updates the progressbar.
        """
        self.pbar.update(self.collect_info())
