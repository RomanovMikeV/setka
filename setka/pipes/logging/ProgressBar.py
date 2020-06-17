import datetime
import time
import os
import collections
import numpy as np

from setka.pipes.Pipe import Pipe
from setka.pipes.logging.progressbar import StageProgressBar, isnotebook
from setka.pipes.logging.progressbar import progress_str
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

    def __init__(self, theme=None, default_width=80):
        super(ProgressBar, self).__init__()
        self.set_priority(-100)

        self.config = theme if theme is not None else main_theme()
        if (theme is None) and isnotebook():
            self.config = adopt2ipython(self.config)
        self.last_epoch = 0
        self.pbar = None
        self.time_est = TimeEstimator()
        self.display_counter = 0
        self.default_width = default_width

    def on_init(self):
        self.last_epoch = self.safe_getattr(self.trainer, '_epoch', 0)

    def get_width(self):
        try:
            return os.get_terminal_size()[0] - 1
        except :
            return self.default_width

    @staticmethod
    def safe_getattr(obj, attr, default=None):
        if hasattr(obj, '__getitem__'):
            return obj[attr] if attr in obj else default
        return getattr(obj, attr) if hasattr(obj, attr) else default


    def before_epoch(self):
        self.display_counter += 1
        if self.last_epoch != self.safe_getattr(self.trainer, '_epoch', 0):
            print('-' * self.get_width())
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
        progress = collections.OrderedDict()
        progress['Ep'] = str(self.trainer._epoch)
        if hasattr(self.trainer, '_n_epochs'):
            progress['Ep'] += '/' + str(self.trainer._n_epochs)

        percentage = float(self.trainer._epoch_iteration) / float(self.trainer._n_iterations)
        progress['Mode'] = self.trainer._mode
        progress['Subset'] = self.trainer._subset
        progress['Iter'] = str(self.trainer._epoch_iteration) + '/' + str(self.trainer._n_iterations)
        progress['Iter'] += ' ' + progress_str(20, percentage)
        progress['Iter'] += ' ' + str(int(percentage * 1000.0) / 10.0) + '%'

        self.time_est.update(percentage)
        progress['Time'] = str(self.time_est)
        self.trainer.status['Progress'] = progress

        self.pbar.update(self.trainer.status)
