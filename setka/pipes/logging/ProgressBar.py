import os

from setka.pipes.Pipe import Pipe
from setka.pipes.logging.progressbar import isnotebook, StageProgressBar
# from setka.pipes.logging.progressbar.theme import adopt2ipython


class ProgressBar(Pipe):
    """
    This pipe shows progress of the training.
    """

    def __init__(self, theme=None, default_width=80):
        super(ProgressBar, self).__init__()
        self.set_priority(-100)

        # self.config = theme if theme is not None else main_theme()
        # if (theme is None) and isnotebook():
        #     self.config = adopt2ipython(self.config)
        self.last_epoch = 0
        self.pbar = None
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

        self.pbar = StageProgressBar(width_function=self.get_width,
                                     display_id='ep{}'.format(self.display_counter))

    def after_epoch(self):
        """
        Clears the progressbar, keeps the status message.
        """
        del self.pbar


    def after_batch(self):
        """
        Updates the progressbar.
        """
        # progress = collections.OrderedDict()
        # progress['Ep'] = str(self.trainer._epoch)
        # if hasattr(self.trainer, '_n_epochs'):
        #     progress['Ep'] += '/' + str(self.trainer._n_epochs)

        # percentage = float(self.trainer._epoch_iteration) / float(self.trainer._n_iterations)
        # progress['Mode'] = self.trainer._mode
        # progress['Subset'] = self.trainer._subset
        # progress_iter = str(self.trainer._epoch_iteration) + '/' + str(self.trainer._n_iterations)
        # progress_iter += ' ' + progress_str(20, percentage)
        # progress_iter += ' ' + str(int(percentage * 1000.0) / 10.0) + '%'

        # self.time_est.update(percentage)
        # progress['Progress']['Time'] = str(self.time_est)
        # self.trainer.status['Progress']['Iter'] = progress_iter

        self.pbar.update(self.trainer.status)
