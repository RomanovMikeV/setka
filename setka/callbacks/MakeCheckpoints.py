import os
import pickle
import time

from .Callback import Callback

class MakeCheckpoints(Callback):
    '''
    This callback makes checkpoints during an experiment. Two checkpoints
    are stored: the best one (with "_best" postfix) (created only in case the
    needed metrics has already been computed) and the last one
    (with "_last" postfix). You should specify the
    metrics that will be monitored to select the best model.

    The checkpoint is computed when epoch starts (on_epoch_begin).

    The checkpoints are saved in the directory ```checkpoints``` and
    in a directory specified in ```trainer._checkpoints_dir```. If the
    ```checkpoints``` directory does not exist -- it will be created.
    '''
    def __init__(self,
                 metric,
                 subset='valid',
                 max_mode=False,
                 name='checkpoint'):

        '''
        Constructor.

        Args:
            metric (str): name of the metric to monitor

            subset (hashable): name of the subset on which the metric will be
                monitored

            max_mode (bool): if True, the higher the metric -- the better the
                model.

            name (str): name of the checkpoint file.
        '''
        self.best_metric = None
        self.metric = metric
        self.max_mode = max_mode
        self.name = name
        self.subset = subset
        self.set_priority(1000)

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')


    # @staticmethod
    # def create_dir(fname):
    #     dirname = os.path.dirname(fname)
    #     if not os.path.exists(dirname):
    #         os.makedirs(dirname)


    def on_epoch_begin(self):

        is_best = False

        if self.trainer.status["mode"] == 'train':
            if hasattr(self.trainer, '_metrics'):
                if self.subset in self.trainer._metrics:
                    if self.metric in self.trainer._metrics[self.subset]:
                        if self.best_metric is None:
                            self.best_metric = (
                                self.trainer._metrics[self.subset][self.metric])
                            is_best = True

                        if ((self.best_metric < self.trainer._metrics[self.subset][self.metric] and
                             self.max_mode) or
                            (self.best_metric > self.trainer._metrics[self.subset][self.metric] and
                             not self.max_mode)):

                             self.best_metric = self.trainer._metrics[self.subset][self.metric]
                             is_best = True

            with open(os.path.join(
                    './checkpoints',
                    self.name + '_latest.pth.tar'), 'wb+') as fout:

                pickle.dump(self, fout)

            if is_best:
                with open(os.path.join(
                        './checkpoints',
                        self.name + '_best.pth.tar'), 'wb+') as fout:

                    pickle.dump(self, fout)
