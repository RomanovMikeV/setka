import os
import torch

from .Pipe import Pipe

class MakeCheckpoints(Pipe):
    '''
    This pipe makes checkpoints during an experiment. Two checkpoints
    are stored: the best one (with "_best" postfix) (created only in case the
    needed metrics has already been computed) and the last one
    (with "_last" postfix). You should specify the
    metrics that will be monitored to select the best model.

    The checkpoint is computed when epoch starts (before_epoch).

    The checkpoints are saved in the directory ```checkpoints``` and
    in a directory specified in ```trainer._checkpoints_dir```. If the
    ```checkpoints``` directory does not exist -- it will be created.

    Args:
        metric (str): name of the metric to monitor

        subset (hashable): name of the subset on which the metric will be
            monitored

        max_mode (bool): if True, the higher the metric -- the better the
            model.

        name (str): name of the checkpoint file.

        log_dir (str): path to the directory, where the checkpoints will be saved.

        keep_best_only (bool): if True, only the last model and the best models are kept.
    '''
    def __init__(self,
                 metric,
                 subset='valid',
                 max_mode=False,
                 name='checkpoint',
                 log_dir='./',
                 keep_best_only=True):

        self.best_metric = None
        self.metric = metric
        self.max_mode = max_mode
        self.name = name
        self.subset = subset
        self.set_priority(1000)
        self.log_dir = log_dir

        self.keep_best_only=keep_best_only

        if not os.path.exists(os.path.join(self.log_dir, 'checkpoints')):
            os.makedirs(os.path.join(self.log_dir, 'checkpoints'))


    def on_init(self):
        self.log_dir = os.path.join(
            self.log_dir,
            'logs',
            self.name,
            str(self.trainer.creation_time)
        )

        if not os.path.exists(os.path.join(self.log_dir, 'checkpoints')):
            os.makedirs(os.path.join(self.log_dir, 'checkpoints'))

    def before_epoch(self):
        '''
        The checkpoints are being saved.
        '''
        is_best = False

        if "mode" in self.trainer.status:
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

                torch.save({'trainer': self.trainer},
                           os.path.join(
                                self.log_dir,
                                'checkpoints',
                                self.name + '_latest.pth.tar'))

                torch.save(self.trainer._model.state_dict(),
                           os.path.join(
                                self.log_dir,
                                'checkpoints',
                                self.name + '_weights_latest.pth.tar'))

                if is_best:
                    torch.save({'trainer': self.trainer},
                            os.path.join(
                            self.log_dir,
                            'checkpoints',
                            self.name + '_best.pth.tar'))

                    torch.save(self.trainer._model.state_dict(),
                            os.path.join(
                            self.log_dir,
                            'checkpoints',
                            self.name + '_weights_best.pth.tar'))


                if not self.keep_best_only:
                    torch.save({'trainer': self.trainer},
                            os.path.join(
                            self.log_dir,
                            'checkpoints',
                            self.name + '_' + str(self.trainer._epoch - 1) + '.pth.tar'))

                    torch.save(self.trainer._model.state_dict(),
                            os.path.join(
                            self.log_dir,
                            'checkpoints',
                            self.name + '_weights_' + str(self.trainer._epoch - 1) + '.pth.tar'))
            
