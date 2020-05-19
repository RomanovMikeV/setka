import os
import torch

from setka.pipes.Pipe import Pipe
from setka.base import collect_random_states


class Checkpointer(Pipe):
    """
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
        subset (hashable): name of the subset on which the metric will be monitored
        max_mode (bool): if True, the higher the metric -- the better the model.
        name (str): name of the checkpoint file.
        log_dir (str): path to the directory, where the checkpoints will be saved.
        keep_best_only (bool): if True, only the last model and the best models are kept.
        checkpoint_freq (int): Number of epochs between checkpoint
        dump_trainer (bool): Make full trainer dumps. Useful for training resume and experiments reproduction
        train_only (bool): Make trainer dumps only after train stage. Otherwise, trainer will be saved after each stage
                           (including validation and testing). Useful for training resume and experiments reproduction
    """
    def __init__(self, metric, subset='valid', max_mode=False, name='experiment', log_dir='runs', keep_best_only=True,
                 checkpoint_freq=1, dump_trainer=True, train_only=False):
        super(Checkpointer, self).__init__()
        self.best_metric = None
        self.metric = metric
        self.max_mode = max_mode
        self.name = name
        self.subset = subset
        self.set_priority({'before_epoch': 1000, 'after_epoch': -1000})
        self.log_dir = log_dir
        self.keep_best_only = keep_best_only
        self.dump_trainer = dump_trainer
        self.train_only = train_only
        self.checkpoint_freq = checkpoint_freq

    def on_init(self):
        self.log_dir = os.path.join(self.log_dir, self.name, str(self.trainer.creation_time).replace(' ', '_').replace(':', '-'))

        if not os.path.exists(os.path.join(self.log_dir, 'checkpoints')):
            os.makedirs(os.path.join(self.log_dir, 'checkpoints'))

    def dump(self, postfix):
        if self.dump_trainer:
            torch.save({'trainer': self.trainer, 'random_states': collect_random_states()},
                       os.path.join(self.log_dir, 'checkpoints', self.name + f'_{postfix}.pth.tar'))

        torch.save(self.trainer._model.state_dict(),
                   os.path.join(self.log_dir, 'checkpoints', self.name + f'_weights_{postfix}.pth.tar'))

    def checkpoint_epoch(self, epoch_n=None):
        is_best = False

        if hasattr(self.trainer, '_metrics') and (self.subset in self.trainer._metrics) \
                and (self.metric in self.trainer._metrics[self.subset]):

            if self.best_metric is None:
                self.best_metric = (self.trainer._metrics[self.subset][self.metric])
                is_best = True

            metric_increased = self.trainer._metrics[self.subset][self.metric] > self.best_metric

            if metric_increased == bool(self.max_mode):
                self.best_metric = self.trainer._metrics[self.subset][self.metric]
                is_best = True

        self.dump('latest')
        if is_best:
            self.dump('best')
        if not self.keep_best_only:
            self.dump(epoch_n if epoch_n is not None else self.trainer._epoch - 1)

    def before_epoch(self):
        """
        The checkpoints are being saved.
        """
        if hasattr(self.trainer, "_mode") and (self.trainer._mode == 'train') and (self.trainer._epoch == 1):
            self.checkpoint_epoch(self.trainer._epoch - 1)

    def after_epoch(self):
        """
        The checkpoints are being saved.
        """
        if hasattr(self.trainer, "_mode") and not (self.trainer._mode != 'train' and self.train_only):
            if self.trainer._epoch % self.checkpoint_freq == 0:
                self.checkpoint_epoch(self.trainer._epoch)
