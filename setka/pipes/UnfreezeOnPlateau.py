from .Pipe import Pipe

class UnfreezeOnPlateau(Pipe):
    '''
    This pipe allows you to unfreeze optimizers one by one when the plateau
    is reached. If used together with ReduceLROnPlateau, should be listed after
    it. It temporarily turns off LR reducing and, instead of it, performs
    unfreezing.
    '''
    def __init__(self,
                 metric,
                 subset='valid',
                 cooldown=5,
                 limit=5,
                 max_mode=False):

        '''
        Constructor.

        Args:
            metric (str) : name of the metric to monitor.

            cooldown (int) : minimal amount of epochs between two learning rate changes.

            limit (int) : amount of epochs since the last improvement of maximum of
                the monitored metric before the learning rate change.

            max_mode (bool) : if True then the higher is the metric the better.
                Otherwise the lower is the metric the better.
        '''

        self.cooldown = cooldown
        self.limit = limit

        self.since_last = 0
        self.since_best = 0

        self.best_metric = None
        self.subset = subset
        self.metric = metric
        self.max_mode = max_mode

        self.optimizer_index = 1


    def on_init(self):
        if hasattr(self.trainer, '_lr_reduce'):
            self.trainer._lr_reduce = False
        self.complete = False


    def before_epoch(self):
        if "OnPlateau" in self.trainer.status:
            del(self.trainer.status["OnPlateau"])

    def after_epoch(self):
        if (self.trainer._mode == 'valid' and
            self.trainer._subset == self.subset):

            if self.best_metric is None:
                self.best_metric = (
                    self.trainer._metrics[self.subset][self.metric])
                self.since_best = 0

            else:
                new_metric = self.trainer._metrics[self.subset][self.metric]

                if ((new_metric > self.best_metric and self.max_mode) or
                    (new_metric < self.best_metric and not self.max_mode)):

                    self.best_metric = new_metric
                    self.since_best = 0


            if self.since_last >= self.cooldown and self.since_best >= self.limit and not self.complete:
                if "OnPlateau" not in self.trainer.status:
                    self.trainer.status["OnPlateau"] = ""
                self.trainer.status["OnPlateau"] += " Unfreezing (optimizer " + str(self.optimizer_index) + ") *** "
                self.trainer._optimizers[self.optimizer_index].is_active = True
                self.since_last = 0
                self.optimizer_index += 1
                if self.optimizer_index >= len(self.trainer._optimizers):
                    self.trainer._lr_reduce = True
                    self.complete = True


            self.since_best += 1
            self.since_last += 1