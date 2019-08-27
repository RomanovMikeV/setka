from .Callback import Callback

class TuneOptimizersOnPlateau(Callback):
    '''
    This callback performs Learning Rate reducing when the plateau
    is reached.

    Args:
        metric (str) : name of the metric to monitor.

        subset (hashable) : key of the subset of training set on which metric
            will be monotored.

        cooldown (int) : minimal amount of epochs between two learning rate changes.

        limit (int) : amount of epochs since the last improvement of maximum of
            the monitored metric before the learning rate change.

        lr_factor (float): learning rate factor. New learning rate will
            be an old one times the lr_factor.

        m_power (float): momentum power. New momentum will be an old momentum
            power m_power.

        max_mode (bool) : if True then the higher is the metric the better.
            Otherwise the lower is the metric the better.
    '''

    def __init__(self,
                 metric,
                 subset='valid',
                 cooldown=5,
                 limit=5,
                 lr_factor=1.0,
                 m_power=1.0,
                 max_mode=False,
                 reset_optimizer=False):

        self.cooldown = cooldown
        self.limit = limit
        self.lr_factor = lr_factor
        self.m_power = m_power

        self.since_last = 0
        self.since_best = 0

        self.best_metric = None
        self.subset = subset
        self.metric = metric
        self.max_mode = max_mode

        self.best_model = None
        self.reset_optimizer = reset_optimizer

        self._lr_mult = 1.0
        self._m_power = 1.0

    def on_init(self):
        self.trainer._lr_reduce = True

    def on_epoch_end(self):
        if (self.trainer._mode == 'validating' and
                self.trainer._subset == self.subset and
                self.trainer._lr_reduce):

            if self.best_metric is None:
                self.best_metric = self.trainer._metrics[self.subset][self.metric]
                self.since_best = 0
                self.best_model_state = self.trainer._model.state_dict()
                self.best_optimizers_state = self.trainer.get_optimizers_states()

            else:
                new_metric = self.trainer._metrics[self.subset][self.metric]

                if ((new_metric > self.best_metric and self.max_mode) or
                        (new_metric < self.best_metric and not self.max_mode)):
                    self.best_metric = new_metric
                    self.since_best = 0
                    self.best_model_state = self.trainer._model.state_dict()
                    self.best_optimizers_state = self.trainer.get_optimizers_states()

            if self.since_last >= self.cooldown and self.since_best >= self.limit:

                self._lr_mult *= self.lr_factor
                self._m_power *= self.m_power

                self.trainer._model.load_state_dict(self.best_model_state)
                self.trainer.set_optimizers_states(self.best_optimizers_state)

                if "action" not in self.trainer._status:
                    self.trainer._status["action"] = ""
                self.trainer._status["action"] += (
                        ' LR factor: ' + str(self._lr_mult) +
                        ' Momentum power: ' + str(self._m_power))
                for optimizer in self.trainer._optimizers:
                    for g in optimizer.optimizer.param_groups:
                        if 'lr' in g:
                            g['lr'] *= self.lr_factor
                        if 'momentum' in g:
                            g['momentum'] **= self.m_power
                        if 'beta' in g:
                            g['beta'] **= self.m_power
                        if 'betas' in g:
                            for index in range(len(g['betas'])):
                                g['betas'][index] **= self.m_power

                    if self.reset_optimizer:
                        optimizer.state = {}
                self.since_last = 0

                self.best_model_state = self.trainer._model.state_dict()
                self.best_optimizers_state = self.trainer.get_optimizers_states()

            self.since_best += 1
            self.since_last += 1