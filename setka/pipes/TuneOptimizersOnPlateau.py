from .Pipe import Pipe
import copy

class TuneOptimizersOnPlateau(Pipe):
    '''
    This pipe performs Learning Rate reducing when the plateau
    is reached.

    Args:
        metric (str) : name of the metric to monitor.

        subset (hashable) : key of the subset of training set on which metric
            will be monotored.

        cooldown (int) : minimal amount of epochs between two learning rate changes.

        patience (int) : amount of epochs since the last improvement of maximum of
            the monitored metric before the learning rate change.

        lr_factor (float): learning rate factor. New learning rate will
            be an old one times the lr_factor.

        m_power (float): momentum power. New momentum will be an old momentum
            power m_power.

        max_mode (bool) : if True then the higher is the metric the better.
            Otherwise the lower is the metric the better.

        reset_optimizer (bool) : if True, the optimizers will be completely reset when
            the plateau is reached.
    '''

    def __init__(self,
                 metric,
                 subset='valid',
                 cooldown=5,
                 patience=5,
                 tolerance=1.0e-3,
                 lr_factor=1.0,
                 m_power=1.0,
                 max_mode=False,
                 reset_optimizer=False):

        self.cooldown = cooldown
        self.patience = patience
        self.lr_factor = lr_factor
        self.m_power = m_power
        self.tolerance = tolerance

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


    def before_epoch(self):
        '''
        Informs the user that the plateau has been reached.
        '''
        if 'OnPlateau' in self.trainer.status:
            del(self.trainer.status["OnPlateau"])


    def update_best(self):

        self.best_metric = self.trainer._metrics[self.subset][self.metric]

        self.since_best = 0
        self.best_model_state = self.trainer._model.state_dict()
        self.best_optimizers = copy.deepcopy(self.trainer._optimizers)


    def after_epoch(self):
        '''
        Checks if the plateau is reached. If yes -- the optimizers are tuned.
        '''

        if (self.trainer._mode == 'valid' and
                self.trainer._subset == self.subset and
                self.trainer._lr_reduce):

            if self.best_metric is None:
                self.update_best()

            else:
                new_metric = self.trainer._metrics[self.subset][self.metric]

                if ((new_metric >= self.best_metric + self.tolerance and self.max_mode) or
                    (new_metric < self.best_metric - self.tolerance and not self.max_mode)):

                    self.update_best()

            if self.since_last >= self.cooldown and self.since_best >= self.patience:

                self._lr_mult *= self.lr_factor
                self._m_power *= self.m_power

                self.trainer._model.load_state_dict(self.best_model_state)
                self.trainer._optimizers = self.best_optimizers

                if "OnPlateau" not in self.trainer.status:
                    self.trainer.status["OnPlateau"] = ''

                self.trainer.status["OnPlateau"] += (' LR factor: ' + str(self._lr_mult) +
                        ' Momentum power: ' + str(self._m_power))

                for optimizer in self.trainer._optimizers:
                    for g in optimizer.optimizer.param_groups:
                        if 'lr' in g:
                            g['lr'] *= self.lr_factor
                        if 'momentum' in g:
                            g['momentum'] **= self.m_power
                        if 'betas' in g:
                            new_betas = []
                            for index in range(len(g['betas'])):
                                new_betas.append(g['betas'][index] ** self.m_power)
                            g['betas'] = new_betas

                    if self.reset_optimizer:
                        optimizer.state = {}
                self.since_last = 0

                self.best_model_state = self.trainer._model.state_dict()
                self.best_optimizers = copy.deepcopy(self.trainer._optimizers)

            self.since_best += 1
            self.since_last += 1
