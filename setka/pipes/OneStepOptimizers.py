from .Pipe import Pipe

class OneStepOptimizers(Pipe):
    '''
    This pipe takes care of the optimization process.

    Stores:
        self.trainer._optimizers: list of optimizers for a model.

    Args:
        optimizers (list of setka.bas.OptimizerSwitch): list of optimizers.
    '''
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def on_init(self):
        self.trainer._optimizers = self.optimizers

    def before_batch(self):
        '''
        Zeros grad for the active optimizers, turns modules with active optimizers to the training mode.
        '''
        if self.trainer._mode == 'train':
            for optimizer in self.trainer._optimizers:
                if optimizer.active:
                    optimizer.optimizer.zero_grad()
                    optimizer.module.train()
                    optimizer.module.requires_grad = True

    def after_batch(self):
        '''
        Active optimizers make step.
        '''
        if self.trainer._mode == 'train':
            for optimizer in self.trainer._optimizers:
                if optimizer.active:
                    optimizer.optimizer.step()

        self.trainer._model.eval()
        self.trainer._model.requires_grad = False