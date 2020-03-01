from setka.pipes.Pipe import Pipe
from setka.base.OptimizerSwitch import OptimizerSwitch


class OneStepOptimizers(Pipe):
    """
    This pipe takes care of the optimization process.

    Attributes:
        self.trainer._optimizers: list of optimizers for a model.

    Args:
        optimizers (list of setka.base.OptimizerSwitch): list of optimizers.
    """
    def __init__(self, optimizers):
        super(OneStepOptimizers, self).__init__()
        self.optimizers = optimizers

    def on_init(self):
        self.trainer._optimizers = self.optimizers
        for opt in self.optimizers:
            if isinstance(opt, OptimizerSwitch):
                opt.trainer = self.trainer

    def before_batch(self):
        """
        Zeros grad for the active optimizers, turns modules with active optimizers to the training mode.
        """
        if self.trainer._mode == 'train':
            for optimizer in self.trainer._optimizers:
                if optimizer.active:
                    optimizer.optimizer.zero_grad()
                    optimizer.module.train()
                    optimizer.module.requires_grad = True

    def after_batch(self):
        """
        Active optimizers make step.
        """
        if self.trainer._mode == 'train':
            for optimizer in self.trainer._optimizers:
                if optimizer.active:
                    optimizer.optimizer.step()

        self.trainer._model.eval()
        self.trainer._model.requires_grad = False
