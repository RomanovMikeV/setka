from .Callback import Callback

class OneStepOptimizers(Callback):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def on_init(self):
        self.trainer._optimizers = self.optimizers

    def on_batch_begin(self):
        if self.trainer._mode == 'train':
            for optimizer in self.trainer._optimizers:
                if self.optimizer.is_active:
                    optimizer.optimizer.zero_grad()
                    optimizer.module.train()

    def on_batch_end(self):
        if self.trainer._mode == 'train':
            for optimizer in self.trainer._optimizers:
                if optimizer.is_active:
                    optimizer.optimizer.step()
                optimizer.module.eval()

