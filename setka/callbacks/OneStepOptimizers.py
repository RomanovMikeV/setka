from .Callback import Callback

class OneStepOptimizers(Callback):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def on_init(self):
        self.trainer._optimizers = self.optimizers

    def on_batch_begin(self):
        if self.trainer._mode == 'train':
            print("Switching to train mode")
            for optimizer in self.trainer._optimizers:
                optimizer.optimizer.zero_grad()

            # Switch the necessary layers to training mode
            for optimizer in self.trainer._optimizers:
                optimizer.module.train()

    def on_batch_end(self):
        for optimizer in self.trainer._optimizers:
            optimizer.optimizer.step()
            
        for optimizer in self.trainer._optimizers:
            optimizer.module.eval()

