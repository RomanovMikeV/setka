from .Callback import Callback

class LRSearch(Callback):

    def __init__(self, factor=0.3, beta=0.99, threshold=3.0):
        self.lr_factor = factor
        self.beta = beta
        self.t = threshold

    def on_train_begin(self):
        self._found_lr = False

    def on_epoch_begin(self):
        self.network_state = self.trainer._model.state_dict()
        self.optimizers_states = self.trainer.get_optimizers_states()

        self.ema = 0.0
        self.ema_var = 0.0
        self.weight = 0.0

    def on_batch_end(self):

        loss = self.trainer._loss.detach()

        self.ema = self.ema * self.beta + loss * (1.0 - self.beta)
        self.ema_var = self.ema_var * self.beta + (1.0 - self.beta) * (loss - self.ema) ** 2
        self.weight = self.weight * self.beta + (1.0 - self.beta)

        reduce_flag = False
        if torch.isnan(self.trainer._loss):
            reduce_flag = True
        elif self.trainer._loss > (self.ema / self.weight + self.threshold * (self.ema_var / self.weight).sqrt()):
            reduce_flag = True

        if reduce_flag and not self.trainer._found_lr:
            self.trainer._model.load_state_dict(self.network_state)
            self.trainer.set_optimizers_states(self.optimizers_states)

            for optimizer in self.trainer._optimizers:
                for g in optimizer.optimizer.param_groups:
                    if 'lr' in g:
                        g['lr'] *= self.lr_factor


    def on_epoch_end(self):
        self._found_lr = True