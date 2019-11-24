from .Pipe import Pipe

class LossHandler(Pipe):
    '''
    Handles loss functions.

    Stores:
        self.trainer._loss -- loss value for the model

    Args:
        criterion (callable): loss function
    '''
    def __init__(self, criterion):
        self.criterion = criterion

        self.set_priority({'on_batch': 9, 'after_batch': -9})

    def on_batch(self):
        '''
        Computes loss in case self.trainer is in mode 'train' or 'valid'.
        '''
        if self.trainer._mode in ["train", "valid"]:
            self.trainer._loss = self.criterion(self.trainer._output, self.trainer._input)
            if self.trainer._mode == "train":
                self.trainer._loss.backward()

            self.trainer.status['loss'] = self.trainer._loss.detach().cpu().item()

    def after_batch(self):
        '''
        Releases loss value in case it is present.
        '''
        if self.trainer._mode in ["train", "valid"]:
            del self.trainer._loss
