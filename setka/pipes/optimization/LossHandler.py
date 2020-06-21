import torch

from setka.pipes.Pipe import Pipe
from copy import deepcopy


class LossHandler(Pipe):
    """
    Handles loss functions.

    Stores:
        self.trainer._loss -- loss value for the model

    Args:
        criterion (callable, list of callable): loss function or list of loss functions
        coefs (list): List of loss functions coefficients
        retain_graph (bool): Retain graph after criterion backward call
    """
    def __init__(self, criterion, coefs=None, retain_graph=None):
        super(LossHandler, self).__init__()
        self.retain_graph = retain_graph
        self.criterion = criterion
        if not isinstance(self.criterion, (tuple, list)):
            self.criterion = [self.criterion]
        self.coefs = coefs if coefs is not None else [1.0] * len(self.criterion)

        if len(self.coefs) != len(self.criterion):
            raise RuntimeError('Number of criterion and coefficients are not equal')

        self.set_priority({'on_batch': 9, 'after_batch': -9})

    def formula(self):
        return 'Loss = ' + ' + '.join([f'{coef} * {str(loss)}' for loss, coef in zip(self.criterion, self.coefs)])

    def on_batch(self):
        """
        Computes loss in case self.trainer is in mode 'train' or 'valid'.
        """
        if self.trainer._mode in ["train", "valid"]:
            self.trainer._loss = 0
            self.trainer._loss_values = {}
            with torch.set_grad_enabled(self.trainer._mode == 'train'):
                for cur_coef, cur_criterion in zip(self.coefs, self.criterion):
                    cur_loss = cur_criterion(self.trainer._output, self.trainer._input)
                    self.trainer._loss = self.trainer._loss + cur_coef * cur_loss
                    self.trainer._loss_values[cur_criterion.__name__] = cur_loss.item()

            if self.trainer._mode == "train":
                self.trainer._loss.backward(retain_graph=self.retain_graph)

            self.trainer.status['Loss'] = self.trainer._loss.detach().cpu().item()
            self.trainer.status['Formula'] = self.formula()

    def after_epoch(self):
        """
        Releases loss value in case it is present.
        """
        if self.trainer._mode in ["train", "valid"]:
            if hasattr(self.trainer, '_loss'):
                del self.trainer._loss
            if hasattr(self.trainer, '_loss_values'):
                del self.trainer._loss_values
