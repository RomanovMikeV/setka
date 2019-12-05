from .Pipe import Pipe

import torch

class ModelHandler(Pipe):
    '''
    One of the core pipes. Knows how to handle simple models correctly.

    Stores:
        model's output in 'self.trainer._output'.

    Args:
        model (torch.nn.Module): model to handle.
    '''
    def __init__(self, model):
        self.model = model
        self.set_priority({'after_batch': -10,
                           'on_batch': 10})

    def on_init(self):
        self.trainer._model = torch.nn.DataParallel(self.model)
        self.trainer._model.eval()

    def before_epoch(self):
        '''
        Switches all the model's modules to the evaluation mode.
        '''
        self.trainer._model.eval()
        
    def on_batch(self):
        '''
        Performs forward pass through the model. Also switches model to eval mode in case the
        trainer's mode is not 'train'.
        '''
        if self.trainer._mode != 'train':
            self.trainer._model.eval()
        self.trainer._output = self.trainer._model(self.trainer._input)
        
    def after_batch(self):
        '''
        Releases self.trainer._output
        '''
        del self.trainer._output
