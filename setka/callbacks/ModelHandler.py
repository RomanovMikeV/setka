from .Callback import Callback

import torch

class ModelHandler(Callback):
    def __init__(self, model):
        self.model = model
        self.set_priority({'on_batch_end': -10,
                           'on_batch_run': 10})

    def on_init(self):
        self.trainer._model = torch.nn.DataParallel(self.model)
        self.trainer._model.eval()

    def on_epoch_begin(self):
        self.trainer._model.eval()
        
    def on_batch_run(self):
        if self.trainer._mode != 'train':
            self.trainer._model.eval()
        self.trainer._output = self.trainer._model({'image': torch.zeros_like(self.trainer._input['image'])})
        
    def on_batch_end(self):
        del self.trainer._output
