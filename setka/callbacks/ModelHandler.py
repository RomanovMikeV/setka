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
        print(self.trainer._input, '<- input')
        self.trainer._output = self.trainer._model(self.trainer._input)
#         print(self.trainer._output.min(), self.trainer._output.max(), '<- output min max')

    def on_batch_end(self):
        del self.trainer._output
