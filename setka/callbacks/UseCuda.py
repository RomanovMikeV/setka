from .Callback import Callback
import torch

class UseCUDA(Callback):
    def __init__(self):
        pass

    def on_init(self):
        self.trainer._model.cuda()

    def on_batch_begin(self):
        if torch.cuda.is_available():
            for index in range(len(self.trainer._input)):
                self.trainer._input[index] = self.trainer._input[index].cuda()
