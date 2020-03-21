import torch

from setka.pipes.Pipe import Pipe


class ModelHandler(Pipe):
    """
    One of the core pipes. Knows how to handle simple models correctly.

    Stores:
        model's output in 'self.trainer._output'.

    Args:
        model (torch.nn.Module): model to handle.
        data_parallel (bool): If true, DataParallel wrapper is used for model
        device_ids (list): Device ids to use for model training
    """
    def __init__(self, model, data_parallel=True, device_ids=None):
        super(ModelHandler, self).__init__()
        self.model = model
        self.data_parallel = data_parallel
        self.device_ids = device_ids

        self.set_priority({'after_batch': -10, 'on_batch': 10})

    def on_init(self):
        if self.data_parallel:
            self.trainer._model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        else:
            self.trainer._model = self.model

        self.trainer._model.eval()
        self.trainer._model.requires_grad = False

    def before_epoch(self):
        """
        Switches all the model's modules to the evaluation mode.
        """
        self.trainer._model.eval()
        
    def on_batch(self):
        """
        Performs forward pass through the model. Also switches model to eval mode in case the
        trainer's mode is not 'train'.
        """
        if self.trainer._mode != 'train':
            self.trainer._model.eval()

        self.trainer._output = self.trainer._model(self.trainer._input)

    def after_batch(self):
        """
        Releases self.trainer._output
        """
        del self.trainer._output
