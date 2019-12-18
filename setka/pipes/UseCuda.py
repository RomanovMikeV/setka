import torch
import numpy as np

from .Pipe import Pipe


class UseCuda(Pipe):
    """
    This pipe moves the tensors and the model to cuda when it is needed.
    """
    def __init__(self, device='cuda:0'):
        super(UseCuda, self).__init__()
        self.device = device

    def move2cuda(self, coll):
        if isinstance(coll, dict):
            for key in coll:
                coll[key] = self.move2cuda(coll[key])
        elif isinstance(coll, (list, tuple)):
            new_coll = []
            for val in coll:
                new_coll.append(self.move2cuda(val))
            coll = new_coll
        elif isinstance(coll, torch.Tensor):
            coll = coll.to(device=self.device)
        else:
            try:
                coll = torch.as_tensor(coll).to(device=self.device)
            except RuntimeError:
                raise RuntimeError('Unknown data type used as input: ' + str(type(coll)) + '. Can`t move to CUDA')

        return coll

    def before_epoch(self):
        """
        Moves model to GPU.
        """
        self.trainer._model.to(device=self.device)
    
    def before_batch(self):
        """
        Moves batch to GPU.
        """
        self.move2cuda(self.trainer._input)
