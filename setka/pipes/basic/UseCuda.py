from setka.pipes.Pipe import Pipe


class UseCuda(Pipe):
    """
    This pipe moves the tensors and the model to cuda when it is needed.
    """
    def __init__(self, device='cuda:0'):
        super(UseCuda, self).__init__()
        self.device = device

    def before_epoch(self):
        """
        Moves model to GPU.
        """
        self.trainer._model.to(device=self.device)
    
    def before_batch(self):
        """
        Moves batch to GPU.
        """
        self.trainer._input = self.trainer.collection_op.to(self.trainer._input, device=self.device)
