import torch

class Residual(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def __call__(self, input):
        return self.block(input) + input