import torch

class BottleNeck2d(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_bottle,
                 n_outs,
                 kernel=(3, 3),
                 batchnorms=True):

        self.batchnorms = batchnorms

        self.pipe = torch.nn.Sequential()

        if self.batchnorms:
            self.pipe.add(torch.nn.BatchNorm2d(n_inputs))

        self.pipe.add(torch.nn.Conv2d(n_inputs, n_bottle, kernel=(1, 1)))

        if self.batchnorms:
            self.pipe.add(torch.nn.BatchNorm2d(n_bottle))

        self.pipe.add(torch.nn.Conv2d(n_bottle, n_bottle, kernel=kernel))

        if self.batchnorms:
            self.pipe.add(torch.nn.BatchNorm2d(n_bottle))

        self.pipe.add(torch.nn.Conv2d(n_bottle, n_output, kernel=(1, 1)))

        if self.batchnorms:
            self.pipe.add(torch.nn.BatchNorm2d(n_outputs))

    def forward(self, input):
        return self.pipe(input)

    def __call__(self, input):
        return self.forward(input)


class ResidualBlock(torch.nn.Module):
    def __init__(self, function):
        self.function = function

    def __call__(self, input):
        return input + self.function(input)


class InceptionBlock(torch.nn.Module):
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, input):
        res = []
        for f in self.functions:
            res.append(f(input))

        return torch.cat(res, dim=0)
