import torch

class BottleNeck(torch.nn.Module):
    '''
    Block of 2-dimentional bottleneck. Squeezes amount of the channels, feeds
    squeezed tensor into the kenel submodule, then unsqueezes the tensor again.
    The behavior is as follows:
    ```
    n_in -> (1x1) conv -> batchnorm ->
    kernel ->
    (1x1) conv -> batchnorm -> n_out
    ```
    '''
    def __init__(self,
                 kernel,
                 n_in,
                 n_bottle,
                 n_out,
                 batchnorms=True,
                 n_dims=2):

        '''
        Constructor for the bottleneck

        Args:
            kernel (torch.nn.Module): kernel transform

            n_in (int): number of input channels to the bottleneck

            n_bottle (int): number of bottleneck channels.

            n_out (int): number of output channels from the bottleneck

            batchnorms (bool): whether to use batchnorms after convolutions
                of the bottleneck

            n_dims (int): number of dimensions of the bottleneck.
        '''

        self.batchnorms = batchnorms

        self.kenel = kernel

        if n_dims == 1:
            self.__bn = torch.nn.BatchNorm1d
            self.__conv = torch.nn.Conv1d
        elif n_dims == 2:
            self.__bn = torch.nn.BatchNorm2d
            self.__conv = torch.nn.Conv2d
        elif n_dims == 3:
            self.__bn = torch.nn.BatchNorm3d
            self.__conv = torch.nn.Conv3d

        self.pipe = torch.nn.Sequential()

        if self.batchnorms:
            self.pipe.add(self.__bn(n_in))

        self.pipe.add(self.__conv(n_in, n_bottle, kernel=1)

        if self.batchnorms:
            self.pipe.add(self.__bn(n_bottle))

        self.pipe.add(kernel)

        if self.batchnorms:
            self.pipe.add(self.__bn(n_bottle))

        self.pipe.add(torch.nn.__conv(n_bottle, n_out, kernel=1)

        if self.batchnorms:
            self.pipe.add(torch.nn.__bn(n_out))

    def forward(self, input):
        return self.pipe(input)

    def __call__(self, input):
        return self.forward(input)


class ResidualBlock(torch.nn.Module):
    '''
    Residual block class. Given the function ```f```, performs the following
    transformation: ```f(x) + x```.
    '''
    
    def __init__(self, function):
        '''
        Constructor

        Args:
            function (torch.nn.Module): function for residual connection.
        '''
        self.function = function

    def __call__(self, input):
        return input + self.function(input)


class InceptionBlock(torch.nn.Module):
    '''
    Inception block. Performs several functions in parallel, results are
    concatenated (in dimension 1).
    '''
    def __init__(self, functions):
        '''
        Constructor of the inception block.
        Args:
            functions (list of torch.nn.Modules): list of transforms to perform.
        '''
        self.functions = functions

    def __call__(self, input):
        res = []
        for f in self.functions:
            res.append(f(input))

        return torch.cat(res, dim=1)

    def forward(self, input):
        self(input)
