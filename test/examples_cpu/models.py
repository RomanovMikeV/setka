import torch

class ThreeLayersFullyConnected(torch.nn.Module):
    def __init__(self,
                 input_side=28,
                 channels=1,
                 n = [100, 100, 10],
                 batch_norms=False):

        super().__init__()
        self.pipe = torch.nn.Sequential()

        n_in = input_side ** 2 * channels

        for index in range(len(n)):
            n_out = n[index]
            if batch_norms:
                self.pipe.add_module('bn' + str(index), torch.nn.BatchNorm1d(n_in))
            self.pipe.add_module('fc' + str(index), torch.nn.Linear(n_in, n_out))
            if not index == len(n) - 1:
                self.pipe.add_module('act' + str(index), torch.nn.ReLU())
            n_in = n_out


    def __call__(self, inputs):
        x = inputs[0].view([inputs[0].size(0), -1])
        x = self.pipe(x)
        return [x]

class LeNetFullyConvolutional(torch.nn.Module):
    def __init__(self,
                 channels=[1,20, 20],
                 n_classes = 10,
                 kernel=7,
                 batch_norms=False):

        super().__init__()
        self.pipe = torch.nn.Sequential()
        for i in range(len(channels) - 1):
            n_in = channels[i]
            n_out = channels[i + 1]

            if batch_norms:
                self.pipe.add_module('bn' + str(i), torch.nn.BatchNorm2d(n_in))

            self.pipe.add_module('conv' + str(i), torch.nn.Conv2d(n_in, n_out, kernel, padding=int((kernel - 1) / 2)))
            self.pipe.add_module('act' + str(i), torch.nn.ReLU())
            self.pipe.add_module('pool' + str(i), torch.nn.MaxPool2d(2))

        self.classifier = torch.nn.Linear(channels[-1], n_classes)

    def __call__(self, inputs):
        x = self.pipe(inputs[0])

        while len(x.size()) != 2:
            x = x.mean(dim=-1)

        x = self.classifier(x)
        return [x]
