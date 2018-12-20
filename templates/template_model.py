import torch
import torchvision
import sklearn.metrics
import matplotlib.pyplot as plt

from scorch.base import OptimizerSwitch
import scorch.base

class Network(scorch.base.Network):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, 5, padding=2)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.fc1 = torch.nn.Linear(6272, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, input):
        res = input[0]
        res = self.pool1(torch.relu(self.conv1(res)))
        res = self.pool2(torch.relu(self.conv2(res)))

        res = res.view([res.size(0), -1])

        res = torch.relu(self.fc1(res))
        res = torch.relu(self.fc2(res))
        res = self.fc3(res)

        return [res]
