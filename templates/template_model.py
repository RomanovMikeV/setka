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


class Socket(scorch.base.Socket):
    def __init__(self, model):
        super(Socket, self).__init__(model)

        self.criterion_f = torch.nn.CrossEntropyLoss()
        self.optimizers = [
            scorch.base.OptimizerSwitch(self.model, torch.optim.Adam, lr=3.0e-4)]


    def criterion(self, preds, target):
        return self.criterion_f(preds[0], target[0])


    def metrics(self, preds, target):
        acc = (preds[0].argmax(dim=1) == target[0]).float().mean()
        return {'main': acc}


    def visualize(self, input, output, id):
        res = {'figures': {}}

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(input[0][0, :, :])

        text = ''
        for number in range(len(output)):
            text += str(number) + ': ' + str(output[0][number].item()) + ' | '
        plt.title(text)

        res['figures'][id] = fig

        plt.close(fig)

        return res
