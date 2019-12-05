import torch

class TensorNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = torch.nn.Conv2d(3, 1, kernel_size=(1, 1), padding=1)
        # self.layer2 = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
        # self.layer3 = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
        # self.layer4 = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)

        # self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(3, 10)

    def __call__(self, input):
        x = input[0]
        # x = self.relu(self.layer1(x))
        # x = self.relu(self.layer2(x))
        # x = self.relu(self.layer3(x))
        # x = self.relu(self.layer4(x))

        x = x.mean(dim=-1).mean(dim=-1)

        x = self.fc(x)

        return x


class ListNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = TensorNet()

    def __call__(self, input):
        x = self.net(input)
        return x,


class DictNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = TensorNet()

    def __call__(self, input):
        x = self.net(input)
        return {'res': x}