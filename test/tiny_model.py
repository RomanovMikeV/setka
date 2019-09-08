import torch

class TensorNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 1, kernel_size=(1, 1), padding=1)
        self.layer2 = torch.nn.Conv2d(1, 1, kernel_size=(1, 1), padding=1)
        self.layer3 = torch.nn.Conv2d(1, 1, kernel_size=(1, 1), padding=1)
        self.layer4 = torch.nn.Conv2d(1, 1, kernel_size=(1, 1), padding=1)

        self.fc = torch.nn.Linear(1, 10)

    def __call__(self, input):
        x = input[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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