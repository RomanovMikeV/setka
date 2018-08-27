import torch
import torchvision
import sklearn.metrics

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, (5, 5), padding=2)
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(64, 256, (5, 5), padding=2)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.classifier = torch.nn.Linear(256, 10)

    def forward(self, input):
        res1 = self.pool1(torch.nn.functional.tanh(self.conv1(input[0])))
        res2 = self.pool2(torch.nn.functional.tanh(self.conv2(res1)))


        x = res2.mean(dim=3).mean(dim=2)

        x = self.classifier(x)

        return [x]

    def __call__(self, input):
        return self.forward(input)




class Socket:
    def __init__(self, model):
        self.model = model

        self.train_modules = torch.nn.ModuleList([
            self.model.conv1,
            self.model.conv2,
            self.model.classifier])

        self.optimizer = torch.optim.Adam(
            self.train_modules.parameters(),
            lr=3.0e-4)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer)

    def criterion(self, pred, target):
        loss_f = torch.nn.CrossEntropyLoss()
        return loss_f(pred[0], target[0])

    def metrics(self, pred, target):
        accuracy = (
            (pred[0].numpy().argmax(axis=1) == target[0].numpy()).sum() /
            pred[0].size(0))
        errors = 1.0 - accuracy
        loss = self.criterion(pred, target)
        return {'main': accuracy,
                'accuracy': accuracy,
                'errors': errors,
                'loss': loss}


    def process(self, inputs, outputs):
        res = {'images': {}, 'texts': {}}

        for index in range(len(inputs[0])):
            res['images'][index] = inputs[0][index]

        for index in range(len(outputs[0])):
            text = ''
            for number in range(len(outputs[0][index])):
                text += str(number) + ': ' + str(outputs[0][index][number]) + '\n'
            res['texts'][index] = (text)

        return res
