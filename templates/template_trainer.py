import torch
import torchvision
import sklearn.metrics
import matplotlib.pyplot as plt

from scorch.base import OptimizerSwitch
import scorch.base

class Trainer(scorch.base.Trainer):
    def __init__(self, model, **kwargs):
        super(Trainer, self).__init__(model, **kwargs)

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
