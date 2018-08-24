import torch
import torchvision
import sklearn.metrics

class Network(torch.nn.Module):
    def __init__(self):
        # Define your model here
        pass

    def forward(self, input):
        # Forward call goes here
        return [x]

    def __call__(self, input):
        return self.forward(input)




class Socket:
    def __init__(self, model):
        self.model = model

        # Select which modules to train
        # These modules will be trained and switched as trainable when needed
        # Others will be switched to the validation mode

        self.train_modules = torch.nn.ModuleList([
            ])

        # Choose your optimizer
        self.optimizer = torch.optim.Adam(
            self.train_modules.parameters(),
            lr=3.0e-4)

        # Choose your scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer)

    def criterion(self, pred, target):
        # Define your criterion
        # It can be your function, but remember that it should be
        # differntiable by pytorch
        loss_f = your_loss_function()
        return loss_f(pred[0], target[0])

    def metrics(self, pred, target):
        # Define here your set of metrics that will be computed after each
        # epoch. The main metric by which we make a decision to reduce the
        # learning rate should be called "main"
        metric_1 = your_metric()
        metric_2 = your_metric()
        loss = self.criterion(pred, target)
        return {'main': metric_1, 'metric_1': metric_1,
                'metric_2': metric_2, 'loss': loss}
