import torch
import torchvision
import sklearn.metrics
import matplotlib.pyplot as plt

class OptimizerSwitch():
    def __init__(self, train_module, optimizer, is_active=True, **kwargs):
        self.optimizer = optimizer(train_module.parameters(), **kwargs)
        self.module = train_module
        self.active = is_active

class Network(torch.nn.Module):
    '''
    The Network itself
    '''
    def __init__(self):
        super(Network, self).__init__()

        # Define your network components here

        self.conv1 = torch.nn.Conv2d(1, 64, (5, 5), padding=2)
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(64, 256, (5, 5), padding=2)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.classifier = torch.nn.Linear(256, 10)

    def forward(self, input):

        # Define your forward propagation function here

        res1 = self.pool1(torch.nn.functional.tanh(self.conv1(input[0])))
        res2 = self.pool2(torch.nn.functional.tanh(self.conv2(res1)))

        x = res2.mean(dim=3).mean(dim=2)

        x = self.classifier(x)

        # Return the list of results

        return [x]

    def __call__(self, input):
        return self.forward(input)

class Socket:
    '''
    The Socket class knows how to treat the  Network class well.
    '''
    def __init__(self, model):

        # The model is plugged into the socket
        self.model = model

        # Define the modules of the network that you are going to train
        # These modules the training loop will switch between training and
        # evaluation modes.

        # Define the optimizers for your network.

        self.optimizers = [
            OptimizerSwitch(
                torch.nn.ModuleList([self.model.conv1, self.model.conv2]),
                torch.optim.Adam, lr=3.0e-4),
            OptimizerSwitch(self.model.classifier, torch.optim.Adam, lr=3.0e-4)
        ]

        self.metric_vals = {}
        self.epoch = 0
        self.iteration = 0

        # Define your scheduler for the network
        self.scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizers[0].optimizer)
        self.scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizers[1].optimizer)

    def criterion(self, output, target):

        # Define here the criterion for optimization

        loss_f = torch.nn.CrossEntropyLoss()

        return loss_f(output[0], target[0])

    # Optional
    def scheduling(self):
        self.scheduler1.step(self.metric_vals['main'])
        self.scheduler2.step(self.metric_vals['main'])

    # Optional
    def metrics(self, output, target):

        # Define here the metrics for your method.
        # The 'main' metrics will be used by scheduler and, also
        # it will be used for best checkpoint selection process.
        # Choose wisely.

        accuracy = (
            float((output[0].argmax(dim=1) == target[0]).sum()) /
            float(output[0].size(0)))
        errors = 1.0 - accuracy
        loss = self.criterion(output, target)

        self.metric_vals = {'main': accuracy,
                'accuracy': accuracy,
                'errors': errors,
                'loss': loss}

        return self.metric_vals

    # Optional
    def process_result(self, input, output, id):

        # This method will be used by test script.
        # The outputs of this function will be collected and saved as
        # the results of your network.

        return output

    # Optional
    def visualize(self, input, output, id):

        # This function is used for visualization with TensorboardX.
        # You can make here:
        # 'images' -- numpy images
        # 'figures' -- matplotlib figures
        # 'texts' -- texts
        # 'audios' -- numpy audios
        # 'graphs' -- onnx graphs
        # 'embeddings' -- embeddings

        # Create the result container
        res = {'figures': {}}

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(input[0][0, :, :])

        text = ''
        for number in range(len(output)):
            text += str(number) + ': ' + str(output[0][number].item()) + ' | '

        plt.title(text)

            # Add item to the container. Note that ID will be used in the
            # tensorboard as the image name

        res['figures'][id] = fig

        plt.close(fig)

        return res
