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
        
        self.pretrain_modules = torch.nn.ModuleList([self.classifier])
        
        self.train_modules = torch.nn.ModuleList([self.conv1,
                                                   self.conv2,
                                                   self.classifier])
        self.pretrain_modules = self.train_modules
    
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
    
    def criterion(self, pred, target):
        loss_f = torch.nn.CrossEntropyLoss()
        return loss_f(pred[0], target[0])
    
    def metrics(self, pred, target):
        accuracy = ((pred[0].numpy().argmax(axis=1) == target[0].numpy()).sum() / pred[0].size(0))
        errors = 1.0 - accuracy
        return [accuracy, errors]