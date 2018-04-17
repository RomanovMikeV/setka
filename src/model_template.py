class Network:
    def __init__(self):
        super(Network, self).__init__()
        pass
     
    def forward(self, input):
        pass
    
class Socket:
    def __init__(self, model):
        self.model = model
    
    def criterion(self, pred, target):
        pass
    
    def metrics(self, pred, target):
        pass