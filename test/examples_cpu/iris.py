import torch
import scorch.base
import scorch.callbacks
import sklearn.datasets
import numpy

class Iris(scorch.base.DataSet):
    def __init__(self, valid_split=0.1, test_split=0.1, mode='train'):
        super()
        data = sklearn.datasets.load_iris()

        X = data['data']
        y = data['target']

        n_valid = int(y.size * valid_split)
        n_test =  int(y.size * test_split)

        order = numpy.random.permutation(y.size)

        X = X[order, :]
        y = y[order]

        self.mode = mode

        self.data = {
            'valid': X[:n_valid, :],
            'test': X[n_valid:n_valid + n_test, :],
            'train': X[n_valid + n_test:, :]
        }

        self.targets = {
            'valid': y[:n_valid],
            'test': y[n_valid:n_valid + n_test],
            'train': y[n_valid + n_test:]
        }

    def __len__(self):
        return len(self.targets[self.mode])

    def __getitem__(self, idx):
        features = torch.Tensor(self.data[self.mode][idx])
        class_id = torch.Tensor(self.targets[self.mode][idx:idx+1])
        return [features], [class_id], self.mode + "_" + str(idx)


class IrisNet(scorch.base.Network):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 100)
        self.fc2 = torch.nn.Linear(100, 3)

    def forward(self, x):
        return [self.fc2(self.fc1(x[0]))]


ds = Iris()
model = IrisNet()


def loss(pred, targ):
    return torch.nn.functional.cross_entropy(pred[0], targ[0][:, 0].long())


def accuracy(pred, targ):
    predicted = pred[0].argmax(dim=1)
    # print(predicted.size())
    # print(targ[0].size())
    return (predicted == targ[0][:, 0].long()).sum(), predicted.numel()


trainer = scorch.base.Trainer(model,
                              optimizers=[scorch.base.OptimizerSwitch(model, torch.optim.Adam, lr=3.0e-3)],
                              criterion=loss,
                              callbacks=[
                                scorch.callbacks.ComputeMetrics(metrics=[loss, accuracy]),
                                scorch.callbacks.ReduceLROnPlateau(metric='loss'),
                                scorch.callbacks.ExponentialWeightAveraging(),
                                scorch.callbacks.WriteToTensorboard()
                              ])

for index in range(100):
    trainer.train_one_epoch(ds, subset='train')
    trainer.validate_one_epoch(ds, subset='train')
    trainer.validate_one_epoch(ds, subset='valid')

assert(trainer._metrics['valid']['accuracy'] > 0.9)
