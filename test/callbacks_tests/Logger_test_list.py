import setka
import setka.base
import setka.callbacks

import torch
import numpy

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

import matplotlib.pyplot as plt

from test_metrics import list_loss as loss
from test_metrics import list_acc as acc

ds = test_dataset.CIFAR10()
model = tiny_model.ListNet()

def view_result(one_input, one_output):
    # print("In view result")
    img = one_input[0]
    img = (img - img.min()) / (img.max() - img.min())
    truth = one_input[1]
    label = one_output

    # print(img.size())

    fig = plt.figure()
    plt.imshow(img.permute(2, 1, 0))
    plt.close()

    signal = numpy.sin(numpy.linspace(0, 1000, 40000))

    return {'figures': {'img': fig},
            'texts': {'img': 'Sample'},
            'images': {'img': img.numpy()},
            'audios': {'img': signal}}



trainer = setka.base.Trainer(callbacks=[
                                 setka.callbacks.DataSetHandler(ds,
                                                                batch_size=4,
                                                                limits={'train': 3, 'valid':3, 'test': 1}),
                                 setka.callbacks.ModelHandler(model),
                                 setka.callbacks.LossHandler(loss),
                                 setka.callbacks.OneStepOptimizers(
                                    [
                                        setka.base.OptimizerSwitch(
                                            model,
                                            torch.optim.SGD,
                                            lr=0.01,
                                            momentum=0.9,
                                            weight_decay=5e-4)
                                    ]
                                 ),
                                 setka.callbacks.ComputeMetrics([loss, acc]),
                                 setka.callbacks.Logger(f=view_result)
                             ])

for index in range(2):
    trainer.one_epoch('train', 'train')
    trainer.one_epoch('valid', 'train')
    trainer.one_epoch('test', 'train')
