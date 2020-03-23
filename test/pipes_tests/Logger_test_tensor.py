import setka
import setka.base
import setka.pipes

import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

import matplotlib.pyplot as plt

from test_metrics import tensor_loss as loss
from test_metrics import tensor_acc as acc

ds = test_dataset.CIFAR10()
model = tiny_model.TensorNet()

def view_result(one_input, one_output):
    # print("In view result")
    img = one_input[0]
    img = (img - img.min()) / (img.max() - img.min())
    truth = one_input[1]
    label = one_output[0]

    # print(img.size())

    fig = plt.figure()
    plt.imshow(img.permute(2, 1, 0))
    plt.close()
    return {'figures': {'img': fig}}



trainer = setka.base.Trainer(pipes=[
                                 setka.pipes.DataSetHandler(ds,
                                                                batch_size=4,
                                                                limits={'train': 3, 'valid':3, 'test': 1}),
                                 setka.pipes.ModelHandler(model),
                                 setka.pipes.LossHandler(loss),
                                 setka.pipes.OneStepOptimizers(
                                    [
                                        setka.base.OptimizerSwitch(
                                            model,
                                            torch.optim.SGD,
                                            lr=0.01,
                                            momentum=0.9,
                                            weight_decay=5e-4)
                                    ]
                                 ),
                                 setka.pipes.ComputeMetrics([loss, acc]),
                                 setka.pipes.Logger(f=view_result)
                             ])

trainer.run_train(2)
trainer.run_epoch('test', 'test', n_iterations=2)
