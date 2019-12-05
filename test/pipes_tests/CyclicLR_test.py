import setka
import setka.base
import setka.pipes

import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

import numpy

from test_metrics import tensor_loss as loss
from test_metrics import tensor_acc as acc

ds = test_dataset.CIFAR10()
model = tiny_model.TensorNet()


def cycle(i):
    return numpy.sin(numpy.pi * i)

trainer = setka.base.Trainer(pipes=[
                                 setka.pipes.DataSetHandler(ds, batch_size=32, limits=2),
                                 setka.pipes.ModelHandler(model),
                                 setka.pipes.LossHandler(loss),
                                 setka.pipes.OneStepOptimizers(
                                    [
                                        setka.base.OptimizerSwitch(
                                            model,
                                            torch.optim.SGD,
                                            lr=0.1,
                                            momentum=0.9,
                                            weight_decay=5e-4)
                                    ]
                                 ),
                                 setka.pipes.CyclicLR(cycle),
                                 setka.pipes.GarbageCollector()
                             ])

trainer.run_train(2)
