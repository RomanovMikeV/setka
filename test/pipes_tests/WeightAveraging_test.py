import setka
import setka.base
import setka.pipes

import torch

import torchvision.datasets
import torchvision.transforms

from torch import nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

import matplotlib.pyplot as plt

from test_metrics import tensor_loss as loss
from test_metrics import tensor_acc as acc

def test_WeightAveraging():
    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()

    trainer = setka.base.Trainer(
        pipes=[
             setka.pipes.DataSetHandler(ds, batch_size=32, limits=5),
             setka.pipes.ModelHandler(model),
             setka.pipes.LossHandler(loss),
             setka.pipes.OneStepOptimizers([
                setka.base.OptimizerSwitch(
                model,
                torch.optim.SGD,
                lr=0.1,
                momentum=0.9,
                weight_decay=5e-4)]),
             setka.pipes.WeightAveraging(
                 epoch_start=0,
                 interval=1)
        ])

    trainer.run_train(3)