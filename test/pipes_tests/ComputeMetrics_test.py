import setka
import torch
import numpy

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset2 as test_dataset

from test_metrics import tensor_loss as loss
from test_metrics import tensor_acc as acc
from test_metrics import const

def test_ComputeMetrics():
    setka.base.environment_setup()

    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()

    trainer = setka.base.Trainer(pipes=[
                                     setka.pipes.DataSetHandler(
                                         ds, batch_size=32, limits=10, shuffle=False),
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
                                     setka.pipes.ComputeMetrics([loss, acc, const],
                                                                    divide_first=[True, False, True],
                                                                    steps_to_compute=2,
                                                                    reduce=True)
                                 ])

    trainer.run_train(2)

    assert(numpy.abs(trainer._metrics['train']['tensor_acc'] - 0.20) < 0.01)
    assert(numpy.abs(trainer._metrics['valid']['tensor_acc'] - 0.19) < 0.01)
    assert(numpy.abs(trainer._metrics['train']['const'] - 1.0) < 0.01)
    assert(numpy.abs(trainer._metrics['valid']['const'] - 1.0) < 0.01)
    assert(numpy.abs(trainer._metrics['train']['tensor_loss'] - 2.18) < 0.01)
    assert(numpy.abs(trainer._metrics['valid']['tensor_loss'] - 2.17) < 0.01)

def test_ComputeMetrics2():
    setka.base.environment_setup()

    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()

    trainer = setka.base.Trainer(pipes=[
                                     setka.pipes.DataSetHandler(
                                         ds, batch_size=32, limits=9, shuffle=False),
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
                                     setka.pipes.ComputeMetrics([loss, acc, const],
                                                                    divide_first=True,
                                                                    steps_to_compute=2,
                                                                    reduce=[True, False, True])
                                 ])

    trainer.run_train(2)

def test_ComputeMetrics3():
    setka.base.environment_setup()

    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()

    cyclic_lr = lambda x: torch.optim.lr_scheduler.CyclicLR(x, base_lr=0.0, max_lr=0.0)
    reduce_lr = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x)

    trainer = setka.base.Trainer(pipes=[
        setka.pipes.DataSetHandler(
            ds, batch_size=32, limits=9, shuffle=False),
        setka.pipes.ModelHandler(model),
        setka.pipes.LossHandler(loss),
        setka.pipes.ComputeMetrics([loss, acc, const],
                                   divide_first=True,
                                   steps_to_compute=2,
                                   reduce=[True, False, True]),
        setka.pipes.OneStepOptimizers(
            [
                setka.base.OptimizerSwitch(
                    model,
                    torch.optim.SGD,
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=5e-4,
                    schedulers={
                        'batch': [cyclic_lr],
                        'epoch': [(reduce_lr, 'valid', 'tensor_acc', 0)]})
            ]
        ),
    ])

    trainer.run_train(20)

