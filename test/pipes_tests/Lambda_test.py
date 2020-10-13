import setka
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

from test_metrics import tensor_loss as loss

def test_Lambda():
    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()

    def print_message():
        print("Message")

    trainer = setka.base.Trainer(pipes=[
                                     setka.pipes.DatasetHandler(ds, batch_size=32, limits=2),
                                     setka.pipes.ModelHandler(model),
                                     setka.pipes.LossHandler(loss),
                                     setka.pipes.OneStepOptimizers(
                                        [
                                            setka.base.Optimizer(
                                                model,
                                                torch.optim.SGD,
                                                lr=0.1,
                                                momentum=0.9,
                                                weight_decay=5e-4)
                                        ]
                                     ),
                                     setka.pipes.Lambda(on_batch_begin=print_message)
                                 ])

    trainer.run_train(2)