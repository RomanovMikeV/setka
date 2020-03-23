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
    img = (img - img.min()) / (img.max() - img.min() + 1.0e-4)
    truth = one_input[1]
    label = one_output

    # print(img.size())

    fig = plt.figure()
    plt.imshow(img.permute(2, 1, 0))
    plt.close()
    return {'figures': {'img': fig}}

trainer = setka.base.Trainer(pipes=[
                                 setka.pipes.DataSetHandler(ds, batch_size=4, limits=2),
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
                                 setka.pipes.ComputeMetrics([loss, acc]),
                                 setka.pipes.TensorBoard(f=view_result)
                             ])

print(trainer.view_batch())

trainer.run_train(2)
trainer.run_epoch('test', 'test', n_iterations=2)
