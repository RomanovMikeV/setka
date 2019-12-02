import setka
import setka.base
import setka.pipes

import torch
import numpy

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset
import tempfile

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

    file = tempfile.SpooledTemporaryFile()
    file.write(b'something')

    return {'figures': {'img': fig},
            'texts': {'img': 'Sample'},
            'images': {'img': (img * 255.0).int().numpy().astype('uint8')},
            'audios': {'img': signal},
            'files': {'img': file, 'ext': 'sht'}}



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
                                 setka.pipes.Logger(f=view_result),
                                 setka.pipes.GarbageCollector()
                             ])

trainer.run_train(n_epochs=2)
trainer.run_epoch('test', 'test', n_iterations=2)
