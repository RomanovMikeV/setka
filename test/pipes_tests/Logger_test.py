import setka
import torch
import os
import sys
import numpy
import tempfile
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

import matplotlib.pyplot as plt

from test_metrics import dict_loss, tensor_loss, list_loss
from test_metrics import dict_acc, tensor_acc, list_acc

def test_Logger_dict():
    loss = dict_loss
    acc = dict_acc

    ds = test_dataset.CIFAR10()
    model = tiny_model.DictNet()


    def view_result(one_input, one_output):
        img = one_input[0]
        img = (img - img.min()) / (img.max() - img.min())
        truth = one_input[1]
        label = one_output['res']

        fig = plt.figure()
        plt.imshow(img.permute(2, 1, 0))
        plt.close()
        return {'figures': {'img': fig}}


    trainer = setka.base.Trainer(pipes=[
                                     setka.pipes.DataSetHandler(ds, batch_size=4,
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
                                     setka.pipes.Logger(f=view_result, name='my_experiment')
                                 ])

    trainer.run_train(n_epochs=2)
    trainer.run_epoch('test', 'test', n_iterations=2)

    assert(os.path.exists(os.path.join('runs', 'my_experiment')))
    assert(len(os.listdir(os.path.join('runs', 'my_experiment'))) > 0)
    last_run = sorted(os.listdir(os.path.join('runs', 'my_experiment')))[-1]
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, '2_figures', 'test_2047', 'img.png')))

    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'bash_command.txt')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'batch_log.json')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'epoch_log.json')))


def test_Logger_list():
    loss = list_loss
    acc = list_acc

    ds = test_dataset.CIFAR10()
    model = tiny_model.ListNet()

    def view_result(one_input, one_output):
        img = one_input[0]
        img = (img - img.min()) / (img.max() - img.min())
        truth = one_input[1]
        label = one_output

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
                'files': {'img.sht': file, 'img': file}}



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
                                     setka.pipes.Logger(f=view_result, name='my_experiment')
                                 ])

    trainer.run_train(n_epochs=2)
    trainer.run_epoch('test', 'test', n_iterations=2)

    assert(os.path.exists(os.path.join('runs', 'my_experiment')))
    assert(len(os.listdir(os.path.join('runs', 'my_experiment'))) > 0)
    last_run = sorted(os.listdir(os.path.join('runs', 'my_experiment')))[-1]
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, '2_figures', 'test_2047', 'img.png')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, '2_audios', 'test_2047', 'img.wav')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, '2_texts', 'test_2047', 'img.txt')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, '2_images', 'test_2047', 'img.png')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, '2_files', 'test_2047', 'img.sht')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, '2_files', 'test_2047', 'img.bin')))

    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'bash_command.txt')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'batch_log.json')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'epoch_log.json')))


def test_Logger_tensor():
    loss = tensor_loss
    acc = tensor_acc

    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()

    def view_result(one_input, one_output):
        img = one_input[0]
        img = (img - img.min()) / (img.max() - img.min())
        truth = one_input[1]
        label = one_output[0]

        fig = plt.figure()
        plt.imshow(img.permute(2, 1, 0))
        plt.close()
        return {'figures': {'img.jpg': fig}}

    trainer = setka.base.Trainer(pipes=[
        setka.pipes.DataSetHandler(ds,
                                   batch_size=4,
                                   limits={'train': 3, 'valid': 3, 'test': 1}),
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
        setka.pipes.Logger(f=view_result, full_snapshot_path=True, name='my_experiment')
    ])

    trainer.run_train(2)
    trainer.run_epoch('test', 'test', n_iterations=2)

    assert(os.path.exists(os.path.join('runs', 'my_experiment')))
    assert(len(os.listdir(os.path.join('runs', 'my_experiment'))) > 0)
    last_run = sorted(os.listdir(os.path.join('runs', 'my_experiment')))[-1]
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, '2_figures', 'test_2047', 'img.jpg')))

    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'bash_command.txt')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'epoch_log.json')))
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_run, 'batch_log.json')))

