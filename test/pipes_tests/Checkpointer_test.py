import setka
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

from test_metrics import tensor_loss as loss
from test_metrics import tensor_acc as acc

def test_MakeCheckpoints():

    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()

    trainer = setka.base.Trainer(pipes=[
                                     setka.pipes.DataSetHandler(ds, batch_size=32, limits=2),
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
                                     setka.pipes.Checkpointer('tensor_acc', max_mode=True, name='my_experiment', keep_best_only=False),
                                     setka.pipes.Checkpointer('tensor_loss', max_mode=False, name='my_experiment_best_only')
                                 ])

    trainer.run_train(5)
    trainer.run_epoch('test', 'test', n_iterations=2)

    path = os.path.join('runs', 'my_experiment')
    last_exp = sorted(os.listdir(path))[-1]

    assert(len(os.listdir(os.path.join('runs', 'my_experiment', last_exp, 'checkpoints'))) == 8 * 2)
    assert(os.path.exists(os.path.join('runs', 'my_experiment', last_exp, 'checkpoints', 'my_experiment_best.pth.tar')))
    assert (
        os.path.exists(os.path.join('runs', 'my_experiment', last_exp, 'checkpoints', 'my_experiment_latest.pth.tar')))
    assert (
        os.path.exists(os.path.join('runs', 'my_experiment',
                                    last_exp, 'checkpoints', 'my_experiment_weights_best.pth.tar')))
    assert (
        os.path.exists(os.path.join('runs', 'my_experiment',
                                    last_exp, 'checkpoints', 'my_experiment_weights_latest.pth.tar')))

    for index in range(6):
        assert (
            os.path.exists(os.path.join('runs', 'my_experiment',
                                        last_exp, 'checkpoints', 'my_experiment_weights_' + str(index) + '.pth.tar')))
        assert (os.path.exists(
            os.path.join('runs', 'my_experiment', last_exp, 'checkpoints', 'my_experiment_' + str(index) + '.pth.tar')))


    # path = os.path.join('runs', 'my_experiment_best_only')
    # last_exp = sorted(os.listdir(path))[-1]
    assert(len(os.listdir(os.path.join('runs', 'my_experiment_best_only', last_exp, 'checkpoints'))) == 4)
    assert(os.path.exists(os.path.join('runs', 'my_experiment_best_only', last_exp, 'checkpoints', 'my_experiment_best_only_best.pth.tar')))
    assert (
        os.path.exists(os.path.join('runs', 'my_experiment_best_only', last_exp, 'checkpoints', 'my_experiment_best_only_latest.pth.tar')))
    assert (
        os.path.exists(os.path.join('runs', 'my_experiment_best_only',
                                    last_exp, 'checkpoints', 'my_experiment_best_only_weights_best.pth.tar')))
    assert (
        os.path.exists(os.path.join('runs', 'my_experiment_best_only',
                                    last_exp, 'checkpoints', 'my_experiment_best_only_weights_latest.pth.tar')))

    latest_weights = torch.load(os.path.join('runs', 'my_experiment_best_only',
                                    last_exp, 'checkpoints', 'my_experiment_best_only_weights_latest.pth.tar'))

    latest_weights = latest_weights

    latest_trainer = torch.load(os.path.join('runs', 'my_experiment_best_only',
                                             last_exp, 'checkpoints', 'my_experiment_best_only_latest.pth.tar'))
    latest_trainer = latest_trainer['trainer']

    assert(latest_trainer._model.state_dict().__str__() == trainer._model.state_dict().__str__())
    assert(latest_weights.__str__() == trainer._model.state_dict().__str__())
