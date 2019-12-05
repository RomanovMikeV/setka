import setka.base
import setka.pipes

import torch

import tiny_model
import test_dataset
from test_metrics import tensor_loss as loss

setka.base.environment_setup()

ds = test_dataset.CIFAR10()
model = tiny_model.TensorNet()

input, target = ds['train', 0]

trainer = setka.base.Trainer(pipes=[
                                 setka.pipes.DataSetHandler(ds, batch_size=32, limits=2, shuffle=True),
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
                                 setka.pipes.Pipe(),
                                 setka.pipes.GarbageCollector()
                             ])

trainer.run_train(n_epochs=2)
trainer.run_epoch(mode='train', subset='train', n_iterations=10)
trainer.run_epoch(mode='valid', subset='valid', n_iterations=10)
trainer.run_epoch(mode='test', subset='valid', n_iterations=10)

print("=====  Training schedule =====")
print(trainer.view_train())
print("===== Epoch schedule =====")
print(trainer.view_epoch())
print("===== Batch schedule =====")
print(trainer.view_batch())

print("===== Trianer =====")
print(trainer.view_pipeline())