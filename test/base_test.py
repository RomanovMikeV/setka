import torch
import setka
import tiny_model
import test_dataset3 as test_dataset
from test_metrics import tensor_loss as loss

def test_base():
    setka.base.environment_setup()

    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()

    # input, target, text, random_ones = ds['train', 0]

    trainer = setka.base.Trainer(pipes=[
                                     setka.pipes.DatasetHandler(ds, batch_size=32, limits=2, shuffle=True),
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
                                     setka.pipes.Pipe()
                                 ],
                                 collection_op=setka.base.CollectionOperator(soft_collate_fn=True))

    trainer.remove_pipe(setka.pipes.LossHandler)
    trainer.add_pipe(setka.pipes.LossHandler(loss))

    trainer.run_train(n_epochs=2)
    trainer.run_epoch(mode='train', subset='train', n_iterations=100)
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


