import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import setka


class SimpleModel(nn.Module):
    def __init__(self, channels, input_channels=3, n_classes=10):
        super().__init__()

        modules = []

        in_c = input_channels
        for out_c in channels:
            modules.append(nn.Conv2d(in_c, out_c, 3, padding=1))
            modules.append(nn.BatchNorm2d(out_c))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.MaxPool2d(2))
            in_c = out_c

        self.encoder = nn.Sequential(*modules)
        self.decoder = nn.Linear(in_c, n_classes)

    def __call__(self, input):
        features = self.encoder(input['image']).mean(dim=-1).mean(dim=-1).unsqueeze(1)
        return self.decoder(features)[:, 0]

    
def loss(pred, input):
    return F.cross_entropy(pred, input['label'])


def acc(pred, input):
    return (input['label'] == pred.argmax(dim=1)).float().sum() / float(pred.size(0))


CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat' , 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def draw_results(one_input, one_output, class_names=CLASS_NAMES, k=3):
    probs = F.softmax(one_output)
    top_classes = torch.topk(probs, k)[1]
    img = (one_input['image'] - one_input['image'].min()) / (one_input['image'].max() - one_input['image'].min())
    
    topk_names = np.array(class_names)[top_classes.detach().cpu().numpy()]
    f = plt.figure()
    plt.imshow(img.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    
    textstr = 'Class: ' + class_names[one_input['label']] + '\n' + 'Preds:\n'
    for i in range(k):
        textstr += f'{i+1}) ' + topk_names[i] + ' ({:.1f})\n'.format(probs[top_classes[i]] * 100.0)
    
    plt.gca().text(1.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    plt.tight_layout()
    plt.close()
    return {'figures': {'image':f}}


setka.base.environment_setup()
LOG_DIR = 'experiments/'
EXPERIMENT_NAME = 'cifar10_2'
SCHEDULE = [
   {'mode': 'train', 'subset': 'train'},
   {'mode': 'valid', 'subset': 'train', 'n_iterations': 100},
   {'mode': 'valid', 'subset': 'valid'},
   {'mode': 'valid', 'subset': 'test'},
   {'mode': 'test', 'subset': 'test', 'n_iterations': 1}
]

ds = setka.datasets.CIFAR10()
model = SimpleModel(channels=[8, 16, 32, 64])
opt = setka.base.OptimizerSwitch(model, torch.optim.Adam, lr=1.0e-3, schedulers=[])


trainer = setka.base.Trainer(
    pipes=[
        setka.pipes.DatasetHandler(ds, batch_size=256, workers=4, shuffle={'train': True, 'valid': True, 'test': False},
                                   epoch_schedule=SCHEDULE),
        setka.pipes.ModelHandler(model, data_parallel=False),
        setka.pipes.UseCuda(),
        setka.pipes.LossHandler(loss),
        setka.pipes.ComputeMetrics([loss, acc]),
        setka.pipes.OneStepOptimizers([opt]),
        setka.pipes.MakeCheckpoints('acc', max_mode=True, log_dir=LOG_DIR, name=EXPERIMENT_NAME),
        setka.pipes.Logger(f=draw_results, log_dir=LOG_DIR, name=EXPERIMENT_NAME, 
                           ignore_list=['*__pycache__*', '*.ipynb_checkpoints*', 'experiments/*']),
        setka.pipes.MultilineProgressBar(),
        setka.pipes.TensorBoard(f=draw_results, log_dir=os.path.join(LOG_DIR, 'runs'), name=EXPERIMENT_NAME)
    ]
)

trainer.run_train(10)