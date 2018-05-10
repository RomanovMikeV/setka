import torch
import torchvision
import skimage.transform
import sys
import argparse

import torchvision.transforms as transform

sys.path.append('../src')

import trainer
import dataset
import model
import utils

## Training parameters

batch_size=32 * 6
num_workers=32
pretraining=True
learning_rate = 3.0e-4
dump_period = 100
epochs = 1000
checkpoint = None
use_cuda = True

## Making datasets

meanstd = {}
normalize = transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_transform = transform.Compose(
    [transform.ToPILImage(),
     transform.Resize(256),
     transform.RandomCrop(224),
     transform.RandomHorizontalFlip(),
     transform.ToTensor(),
     normalize
     ])

valid_transform = transform.Compose(
    [transform.ToPILImage(),
     transform.Resize(224),
     transform.CenterCrop(224),
     transform.ToTensor(),
     normalize])

ds_index = dataset.DataSetIndex('')
train_dataset = dataset.DataSet(
    ds_index, train_transform, mode='train')

valid_dataset = dataset.DataSet(
    ds_index, valid_transform, mode='valid')

## Creating dataloaders from the datasets

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           drop_last=True,
                                           collate_fn=utils.default_collate)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           drop_last=False,
                                           collate_fn=utils.default_collate)

## Creating neural network and a socket for it

net = model.Network().float()
net = torch.nn.parallel.DataParallel(net).eval()

if use_cuda:
    net = net.cuda()
    
socket = model.Socket(net)


if pretraining:
    
    ## Specify here modules for pretraining
    pretrain_modules = socket.model.module.pretrain_modules
    
    pretrain_optimizer = torch.optim.Adam(
        pretrain_modules.parameters(), 
        lr=learning_rate)
    
    my_pretrainer = trainer.Trainer(socket, 
                                    pretrain_optimizer,
                                    verbose=1000,
                                    train_modules=pretrain_modules,
                                    use_cuda=use_cuda)
    for index in range(epochs):
        loss = my_pretrainer.train(train_loader)
        train_metrics = my_pretrainer.validate(train_loader)
        valid_metrics = my_pretrainer.validate(valid_loader)
        metrics =  '\t '.join([str(x) for x in train_metrics + valid_metrics])
        print(str(index) + '\t' + metrics)
        
        if index % dump_period == 0:
            my_pretrainer.make_checkpoint('pretraining_', info=metrics)

else:
    train_modules = socket.model.module.train_modules
    
    if checkpoint:
        train_optimizer = torch.optim.Adam(train_modules.parameters(), lr=learning_rate)
        
        my_trainer = trainer.Trainer(socket, 
                                     train_optimizer, 
                                     verbose=10, 
                                     train_modules=train_modules,
                                     use_cuda=use_cuda)
        
        my_trainer.load_checkpoint(checkpoint)

    socket = my_trainer.socket
    train_modules = socket.model.parameters()
    train_optimizer = torch.optim.Adam(train_modules, lr=learning_rate)
    my_trainer.optimizer = train_optimizer

    for index in range(epochs):
        loss = my_trainer.train(train_loader)
        train_metrics = my_trainer.validate(train_loader)
        valid_metrics = my_trainer.validate(valid_loader)
        metrics =  '\t '.join([str(x) for x in train_metrics + valid_metrics])
        print(str(index) + '\t' + metrics)

        if index % dump_period == 0:
            my_trainer.make_checkpoint(info=metrics)