import torch
import torchvision
import skimage.transform
import sys
import argparse

sys.path.append('../src')

import trainer
import dataset
import model
import utils

## Training parameters

batch_size=4
num_workers=2
pretraining=True
learning_rate = 3.0e-4
dump_period = 100
epochs = 1000
checkpoint = None

## Making datasets

train_dataset = dataset.DataSet('../data/', 
                                mode='train')

valid_dataset = dataset.DataSet('../data/', 
                                mode='valid')

## Creating dataloaders from the datasets

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           drop_last=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           drop_last=False)

## Creating neural network and a socket for it

net = model.Network().float().cuda()
socket = model.Socket(net)

if pretraining:
    
    ## Specify here modules for pretraining
    pretrain_modules = torch.nn.ModuleList([]).parameters()
    
    pretrain_optimizer = torch.optim.Adam(pretrain_modules, lr=learning_rate)
    my_pretrainer = trainer.Trainer(socket, pretrain_optimizer)
    for index in range(epochs):
        loss = my_pretrainer.train(train_loader)
        train_metrics = my_pretrainer.validate(train_loader)
        valid_metrics = my_pretrainer.validate(valid_loader)
        metrics =  '\t '.join([str(x) for x in train_metrics + valid_metrics])
        print(epoch + '\t' + metrics)
        
        if index % dump_period == 0:
            my_pretrainer.make_checkpoint('pretraining_', info=metrics)

else:
    train_modules = socket.model.parameters()
    
    if checkpoint:
        train_optimizer = torch.optim.Adam(train_modules, lr=learning_rate)
        my_trainer = trainer.Trainer(socket, train_optimizer)
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
        print(epoch + '\t' + metrics)

        if index % dump_period == 0:
            my_trainer.make_checkpoint(info=metrics)