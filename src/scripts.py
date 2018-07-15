import torch
import torchvision
import skimage.transform
import sys
import argparse
import time
import importlib.util

import torchvision.transforms as transform

import trainer
import utils

def train(model_source_path,
          dataset_source_path,
          dataset_path,
          batch_size=32,
          num_workers=4,
          pretraining=False,
          learning_rate=3.0e-4,
          dump_period=1,
          epochs=1000,
          checkpoint=None,
          use_cuda=False,
          validate_on_train=False,
          max_train_iterations=-1,
          max_valid_iterations=-1,
          verbosity=-1,
          checkpoint_prefix=''):
    
    spec = importlib.util.spec_from_file_location("model", model_source_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    spec = importlib.util.spec_from_file_location("dataset", dataset_source_path)
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    ## Creating neural network and a socket for it

    net = model.Network()
    net = torch.nn.parallel.DataParallel(net).eval()

    if use_cuda:
        net = net.cuda()
    
    socket = model.Socket(net)

   
    # Creating the datasets
 
    ds_index = dataset.DataSetIndex(dataset_path)
    train_dataset = dataset.DataSet(
        ds_index, mode='train')

    valid_dataset = dataset.DataSet(
        ds_index, mode='valid')

    ## Creating dataloaders from the datasets

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=False,
                                               collate_fn=utils.default_collate)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               pin_memory=False,
                                               collate_fn=utils.default_collate)
   


    my_trainer = trainer.Trainer(socket,
                                 verbosity=verbosity,
                                 use_cuda=use_cuda,
                                 max_train_iterations=max_train_iterations,
                                 max_valid_iterations=max_valid_iterations)
    best_metrics = None

    if checkpoint is not None:
        my_trainer = trainer.load_from_checkpoint(checkpoint,
                                                  socket,
                                                  verbosity=verbosity,
                                                  use_cuda=use_cuda,
                                                  max_train_iterations=max_train_iterations,
                                                  max_valid_iterations=max_valid_iterations)
        best_metrics = my_trainer.metrics
        
        
    for index in range(epochs):
        loss = my_trainer.train(train_loader)
        metrics = []
        if validate_on_train:
            for metric in my_trainer.validate(train_loader):
                metrics.append(metric)
                    
        for metric in my_trainer.validate(valid_loader):
            metrics.append(metric)
            
        if index % dump_period == 0:
            is_best = False
            if best_metrics == None:
                is_best = True
            else:
                if best_metrics[0] < my_trainer.metrics[0]:
                    is_best = True
            if is_best:
                best_metrics = my_trainer.metrics

            my_trainer.make_checkpoint(checkpoint_prefix, info=metrics, is_best=is_best)
