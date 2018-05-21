import torch
import torchvision
import skimage.transform
import sys
import argparse
import time

import torchvision.transforms as transform

sys.path.append('../src')

import trainer
import dataset
import model
import utils

## Training parameters

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('-b','--batch-size', help='Batch size in training', default=32)
parser.add_argument('-w','--workers', help='Number of workers', default=4)
parser.add_argument('--pretraining', help='Pretraining mode', action='store_true')
parser.add_argument('-lr', '--learning-rate', help='Learning rate', default=3.0e-4)
parser.add_argument('-d', '--dump-period', help='Dump period', default=1)
parser.add_argument('-e', '--epochs', help='Number of epochs to perform', default=1000)
parser.add_argument('-c', '--checkpoint', help='Checkpoint to load from', default=None)
parser.add_argument('--use-cuda', help='Use cuda for training', action='store_true')
parser.add_argument('--validate-on-train', help='Validate on train', action='store_true')
args = vars(parser.parse_args())

batch_size = args['batch_size']
num_workers = args['workers']
pretraining = args['pretraining']
learning_rate = args['learning_rate']
dump_period = args['dump_period']
epochs = args['epochs']
checkpoint = args['checkpoint']
use_cuda = args['use_cuda']
validate_on_train = args['validate_on_train']

## Making datasets
if __name__ == '__main__':
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

    ds_index = dataset.DataSetIndex('../../datasets/MNIST')
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

    ## Creating neural network and a socket for it

    net = model.Network()
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
                                        verbose=1,
                                        train_modules=pretrain_modules,
                                        use_cuda=use_cuda)
        for index in range(epochs):
            loss = my_pretrainer.train(train_loader)
            metrics = []
            if validate_on_train:
                for metric in my_pretrainer.validate(train_loader):
                    metrics.append(metric)
                    
            for metric in my_pretrainer.validate(valid_loader):
                metrics.append(metric)
            
            metrics =  '\t '.join([str(x) for x in metrics])
            print(str(index) + '\t' + metrics)
        
            if index % dump_period == 0:
                my_pretrainer.make_checkpoint('pretraining_', info=metrics)

    else:
        train_modules = socket.model.module.train_modules
    
        if checkpoint:
            train_optimizer = torch.optim.Adam(train_modules.parameters(), lr=learning_rate)
        
            my_trainer = trainer.Trainer(socket, 
                                         train_optimizer, 
                                         verbose=1, 
                                         train_modules=train_modules,
                                         use_cuda=use_cuda)
        
            my_trainer.load_checkpoint(checkpoint)

        socket = my_trainer.socket
        train_modules = socket.model.parameters()
        train_optimizer = torch.optim.Adam(train_modules, lr=learning_rate)
        my_trainer.optimizer = train_optimizer

        for index in range(epochs):
            loss = my_trainer.train(train_loader)
            #train_metrics = my_trainer.validate(train_loader)
            valid_metrics = my_trainer.validate(valid_loader)
            metrics =  '\t '.join([str(x) for x in train_metrics + valid_metrics])
            print(str(index) + '\t' + metrics)

            if index % dump_period == 0:
                my_trainer.make_checkpoint(info=metrics)