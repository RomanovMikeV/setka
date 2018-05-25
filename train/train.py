import torch
import torchvision
import skimage.transform
import sys
import argparse
import time
import importlib.util

import torchvision.transforms as transform

sys.path.append('../src')

import trainer
import utils

## Training parameters

parser = argparse.ArgumentParser(
    description='Script to train a specified model with a specified dataset.')
parser.add_argument('-b','--batch-size', help='Batch size', default=32, type=int)
parser.add_argument('-w','--workers', help='Number of workers in a dataloader', default=4, type=int)
parser.add_argument('--pretraining', help='Pretraining mode', action='store_true')
parser.add_argument('-lr', '--learning-rate', help='Learning rate', default=3.0e-4, type=float)
parser.add_argument('-d', '--dump-period', help='Dump period', default=1, type=int)
parser.add_argument('-e', '--epochs', help='Number of epochs to perform', default=1000, type=int)
parser.add_argument('-c', '--checkpoint', help='Checkpoint to load from', default=None)
parser.add_argument('--use-cuda', help='Use cuda for training', action='store_true')
parser.add_argument('--validate-on-train', help='Validate on train', action='store_true')
parser.add_argument('--model', help='File with a model specifications', required=True)
parser.add_argument('--dataset', help='File with a dataset sepcification', required=True)
parser.add_argument('--max-train-iterations', help='Maximum training iterations', default=-1, type=int)
parser.add_argument('--max-valid-iterations', help='Maximum validation iterations', default=-1, type=int)
parser.add_argument('-dp', '--dataset-path', help='Path to the dataset', required=True)
parser.add_argument('-v', '--verbosity', 
                    help='-1 for no output, 0 for epoch output, positive number is printout frequency', 
                    default=-1, type=int)
parser.add_argument('-cp', '--checkpoint-prefix', help='Prefix to the checkpoint name', default='')
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
max_train_iterations = args['max_train_iterations']
max_valid_iterations = args['max_valid_iterations']
dataset_path = args['dataset_path']
verbosity = args['verbosity']
checkpoint_prefix = args['checkpoint_prefix']

spec = importlib.util.spec_from_file_location("model", args['model'])
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

spec = importlib.util.spec_from_file_location("dataset", args['dataset'])
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)

## Making datasets
if __name__ == '__main__':
    
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

    ## Creating neural network and a socket for it

    net = model.Network()
    net = torch.nn.parallel.DataParallel(net).eval()

    if use_cuda:
        net = net.cuda()
    
    socket = model.Socket(net)

    train_modules = socket.model.module.train_modules
    
    train_optimizer = torch.optim.Adam(
            train_modules.parameters(), 
            lr=learning_rate)
    
    if pretraining:
        train_modules = socket.mode.module.pretrain_modules
        checkpoint_prefix = 'pretraining' + checkpoint_prefix
    
    my_trainer = trainer.Trainer(socket, 
                                 train_optimizer,
                                 train_modules,
                                 verbosity=verbosity,
                                 use_cuda=use_cuda,
                                 max_train_iterations=max_train_iterations,
                                 max_valid_iterations=max_valid_iterations)
    if checkpoint is not None:
        my_trainer = trainer.load_from_checkpoint(checkpoint,
                                                  socket,
                                                  train_optimizer,
                                                  train_modules,
                                                  verbosity=verbosity,
                                                  use_cuda=use_cuda,
                                                  max_train_iterations=max_train_iterations,
                                                  max_valid_iterations=max_valid_iterations)
        
        
    for index in range(epochs):
        loss = my_trainer.train(train_loader)
        metrics = []
        if validate_on_train:
            for metric in my_trainer.validate(train_loader):
                metrics.append(metric)
                    
        for metric in my_trainer.validate(valid_loader):
            metrics.append(metric)
            
        if index % dump_period == 0:
                my_trainer.make_checkpoint(checkpoint_prefix, info=metrics)