import torch
import numpy
import time
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer():
    def __init__(self,
                 socket,
                 optimizer,
                 train_modules,
                 verbose=-1,
                 max_train_iterations=-1,
                 max_valid_iterations=-1,
                 metric_mode='max',
                 use_cuda=True):
        
        self.socket = socket
        self.epoch = 0
        self.optimizer = optimizer
        self.verbose = verbose
        self.train_modules = train_modules
        self.max_train_iterations = max_train_iterations
        self.max_valid_iterations = max_valid_iterations
        self.use_cuda = use_cuda
        
    def train(self, 
              train_loader):
        
        self.epoch += 1
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        
        if self.verbose >= 0:
            print("Training epoch ", self.epoch)
    
        for i, (input, target) in enumerate(train_loader):
            
            if self.use_cuda:
                for index in range(len(input)):
                    input[index] = torch.autograd.Variable(input[index].cuda())
                    
            if self.use_cuda:
                for index in range(len(target)):
                    target[index] = torch.autograd.Variable(target[index].cuda())
            
            self.train_modules.train()
            output = self.socket.model.forward(input)
            self.train_modules.eval()
            
            loss = self.socket.criterion(output, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.data[0])
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            del input, target
            del loss, output
            
            # Print status
            if self.verbose >= 1:
                if i % self.verbose == 0:
                    print("Ep {0}[{1}/{2}]\t"
                          "Time  {time.avg:.2f}({time.val:.2f})\t"
                          "Data  {data.avg:.2f}({data.val:.2f})\t"
                          "Loss  {loss.avg:.2e}({loss.val:.2e})\t"
                          "LR {lr:.2e}".format(
                              self.epoch, i, len(train_loader),
                              time=batch_time,
                              data=data_time,
                              loss=losses,
                              lr=self.optimizer.param_groups[-1]['lr']))
            
            # Stop epoch if needed
            if self.max_train_iterations > 0:
                if i % self.max_train_iterations == 0 and i != 0:
                    break
        
        time.sleep(1)
        return losses.avg
            
    
    def validate(self, valid_loader):
        outputs = []
        targets = []
        
        if self.verbose >= 0:
            print("Validating epoch", self.epoch)
        
        for i, (input, target) in enumerate(valid_loader):
            
            if self.use_cuda:
                for index in range(len(input)):
                    input[index] = torch.autograd.Variable(input[index].cuda())
            
            if self.use_cuda:
                for index in range(len(target)):
                    target[index] = torch.autograd.Variable(target[index].cuda())
            
            output = self.socket.model.forward(input)
            loss = self.socket.criterion(output, target)
            
            if self.use_cuda:
                for index in range(len(output)):
                    output[index] = output[index].cpu()
                    
            if self.use_cuda:
                for index in range(len(target)):
                    target[index] = target[index].cpu()
            
            
            for index in range(len(output)):
                output[index] = output[index].data
                
            for index in range(len(target)):
                target[index] = target[index].data
            
            outputs.append(output)
            targets.append(target)
            
            if self.max_valid_iterations > 0:
                if i % self.max_valid_iterations == 0 and i != 0:
                    break
                    
            del loss, input
            
        if self.verbose >= 0:
            print("Validation results computed, making metrics")
            
        outputs = list(zip(*outputs))
        targets = list(zip(*targets))
        
        for index in range(len(outputs)):
            outputs[index] = torch.cat(outputs[index], dim=0)
            
        for index in range(len(targets)):
            targets[index] = torch.cat(targets[index], dim=0)
        
        metrics = self.socket.metrics(outputs, targets)
        if self.verbose >= 0:
            print("Metrics:\t", "\t".join(["{:.2e}".format(x) for x in metrics]) )
        
        time.sleep(1)
        return metrics
    
    def make_checkpoint(self, prefix='./', is_best=False, info=""):
        
        checkpoint = {
            "epoch": self.epoch,
            "model_state": self.socket.model.state_dict(),
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer.state_dict(),
            "verbose": self.verbose,
            "info": info}
        
        torch.save(checkpoint, prefix + 'checkpoint.pth.tar')
        
        if is_best:
            shutil.copy(prefix + 'checkpoint.pth.tar', prefix + 'checkpoint_best.pth.tar')
    
    def load_checkpoint(self, checkpoint_name, load_model_structure=False):
        checkpoint = torch.load(checkpoint_name)
        print("Loaded ", checkpoint_name)
        print(checkpoint["info"])
        
        self.epoch = checkpoint['epoch']
        self.optimizer = checkpoint['optimizer']
        self.verbose = checkpoint['verbose']
        
        if load_model_structure:
            self.socket = checkpoint['socket']
        
        self.socket.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
    