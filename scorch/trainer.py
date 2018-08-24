import numpy
import time
import torch
import shutil
import gc
import collections

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
                 verbosity=-1,
                 max_train_iterations=-1,
                 max_valid_iterations=-1,
                 metric_mode='max',
                 use_cuda=True,
                 name='Trainer'):

        self.socket = socket
        self.epoch = 0
        self.verbosity = verbosity
        self.max_train_iterations = max_train_iterations
        self.max_valid_iterations = max_valid_iterations
        self.use_cuda = use_cuda
        self.metrics = None

    def train(self,
              train_loader):

        self.epoch += 1

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        if self.verbosity >= 0:
            print("Training epoch ", self.epoch)

        iterator = iter(train_loader)
        n_iterations = len(train_loader)
        if self.max_train_iterations > 0:
            n_iterations = min(len(train_loader), self.max_train_iterations)

        for i in range(n_iterations):
            input, target = next(iterator)

            if self.use_cuda:
                for index in range(len(input)):
                    input[index] = input[index].cuda()

            if self.use_cuda:
                for index in range(len(target)):
                    target[index] = target[index].cuda()

            self.socket.train_modules.train()
            output = self.socket.model.forward(input)


            loss = self.socket.criterion(output, target)

            self.socket.optimizer.zero_grad()
            loss.backward()
            self.socket.optimizer.step()
            self.socket.train_modules.eval()

            losses.update(loss.data.item())

            batch_time.update(time.time() - end)
            end = time.time()

            del input, target
            del loss, output

            # Print status
            if self.verbosity >= 1:
                if i % self.verbosity == 0:
                    print("Ep {0}[{1}/{2}]\t"
                          "Time  {time.avg:.2f}({time.val:.2f})\t"
                          "Data  {data.avg:.2f}({data.val:.2f})\t"
                          "Loss  {loss.avg:.2e}({loss.val:.2e})\t"
                          "LR {lr:.2e}".format(
                              self.epoch,
                              i + 1,
                              n_iterations,
                              time=batch_time,
                              data=data_time,
                              loss=losses,
                              lr=self.socket.optimizer.param_groups[-1]['lr']))

            # Stop epoch if needed
            if i + 1 >= n_iterations:
                break

        train_loader.dataset.ds_index.shuffle()

        time.sleep(1)
        gc.collect()
        return losses.avg


    def validate(self, valid_loader):
        gc.enable()

        outputs = []
        targets = []

        losses = []

        if self.verbosity >= 0:
            print("Validating epoch", self.epoch)


        iterator = iter(valid_loader)
        n_iterations = len(valid_loader)
        if self.max_valid_iterations > 0:
            n_iterations = min(len(valid_loader), self.max_valid_iterations)


        for i in range(n_iterations):
            input, target = next(iterator)
            if self.use_cuda:
                for index in range(len(input)):
                    input[index] = input[index].cuda()

            if self.use_cuda:
                for index in range(len(target)):
                    target[index] = target[index].cuda()

            output = self.socket.model.forward(input)
            loss = self.socket.criterion(output, target)

            losses.append(loss.data.item())


            for index in range(len(output)):
                output[index] = output[index].detach()

            for index in range(len(target)):
                target[index] = target[index].detach()


            if self.use_cuda:
                for index in range(len(output)):
                    output[index] = output[index].cpu()

            if self.use_cuda:
                for index in range(len(target)):
                    target[index] = target[index].cpu()


            for index in range(len(output)):
                output[index] = output[index]

            for index in range(len(target)):
                target[index] = target[index]

            outputs.append(output)
            targets.append(target)

            if i + 1 >= n_iterations:
                break

            del input
            del output, target

        outputs = list(zip(*outputs))
        targets = list(zip(*targets))

        for index in range(len(outputs)):
            outputs[index] = torch.cat(outputs[index], dim=0)

        for index in range(len(targets)):
            targets[index] = torch.cat(targets[index], dim=0)

        metrics = self.socket.metrics(outputs, targets)

        del outputs, targets
        gc.collect()

        if self.verbosity >= 0:
            print("Metrics:\t", "\t\t".join(["{}:{:.2e}".format(x, metrics[x]) for x in metrics]) )

        time.sleep(1)

        self.metrics = metrics
        gc.collect()

        return metrics


    def make_checkpoint(self, prefix='./', is_best=False, info=""):

        checkpoint = {
            "epoch": self.epoch,
            "model_state": self.socket.model.state_dict(),
            "optimizer_state": self.socket.optimizer.state_dict(),
            "verbosity": self.verbosity,
            "info": info,
            "metrics": self.metrics}

        torch.save(checkpoint, prefix + 'checkpoint.pth.tar')

        if is_best:
            shutil.copy(prefix + 'checkpoint.pth.tar', prefix + 'checkpoint_best.pth.tar')

    def restore_from_checkpoit(self, checkpoint_name, load_model_structure=False):
        checkpoint = torch.load(checkpoint_name)
        print("Loaded ", checkpoint_name)
        print(checkpoint["info"])

        self.epoch = checkpoint['epoch']
        #self.optimizer = checkpoint['optimizer']
        self.verbosity = checkpoint['verbosity']
        self.metrics = checkpoint['metrics']

        if load_model_structure:
            self.socket = checkpoint['socket']

        self.socket.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])



def load_from_checkpoint(checkpoint_name,
                         socket,
                         verbosity=-1,
                         max_train_iterations=-1,
                         max_valid_iterations=-1,
                         metric_mode='max',
                         use_cuda=True):

    checkpoint = torch.load(checkpoint_name)
    print(checkpoint['info'])
    print("Model restored from", checkpoint_name)

    restored_trainer = Trainer(socket,
                               verbosity=verbosity,
                               max_train_iterations=max_train_iterations,
                               max_valid_iterations=max_valid_iterations,
                               metric_mode=metric_mode,
                               use_cuda=use_cuda)

    restored_trainer.epoch = checkpoint['epoch']
    #restored_trainer.optimizer = checkpoint['optimizer']
    restored_trainer.socket.model.load_state_dict(checkpoint["model_state"])
    try:
        restored_trainer.socket.optimizer.load_state_dict(checkpoint["optimizer_state"])
    except:
        print('Failed to load optimizer state, starting to train from scratch')

    return restored_trainer
