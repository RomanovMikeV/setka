import numpy
import time
import torch
import shutil
import gc
import collections
import os
from tqdm import tqdm

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
                 silent=False,
                 max_train_iterations=-1,
                 max_valid_iterations=-1,
                 max_test_iterations=-1,
                 metric_mode='max',
                 use_cuda=True,
                 name='Trainer'):

        self.socket = socket
        self.epoch = 0
        self.silent = silent
        self.max_train_iterations = max_train_iterations
        self.max_valid_iterations = max_valid_iterations
        self.max_test_iterations = max_test_iterations
        self.use_cuda = use_cuda
        self.metrics = None

    def train(self,
              train_loader):

        gc.enable()
        self.epoch += 1

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()



        iterator = iter(train_loader)
        n_iterations = len(train_loader)
        if self.max_train_iterations >= 0:
            n_iterations = min(len(train_loader), self.max_train_iterations)

        self.socket.model.eval()

        gc.collect()

        pbar = tqdm(range(n_iterations), ascii=True, disable=self.silent)
        pbar.set_description("Ep: - "
                "Time: ----(----)  "
                "Data: ----(----)  "
                "Loss:  --------(--------)  "
                "LR: --------")

        end = time.time()

        for i in pbar:
            start = time.time()
            input, target = next(iterator)
            data_time.update(time.time() - start)

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
            line = ("Train: {0} "
                    "Time: {time.avg:.2f}({time.val:.2f})  "
                    "Data: {data.avg:.2f}({data.val:.2f})  "
                    "Loss:  {loss.avg:.2e}({loss.val:.2e})  "
                    "LR: {lr:.2e}".format(
                        self.epoch,
                        time=batch_time,
                        data=data_time,
                        loss=losses,
                        lr=self.socket.optimizer.param_groups[-1]['lr']))

            pbar.set_description(line)

        train_loader.dataset.ds_index.shuffle()

        time.sleep(1)
        gc.collect()
        return losses.avg


    def validate(self, valid_loader):

        gc.enable()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        outputs = []
        targets = []

        iterator = iter(valid_loader)
        n_iterations = len(valid_loader)
        if self.max_valid_iterations >= 0:
            n_iterations = min(len(valid_loader), self.max_valid_iterations)

        self.socket.model.eval()

        pbar = tqdm(range(n_iterations), ascii=True, disable=self.silent)

        end = time.time()
        pbar.set_description("Valid: - "
                "Time: ----(----)  "
                "Data: ----(----)  "
                "Loss:  --------(--------)  "
                "LR: --------")

        for i in pbar:

            start = time.time()
            input, target = next(iterator)
            data_time.update(time.time() - start)

            if self.use_cuda:
                for index in range(len(input)):
                    input[index] = input[index].cuda()

            if self.use_cuda:
                for index in range(len(target)):
                    target[index] = target[index].cuda()

            output = self.socket.model.forward(input)
            loss = self.socket.criterion(output, target)

            losses.update(loss.data.item())


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

            batch_time.update(time.time() - end)
            end = time.time()

            line = ("Valid: {0} "
                    "Time: {time.avg:.2f}({time.val:.2f})  "
                    "Data: {data.avg:.2f}({data.val:.2f})  "
                    "Loss:  {loss.avg:.2e}({loss.val:.2e})  "
                    "LR: {lr:.2e}".format(
                        self.epoch,
                        time=batch_time,
                        data=data_time,
                        loss=losses,
                        lr=self.socket.optimizer.param_groups[-1]['lr']))

            pbar.set_description(line)

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

        print("Metrics: | ", " | ".join(["{}:{:.2e}".format(x, metrics[x]) for x in metrics]), " |" )

        time.sleep(1)

        self.metrics = metrics
        gc.collect()

        return metrics


    def test(self, test_dataloader):
        gc.enable()

        iterator = iter(test_dataloader)
        n_iterations = len(test_dataloader)

        if self.max_test_iterations >= 0:
            n_iterations = min(self.max_test_iterations, n_iterations)

        self.socket.model.eval()

        inputs = []
        outputs = []

        gc.collect()

        pbar = tqdm(range(n_iterations), ascii=True, disable=self.silent)
        pbar.set_description("Testing: ")

        for i in pbar:

            input, target = next(iterator)

            if self.use_cuda:
                for index in range(len(input)):
                    input[index] = input[index].cuda()

            output = self.socket.model(input)

            for index in range(len(output)):
                output[index] = output[index].detach()

            if self.use_cuda:
                for index in range(len(output)):
                    output[index] = output[index].cpu()

            gc.collect()

            inputs.append(input)
            outputs.append(output)

        return inputs, outputs



    def make_checkpoint(self, prefix='./', is_best=False, info=""):

        checkpoint = {
            "epoch": self.epoch,
            "model_state": self.socket.model.state_dict(),
            "optimizer_state": self.socket.optimizer.state_dict(),
            "info": info,
            "metrics": self.metrics}

        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        torch.save(checkpoint,
                   'checkpoints/' + prefix + '.pth.tar')

        if is_best:
            shutil.copy('checkpoints/' + prefix + '.pth.tar',
                        'checkpoints/' + prefix + '_best.pth.tar')



def load_from_checkpoint(checkpoint_name,
                         socket,
                         silent=False,
                         max_train_iterations=-1,
                         max_valid_iterations=-1,
                         metric_mode='max',
                         use_cuda=True):

    checkpoint = torch.load(checkpoint_name)
    print(checkpoint['info'])
    print("Model restored from", checkpoint_name)

    restored_trainer = Trainer(socket,
                               silent=silent,
                               max_train_iterations=max_train_iterations,
                               max_valid_iterations=max_valid_iterations,
                               metric_mode=metric_mode,
                               use_cuda=use_cuda)

    restored_trainer.epoch = checkpoint['epoch']
    restored_trainer.socket.model.load_state_dict(checkpoint["model_state"])
    try:
        restored_trainer.socket.optimizer.load_state_dict(checkpoint["optimizer_state"])
    except:
        print('Failed to load optimizer state, starting to train from scratch')

    return restored_trainer
