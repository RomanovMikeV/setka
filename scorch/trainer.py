import numpy
import time
import torch
import shutil
import gc
import collections
import os
from tqdm import tqdm
import horovod.torch as hvd

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
        self.silent = silent
        self.max_train_iterations = max_train_iterations
        self.max_valid_iterations = max_valid_iterations
        self.max_test_iterations = max_test_iterations
        self.use_cuda = use_cuda

    def train(self,
              train_loader):

        train_loader.dataset.shuffle()

        gc.enable()
        self.socket.epoch += 1

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()



        iterator = iter(train_loader)
        n_iterations = len(train_loader)
        if self.max_train_iterations >= 0:
            n_iterations = min(len(train_loader), self.max_train_iterations)

        self.socket.model.eval()

        gc.collect()

        pbar = tqdm(
            range(n_iterations), ascii=True,
            disable=self.silent, ncols=0)
        pbar.set_description(
            "Train -  "
            "D ----(----)  "
            "L --------(--------)")

        end = time.time()

        for i in pbar:
            start = time.time()
            input, target, ids = next(iterator)
            data_time.update(time.time() - start)

            if self.use_cuda:
                for index in range(len(input)):
                    input[index] = input[index].cuda()

                for index in range(len(target)):
                    target[index] = target[index].cuda()

            for opt_index in range(len(self.socket.optimizers)):
                if self.socket.optimizers[opt_index].active:
                    self.socket.optimizers[opt_index].module.train()

            output = self.socket.model.forward(input)
            loss = self.socket.criterion(output, target)

            for opt_index in range(len(self.socket.optimizers)):
                self.socket.optimizers[opt_index].optimizer.zero_grad()

            loss.backward()

            for opt_index in range(len(self.socket.optimizers)):
                if self.socket.optimizers[opt_index].active:
                    self.socket.optimizers[opt_index].optimizer.step()

            for opt_index in range(len(self.socket.optimizers)):
                if self.socket.optimizers[opt_index].active:
                    self.socket.optimizers[opt_index].module.eval()

            self.socket.iteration += 1

            losses.update(loss.data.item())

            batch_time.update(time.time() - end)
            end = time.time()

            del input, target
            del loss, output

            # Print status
            line = ("Train {0}  "
                    "D {data.avg:.2f}({data.val:.2f})  "
                    "L {loss.avg:.2e}({loss.val:.2e})  ".format(
                        self.socket.epoch,
                        time=batch_time,
                        data=data_time,
                        loss=losses))

            pbar.set_description(line)

        time.sleep(1)
        gc.collect()
        return losses.avg


    def validate(self, valid_loader):

        metrics = {}
        with torch.no_grad():

            valid_loader.dataset.shuffle()

            gc.enable()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            #if hvd.rank() == 0:
            outputs = []
            targets = []

            iterator = iter(valid_loader)
            n_iterations = len(valid_loader)
            if self.max_valid_iterations >= 0:
                n_iterations = min(len(valid_loader), self.max_valid_iterations)

            self.socket.model.eval()

            pbar = tqdm(
                range(n_iterations), ascii=True,
                disable=self.silent, ncols=0)

            end = time.time()
            pbar.set_description(
                "Valid -  "
                "D ----(----)  "
                "L --------(--------)")

            for i in pbar:

                start = time.time()
                input, target, ids = next(iterator)
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


                #if self.use_cuda:
                #    for index in range(len(output)):
                #        output[index] = output[index].cpu()

                #    for index in range(len(target)):
                #        target[index] = target[index].cpu()

                #if hvd.rank() == 0:
                for index in range(len(output)):
                    output[index] = hvd.allgather(output[index])

                for index in range(len(target)):
                    target[index] = hvd.allgather(target[index])

                outputs.append(output)
                targets.append(target)

                batch_time.update(time.time() - end)
                end = time.time()

                line = ("Valid {0}  "
                        "D {data.avg:.2f}({data.val:.2f})  "
                        "L {loss.avg:.2e}({loss.val:.2e})  ".format(
                            self.socket.epoch,
                            data=data_time,
                            loss=losses))

                del input
                del output, target

                if i == len(pbar) - 1: # and hvd.rank() == 0:
                    outputs = list(zip(*outputs))
                    targets = list(zip(*targets))

                    for index in range(len(outputs)):
                        outputs[index] = torch.cat(outputs[index], dim=0)

                    for index in range(len(targets)):
                        targets[index] = torch.cat(targets[index], dim=0)

                    metrics = {}

                    if hasattr(self.socket, "metrics"):
                        metrics = self.socket.metrics(outputs, targets)
                        for metric in metrics:
                            self.metrics = metrics

                        line += "Metrics: " + " ".join(
                            ["{}:{:.2e}".format(x, metrics[x]) for x in metrics])

                pbar.set_description(line)

            #if hvd.rank() == 0:
            del outputs, targets

            gc.collect()

            gc.collect()

        return metrics


    def test(self, test_loader, solo_test=False):
        with torch.no_grad():
            gc.enable()

            test_loader.dataset.shuffle()

            iterator = iter(test_loader)
            n_iterations = len(test_loader)

            if self.max_test_iterations >= 0:
                n_iterations = min(self.max_test_iterations, n_iterations)

            self.socket.model.eval()

            gc.collect()

            pbar = tqdm(range(n_iterations), ascii=True,
                        disable=self.silent, ncols=0)
            pbar.set_description("Test  ")

            for i in pbar:

                input, _, id = next(iterator)

                if self.use_cuda:
                    for index in range(len(input)):
                        input[index] = input[index].cuda()

                output = self.socket.model(input)

                for index in range(len(output)):
                    if not solo_test:
                        output[index] = hvd.allgather(output[index])

                    #print(output[index], '<- after')

                for index in range(len(input)):
                    if not solo_test:
                        input[index] = hvd.allgather(input[index])

                #if self.use_cuda:
                #    for index in range(len(output)):
                #        output[index] = output[index].cpu()
                #
                #    for index in range(len(input)):
                #        input[index] = input[index].cpu()

                gc.collect()

                yield input, output, id


    def make_checkpoint(self, prefix='./', is_best=False, info=""):

        checkpoint = {
            "epoch": self.socket.epoch,
            "iteration": self.socket.iteration,
            "model_state": self.socket.model.state_dict(),
            #"optimizer_state": self.socket.optimizer.state_dict(),
            "info": info,
            "metrics": self.socket.metric_vals}

        for opt_index in range(len(self.socket.optimizers)):
            checkpoint['optimizer_state_' + str(opt_index)] = self.socket.optimizers[opt_index].optimizer.state_dict()
            checkpoint['optimizer_switch_' + str(opt_index)] = self.socket.optimizers[opt_index].active

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
                         max_test_iterations=-1,
                         metric_mode='max',
                         use_cuda=True,
                         new_optimizer=False):

    checkpoint = torch.load(checkpoint_name)
    print(checkpoint['info'])
    print("Model restored from", checkpoint_name)

    restored_trainer = Trainer(socket,
                               silent=silent,
                               max_train_iterations=max_train_iterations,
                               max_valid_iterations=max_valid_iterations,
                               max_test_iterations=max_test_iterations,
                               metric_mode=metric_mode,
                               use_cuda=use_cuda)

    restored_trainer.socket.epoch = checkpoint['epoch']
    restored_trainer.socket.iteration = checkpoint['iteration']
    restored_trainer.socket.model.load_state_dict(checkpoint["model_state"])
    restored_trainer.socket.metric_vals = checkpoint["metrics"]
    #restored_trainer.metrics = checkpoint["metrics"]

    if not new_optimizer:
        for opt_index in range(len(restored_trainer.socket.optimizers)):
            try:
                restored_trainer.socket.optimizers[index].optimizer.load_state_dict(checkpoint["optimizer_state_" + str(opt_index)])
                restored_trainer.socket.optimizers[index].active = checkpoint["optimizer_active_" + str(opt_index)]
            except:
                print('Failed to load optimizer ' + str(opt_index) + '.')

    return restored_trainer
