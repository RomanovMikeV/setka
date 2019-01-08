import os
import shutil
import torch
import tensorboardX

from . import internal

class Callback():
    '''
    Callback basic class.

    Callback has the following methods:

    __init__(self) -- constructor

    on_train_begin(self) -- method that is executed when training starts
        (in the beginning of ```trainer.train```)

    on_train_end(self) -- method that is executed when training starts
        (in the end of ```trainer.train```)

    on_epoch_begin(self) -- method that is executed when the epoch starts
        (in the beginning of ```train_one_epoch```, ```validate_one_epoch```,
        ```predict```)

    on_epoch_end(self) -- method that is executed when the epoch starts
        (in the end of ```train_one_epoch```, ```validate_one_epoch```,
        ```predict```)

    on_batch_begin(self) -- method that is executed when the batch starts
        (in the beginning of training cycle body in ```train_one_epoch```,
        ```validate_one_epoch```, ```predict```)

    on_batch_end(self) -- method that is executed when the batch starts
        (in the end of training cycle body in ```train_one_epoch```,
        ```validate_one_epoch```, ```predict```)

    set_trainer(self, trainer) -- method that links the trainer to the callback.
    '''

    def __init__(self):
        pass

    def on_init(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def set_trainer(self, trainer):
        self.trainer = trainer


class PrintCallback(Callback):
    '''
    Callback for testing.
    '''
    def on_init(self):
        print("Init")

    def on_train_begin(self):
        print("Starting training")

    def on_train_end(self):
        print("Finishing training")

    def on_epoch_begin(self):
        print("Epoch strating")

    def on_epoch_end(self):
        print("Epoch end")

    def on_batch_begin(self):
        print("Batch start")

    def on_batch_end(self):
        print("Batch end")


class SaveResult(Callback):
    '''
    '''
    def __init__(self, f=None, mode='predicting', dir='predictions'):
        self.f = f
        self.dir = dir
        self.index = 0
        self.mode = mode

        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

    def on_batch_end(self):

        if self.trainer._mode == self.mode:
            res = {}
            for index in range(len(self.trainer._ids)):

                one_input = []
                for input_index in range(len(self.trainer._input)):
                    one_input.append(self.trainer._input[input_index][index])

                one_output = []
                for output_index in range(len(self.trainer._output)):
                    one_output.append(self.trainer._output[output_index][index])

                if self.f is not None:
                    res[self.trainer._ids[index]] = self.f(
                        one_input,
                        one_output)
                else:
                    res[self.trainer._ids[index]] = one_output

            #print("Saving result")
            torch.save(res, os.path.join(self.dir, str(self.index) + '.pth.tar'))

            del self.trainer._output

            self.trainer._output = res

            self.index += 1


class ShuffleDataset(Callback):
    def __init__(self, shuffle_valid=False):
        self.shuffle_valid = shuffle_valid

    def on_epoch_begin(self):
        if (self.trainer._mode == 'training' or
            (self.trainer_mode == 'validation' and self.shuffle_valid)):

            self.trainer._dataset.shuffle()


class ComputeMetrics(Callback):
    def __init__(self,
                 metrics=None,
                 steps_to_compute=1):

        self.steps_to_compute = steps_to_compute
        self.metrics = metrics
        self.steps = 0
        self.avg_metrics = {}

    def on_train_begin(self):
        if self.metrics is None:
            self.metrics = {'main': self.trainer._criterion}

    def on_epoch_begin(self):
        # clear all cache
        self.outputs = []
        self.targets = []

    def on_batch_end(self):
        if self.trainer._mode == 'training' or self.trainer._mode == 'validating':
            self.steps += 1

            one_output = [x.detach() for x in self.trainer._output]
            one_target = [x.detach() for x in self.trainer._target]

            self.outputs.append(one_output)
            self.targets.append(one_target)

            if self.steps >= self.steps_to_compute:

                self.outputs = list(zip(*self.outputs))
                self.targets = list(zip(*self.targets))

                for index in range(len(self.outputs)):
                    self.outputs[index] = torch.cat(self.outputs[index], dim=0)

                for index in range(len(self.targets)):
                    self.targets[index] = torch.cat(self.targets[index], dim=0)

                metrics = {x:self.metrics[x](self.outputs, self.targets)
                       for x in self.metrics}

                for metric in metrics:
                    if metric not in self.avg_metrics:
                        self.avg_metrics[metric] = internal.AverageMeter()
                    self.avg_metrics[metric].update(metrics[metric])

                del self.outputs
                del self.targets

                self.outputs = []
                self.targets = []
                self.steps = 0

            self.trainer._line += " ".join(
               ["{}: {:.2e}({:.2e})".format(x, self.avg_metrics[x].avg,
                self.avg_metrics[x].val)
                for x in self.avg_metrics])

    def on_epoch_end(self):
        # finilize metrics

        del self.outputs
        del self.targets

        self.avg_metrics = {x:self.avg_metrics[x].avg for x in self.avg_metrics}

        if self.trainer._mode == 'validating' and self.trainer._subset == 'train':
            self.trainer._train_metrics = self.avg_metrics
        elif self.trainer._mode == 'validating' and self.trainer._subset == 'valid':
            self.trainer._valid_metrics = self.avg_metrics

        self.avg_metrics = {}


class MakeCheckpoints(Callback):
    def __init__(self,
                 metric='main',
                 mode='max',
                 name='checkpoint'):
        self.best_metric = None
        self.metric = metric
        self.mode = mode
        self.name = name

    def on_epoch_end(self):

        is_best = False

        if self.trainer._mode == 'validating' and self.trainer._subset == 'valid':
            if self.best_metric is None:
                self.best_metric = self.trainer._valid_metrics[self.metric]
                is_best = True

            if ((self.best_metric < self.trainer._valid_metrics[self.metric] and
                 self.mode == 'max') or
                (self.best_metric > self.trainer._valid_metrics[self.metric] and
                 self.mode == 'min')):

                 self.best_metric = self.trainer._valid_metrics[self.metric]
                 is_best = True

            self.trainer.save(self.name + '.pth.tar')
            if is_best:
                self.trainer.save(self.name + '_best.pth.tar')


class WriteToTensorboard(Callback):
    def __init__(self,
                 processing_f=None,
                 write_flag=True,
                 name='checkpoint'):
        self.tb_writer = tensorboardX.SummaryWriter()
        self.processing_f = processing_f
        self.write_flag = write_flag
        self.name = name

    def on_epoch_end(self):
        if self.trainer._mode == 'validating':
            if self.trainer._subset == 'valid':
                for metric_name in self.trainer._valid_metrics:
                    data = {}

                    data['valid'] = self.trainer._valid_metrics[metric_name]

                    if hasattr(self.trainer, '_train_metrics'):
                        data['train'] = self.trainer._train_metrics[metric_name]

                    self.tb_writer.add_scalars(
                        'metrics/' + metric_name + '/' + self.name,
                        data,
                        self.trainer._epoch)

            #if self.trainer._subset == 'valid':
            #    for metric_name in self.trainer._valid_metrics:
            #        self.tb_writer.add_scalars(
            #            'metrics/' + metric_name + '/' + self.name,
            #            self.trainer._valid_metrics[metric_name],
            #            self.trainer._epoch)
