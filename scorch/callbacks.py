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
    Callback for saving predictions of the model. The results are
    stored in a specified directory. Batches are processed with the specified
    function ```f```. The directory is flushed during the __init__ and the
    result is saved when the batch is finished (on_batch_end).
    '''
    def __init__(self, f=None, mode='predicting', dir='predictions'):
        '''
        Constructor

        Args:
            f (function): function to process the predictions.

            mode (string): mode of the model when the predictions should be
            saved.

            dir (string): directory where the results should be stored.
        '''
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

            self.index += 1


class ShuffleDataset(Callback):
    '''
    Callback to shuffle datasets before the epoch starts (on_epoch_begin).
    '''
    def __init__(self, shuffle_valid=False):
        '''
        Constructor.

        shuffle_valid (bool): shuffle validation dataset too if True. If False
            only the training dataset will be shuffled.
        '''
        self.shuffle_valid = shuffle_valid

    def on_epoch_begin(self):
        if (self.trainer._mode == 'training' or
            (self.trainer_mode == 'validation' and self.shuffle_valid)):

            self.trainer._dataset.shuffle()


class ComputeMetrics(Callback):
    '''
    Callbacks for metric computation.
    '''
    def __init__(self,
                 metrics=None,
                 steps_to_compute=1):

        '''
        Constructor.

        metrics (dict): dictionary with metrics to compute.

        steps_to_compute (int): how many steps to perform before metrics
            computation.
        '''

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
        self.steps = 0

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

                self.outputs.clear()
                self.targets.clear()
                self.steps = 0

            self.trainer._line += " ".join(
               ["{}: {:.2e}({:.2e})".format(x, self.avg_metrics[x].avg,
                self.avg_metrics[x].val)
                for x in self.avg_metrics])

    def on_epoch_end(self):
        # finilize metrics

        self.outputs.clear()
        self.targets.clear()

        self.avg_metrics = {x:self.avg_metrics[x].avg for x in self.avg_metrics}

        if self.trainer._mode == 'validating' and self.trainer._subset == 'train':
            self.trainer._train_metrics = self.avg_metrics
        elif self.trainer._mode == 'validating' and self.trainer._subset == 'valid':
            self.trainer._val_metrics = self.avg_metrics

        self.avg_metrics = {}


class MakeCheckpoints(Callback):
    '''
    Callback to make checkpoints. This callback takes into account the
    ```metric```. If ```max_metric``` is ```True``` -- the higher the
    monitored metric is the better the model is, if not -- the lower the
    monitored metric is the better the model is. Checkpoint name is specified
    with ```name```, the best model is saved with ```_best``` postfix.
    '''
    def __init__(self,
                 metric='main',
                 max_mode=False,
                 name='checkpoint'):
        self.best_metric = None
        self.metric = metric
        self.max_mode = max_mode
        self.name = name

    def on_epoch_end(self):

        is_best = False

        if self.trainer._mode == 'validating' and self.trainer._subset == 'valid':
            if self.best_metric is None:
                self.best_metric = self.trainer._val_metrics[self.metric]
                is_best = True

            if ((self.best_metric < self.trainer._val_metrics[self.metric] and
                 self.max_mode) or
                (self.best_metric > self.trainer._val_metrics[self.metric] and
                 not self.max_mode)):

                 self.best_metric = self.trainer._val_metrics[self.metric]
                 is_best = True

            self.trainer.save(self.name + '.pth.tar')
            if is_best:
                self.trainer.save(self.name + '_best.pth.tar')


class WriteToTensorboard(Callback):
    '''
    Callback to write the metrics to the TensorboardX.
    If ```write_flag``` is ```False``` the results are not written to the
    Tensorboard, ```True``` by default. ```name``` is a label of the model.
    It is possible to add ```processing_f``` function that takes as inputs:
    input, target and output and returns the dictionary with the outputs for
    visualization. This dictionary may be represented as:
    {
        "images": dict with numpy images,
        "texts": dict with texts,
        "audios": dict with numpy audios,
        "figures": dict with matplotlib figures,
        "graphs": dict with onnx graphs,
        "embeddings": dict with embeddings
    }
    Each of the dicts should have the following structure: {sample_id: result}.
    '''
    def __init__(self,
                 processing_f=None,
                 write_flag=True,
                 name='checkpoint'):
        self.tb_writer = tensorboardX.SummaryWriter()
        self.f = processing_f
        self.write_flag = write_flag
        self.name = name

    def on_epoch_end(self):
        if self.trainer._mode == 'validating' and self.write_flag:
            if self.trainer._subset == 'valid':
                for metric_name in self.trainer._val_metrics:
                    data = {}

                    data['valid'] = self.trainer._val_metrics[metric_name]

                    if hasattr(self.trainer, '_train_metrics'):
                        data['train'] = self.trainer._train_metrics[metric_name]

                    self.tb_writer.add_scalars(
                        self.name + '/' + metric_name,
                        data,
                        self.trainer._epoch)

    def show(self, to_show, id):

        type_writers = {
            'images': self.tb_writer.add_image,
            'texts': self.tb_writer.add_text,
            'audios': self.tb_writer.add_audio,
            'figures': (lambda x, y, z: self.tb_writer.add_figure(x, y, z)),
            'graphs': self.tb_writer.add_graph,
            'embeddings': self.tb_writer.add_embedding}

        for type in type_writers:
            if type in to_show:
                for desc in to_show[type]:
                    type_writers[type](self.name + '/' + str(id) + '/' + desc,
                        to_show[type][desc], str(self.trainer._epoch))

    def on_batch_end(self):
        if self.trainer._mode == 'predicting' and self.write_flag and (self.f is not None):
            for index in range(len(self.trainer._ids)):

                one_input = []
                for input_index in range(len(self.trainer._input)):
                    one_input.append(self.trainer._input[input_index][index])

                one_target = []
                for target_index in range(len(self.trainer._target)):
                    one_target.append(self.trainer._target[target_index][index])

                one_output = []
                for output_index in range(len(self.trainer._output)):
                    one_output.append(self.trainer._output[output_index][index])

                res = self.f(one_input, one_target, one_output)
                id = self.trainer._ids[index]

                self.show(res, id)


class UnfreezeOnPlateau(Callback):
    '''
    This callback allows you to unfreeze optimizers one by one when the plateau
    is reached. If used together with ReduceLROnPlateau, should be listed after
    it. The following arguments may be specified:

    metric (str) : name of the metric to monitor.

    cooldown (int) : minimal amount of epochs between two learning rate changes.

    limit (int) : amount of epochs since the last improvement of maximum of
        the monitored metric before the learning rate change.

    max_mode (bool) : if True then the higher is the metric the better.
        Otherwise the lower is the metric the better.
    '''
    def __init__(self, cooldown=5, limit=5, metric='loss', max_mode=False):
        self.cooldown = cooldown
        self.limit = limit

        self.since_last = 0
        self.since_best = 0

        self.best_metric = None
        self.metric = metric
        self.max_mode = max_mode

        self.optimizer_index = 1


    def on_init(self):
        self.trainer._lr_reduce = False
        self.complete = False


    def on_epoch_end(self):
        if self.trainer._mode == 'validating' and self.trainer._subset == 'valid':
            if self.best_metric is None:
                self.best_metric = self.trainer._val_metrics[self.metric]
                self.since_best = 0

            else:
                new_metric = self.trainer._val_metrics[self.metric]

                if ((new_metric > self.best_metric and self.max_mode) or
                    (new_metric < self.best_metric and not self.max_mode)):

                    self.best_metric = new_metric
                    self.since_best = 0

            if self.since_last >= self.cooldown and self.since_best >= self.limit and not self.complete:
                print("Unfreezing optimizer ", str(self.optimizer_index))
                self.trainer._optimizers[self.optimizer_index].is_active = True
                self.since_last = 0
                self.optimizer_index += 1
                if self.optimizer_index >= len(self.trainer._optimizers):
                    self.trainer._lr_reduce = True
                    self.complete = True

            self.since_best += 1
            self.since_last += 1


class ReduceLROnPlateau(Callback):
    '''
    This callback allows you to unfreeze optimizers one by one when the plateau
    is reached. The following arguments may be specified:

    metric (str) : name of the metric to monitor.

    cooldown (int) : minimal amount of epochs between two learning rate changes.

    limit (int) : amount of epochs since the last improvement of maximum of
        the monitored metric before the learning rate change.

    max_mode (bool) : if True then the higher is the metric the better.
        Otherwise the lower is the metric the better.
    '''
    def __init__(self, cooldown=5, limit=5, factor=0.5, metric='loss', max_mode=False):
        self.cooldown = cooldown
        self.limit = limit
        self.factor = factor

        self.since_last = 0
        self.since_best = 0

        self.best_metric = None
        self.metric = metric
        self.max_mode = max_mode

        self.optimizer_index = 1


    def on_init(self):
        self.trainer._lr_reduce=True


    def on_epoch_end(self):
        if self.trainer._mode == 'validating' and self.trainer._subset == 'valid' and self.trainer._lr_reduce:
            if self.best_metric is None:
                self.best_metric = self.trainer._val_metrics[self.metric]
                self.since_best = 0

            else:
                new_metric = self.trainer._val_metrics[self.metric]

                if ((new_metric > self.best_metric and self.max_mode) or
                    (new_metric < self.best_metric and not self.max_mode)):

                    self.best_metric = new_metric
                    self.since_best = 0

            if self.since_last >= self.cooldown and self.since_best >= self.limit:
                print('Reducing learning rate')
                for optimizer in self.trainer._optimizers:
                    for g in optimizer.optimizer.param_groups:
                        g['lr'] *= self.factor
                self.since_last = 0

        self.since_best += 1
        self.since_last += 1



class CyclicLR(Callback):
    '''
    This callback allows cyclic learning rate. It takes the learning rate that
    is set to the optimizer in the beginning of the epoch and
    in the end of the epoch the learning rate is set to the same value as it was
    in the beginning.
    '''
    def __init__(self, period_f):
        self.period_f = period_f

    def on_epoch_begin(self):
        self.lrs = []
        for optimizer in self.trainer._optimizers:
            self.lrs.append([])
            for index in range(len(optimizer.optimizer.param_groups)):
                self.lrs[-1].append(
                    optimizer.optimizer.param_groups[index]['lr'])

    def on_batch_begin(self):
        # Here we should compute the learning rate based on a progress in an
        # epoch.
        for optim_index in range(len(self.trainer._optimizers)):
            for group_index in range(len(self.trainer._optimizers[optim_index].optimizer.param_groups)):
                self.trainer._optimizers[optim_index].optimizer.param_groups[group_index]['lr'] = (
                    self.lrs[optim_index][group_index] * self.period_f(self.trainer._progress))
       # lr = self.trainer.group_params['lr']

    def on_epoch_end(self):
        for optim_index in range(len(self.lrs)):
            for group_index in range(len(self.trainer._optimizers[optim_index].optimizer.param_groups)):
                self.trainer._optimizers[optim_index].optimizer.param_groups[group_index]['lr'] = (
                    self.lrs[optim_index][group_index])


class LearningRateScheduler(Callback):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_batch_begin(self):
        pass
