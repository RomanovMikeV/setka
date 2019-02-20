import os
import shutil
import torch
import tensorboardX
import numpy
import datetime
import sys
import shutil
import zipfile
import scipy.io.wavfile
import skimage

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


class SaveResult(Callback):
    '''
    Callback for saving predictions of the model. The results are
    stored in a specified directory. Batches are processed with the specified
    function ```f```. The directory is flushed during the __init__ and the
    result is saved when the batch is finished (on_batch_end).
    '''
    def __init__(self, processing_f=None, mode='predicting', dir='predictions'):
        '''
        Constructor

        Args:
            f (function): function to process the predictions.

            mode (string): mode of the model when the predictions should be
            saved.

            dir (string): directory where the results should be stored.
        '''
        self.f = processing_f
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
            (self.trainer._mode == 'validation' and self.shuffle_valid)):

            self.trainer._ds_wrapper.shuffle()


class ComputeMetrics(Callback):
    '''
    Callbacks for metric computation.
    '''
    def __init__(self,
                 metrics=None,
                 divide_first=None,
                 steps_to_compute=1):

        '''
        Constructor.
        metrics (dict): dictionary with metrics to compute.
        steps_to_compute (int): how many steps to perform before metrics
            computation.
        '''

        self.steps_to_compute = steps_to_compute
        self.metrics = metrics
        self.names = []

        if self.metrics is None:
            def loss(inp, targ):
                return self.trainer._criterion(inp, targ), 1.0
            self.names = ['loss']
            self.metrics = [loss]

        else:
            for index in range(len(self.metrics)):
                self.names.append(self.metrics[index].__name__)

        if divide_first is None:
            self.divide_first = [True] * len(self.metrics)
        else:
            self.divide_first = divide_first

        self.steps = 0


    def reset(self):
        self.enumerators = []
        self.denominators = []

        for index in self.metrics:
            self.enumerators.append(None)
            self.denominators.append(None)


    def on_epoch_begin(self):
        # clear all cache
        self.reset()

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

                for index in range(len(self.metrics)):
                    enumerators, denominators = self.metrics[index](self.outputs, self.targets)
                    enumerators = numpy.array(enumerators)
                    denominators = numpy.array(denominators)

                    if self.enumerators[index] is None:
                        self.enumerators[index] = enumerators
                        self.denominators[index] = denominators
                    else:
                        self.enumerators[index] += enumerators
                        self.denominators[index] += denominators

                self.avg_values = {}
                for index in range(len(self.enumerators)):
                    if self.divide_first[index]:
                        self.avg_values[self.names[index]] = (
                            (self.enumerators[index] /
                            (self.denominators[index] + 1.0e-12)).mean())
                    else:
                        self.avg_values[self.names[index]] = (
                            self.enumerators[index].sum() /
                            (self.denominators[index].sum() + 1.0e-12))


                self.outputs.clear()
                self.targets.clear()
                self.steps = 0

            self.trainer._line += " ".join(
               ["{}: {:.2e}".format(x, self.avg_values[x])
                for x in self.avg_values])


    def on_epoch_end(self):
        # finilize metrics

        self.outputs.clear()
        self.targets.clear()

        if self.trainer._mode == 'validating':
            if not hasattr(self.trainer, '_metrics'):
                self.trainer._metrics = {}
            self.trainer._metrics[self.trainer._subset] = self.avg_values

        self.avg_values = {}


class MakeCheckpoints(Callback):
    '''
    Callback to make checkpoints. This callback takes into account the
    ```metric```. If ```max_metric``` is ```True``` -- the higher the
    monitored metric is the better the model is, if not -- the lower the
    monitored metric is the better the model is. Checkpoint name is specified
    with ```name```, the best model is saved with ```_best``` postfix.
    '''
    def __init__(self,
                 metric,
                 subset='valid',
                 max_mode=False,
                 name='checkpoint'):
        self.best_metric = None
        self.metric = metric
        self.max_mode = max_mode
        self.name = name
        self.subset = subset


    def on_epoch_end(self):

        is_best = False

        if self.trainer._mode == 'validating' and self.trainer._subset == self.subset:
            if self.best_metric is None:
                self.best_metric = self.trainer._metrics[self.subset][self.metric]
                is_best = True

            if ((self.best_metric < self.trainer._metrics[self.subset][self.metric] and
                 self.max_mode) or
                (self.best_metric > self.trainer._metrics[self.subset][self.metric] and
                 not self.max_mode)):

                 self.best_metric = self.trainer._metrics[self.subset][self.metric]
                 is_best = True

            self.trainer.save('checkpoints/' + self.name + '.pth.tar')
            if is_best:
                self.trainer.save('checkpoints/' + self.name + '_best.pth.tar')


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

    def on_epoch_begin(self):
        if self.trainer._mode == 'training' and self.write_flag:
            if hasattr(self.trainer, '_metrics'):

                for subset in self.trainer._metrics:
                    data = {}
                    for metric_name in self.trainer._metrics[subset]:

                        data[subset] = (
                            self.trainer._metrics[subset][metric_name])

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


class Logger(Callback):
    '''
    This callback saves all the information that is important for experiment
    reproduction.
    '''
    def __init__(self,
                 processing_f=None,
                 write_flag=True,
                 name='checkpoint',
                 metric='loss',
                 subset='valid',
                 max_mode=False):

        self.f = processing_f
        self.write_flag = write_flag
        self.name = name
        self.best_metric = None
        self.subset = subset
        self.metric = metric
        self.max_mode = max_mode

    def on_init(self):
        if not os.path.exists('./logs'):
            os.mkdir('logs')

        root_path = os.path.join('logs', self.name)
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        self.root_path = os.path.join(root_path, str(datetime.datetime.now()))

        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)

        self.paths = {'root': self.root_path}

        with open(os.path.join(self.root_path, 'bash_command.txt'), 'w+') as fout:
            fout.write(' '.join(sys.argv))

        command_root_dir = sys.argv[0].split('/')
        if len(command_root_dir) <= 1:
            command_root_dir = '.'
        else:
            command_root_dir = '/'.join(command_root_dir[:-1])

        zip = zipfile.ZipFile(os.path.join(self.root_path, 'snapshot.zip'), 'w')

        for file in os.listdir(command_root_dir):
            if (file != 'checkpoints' and
                file != 'logs' and
                file != 'predictions' and
                file != 'runs' and
                file[0] != '.'):

                zip.write(os.path.join(command_root_dir, file))


    def on_epoch_begin(self):
        if self.trainer._mode == 'training':
            if hasattr(self.trainer, '_metrics'):
                print(self.trainer._metrics)
            with open(os.path.join(self.root_path, 'metrics.txt'), 'a+') as fout:
                if hasattr(self.trainer, '_metrics'):
                    fout.write(
                        str(self.trainer._epoch) + '\t' +
                        str(self.trainer._metrics) + '\n')

            last_checkpoint_name = os.path.join(self.root_path, 'checkpoints', 'last.pth.tar')
            best_checkpoint_name = os.path.join(self.root_path, 'checkpoints', 'best.pth.tar')

            self.create_dir(last_checkpoint_name)
            self.trainer.save(last_checkpoint_name)

            if hasattr(self.trainer, '_metrics'):
                is_best = False
                if self.best_metric is None:
                    self.best_metric = self.trainer._metrics[self.subset][self.metric]
                    is_best = True

                if ((self.best_metric < self.trainer._metrics[self.subset][self.metric] and
                     self.max_mode) or
                    (self.best_metric > self.trainer._metrics[self.subset][self.metric] and
                     not self.max_mode)):

                     self.best_metric = self.trainer._metrics[self.subset][self.metric]
                     is_best = True

                if is_best:
                    self.trainer.save(best_checkpoint_name)



    @staticmethod
    def create_dir(fname):
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def save_image(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.png')
        if len(content.shape) == 3:
            content = content.swapaxes(0, 2).swapaxes(0, 1)
        self.create_dir(fname)
        skimage.io.imsave(
            fname,
            content)

    def save_text(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.txt')
        self.create_dir(fname)
        with open(fname, 'w+') as fout:
            fout.write(content)

    def save_audio(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.wav')
        self.create_dir(fname)
        scipy.io.wavfile.write(
            fname,
            44100,
            content)

    def save_figure(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.png')
        self.create_dir(fname)
        content.savefig(fname)


    def show(self, to_show, id):
        type_writers = {
            'images': self.save_image,
            'texts': self.save_text,
            'audios': self.save_audio,
            'figures': self.save_figure}

        for type in type_writers:
            if type in to_show:
                for desc in to_show[type]:
                    type_writers[type](type + '/' + str(id) + '/' + desc,
                        to_show[type][desc], str(self.trainer._epoch))

    def on_batch_end(self):

        if self.trainer._mode == 'training':
            with open(os.path.join(self.root_path, 'loss.txt'), 'a+') as fout:
                fout.write(str(self.trainer._epoch) + '\t' +
                           str(self.trainer._loss.detach().cpu().item()) + '\n')

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

    def on_epoch_end(self):
        line = 'MODE: ' + self.trainer._mode + '\tSUBSET: ' + self.trainer._subset + '\tINFO: ' + self.trainer._line
        with open(os.path.join(self.root_path, 'log.txt'), 'a+') as fout:
            fout.write(line + '\n')




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
    def __init__(self,
                 metric,
                 subset='valid',
                 cooldown=5,
                 limit=5,
                 max_mode=False):

        self.cooldown = cooldown
        self.limit = limit

        self.since_last = 0
        self.since_best = 0

        self.best_metric = None
        self.subset = subset
        self.metric = metric
        self.max_mode = max_mode

        self.optimizer_index = 1


    def on_init(self):
        if hasattr(self.trainer, '_lr_reduce'):
            self.trainer._lr_reduce = False
        self.complete = False


    def on_epoch_end(self):
        if (self.trainer._mode == 'validating' and
            self.trainer._subset == self.subset):

            if self.best_metric is None:
                self.best_metric = (
                    self.trainer._metrics[self.subset][self.metric])
                self.since_best = 0

            else:
                new_metric = self.trainer._metrics[self.subset][self.metric]

                if ((new_metric > self.best_metric and self.max_mode) or
                    (new_metric < self.best_metric and not self.max_mode)):

                    self.best_metric = new_metric
                    self.since_best = 0


            if self.since_last >= self.cooldown and self.since_best >= self.limit and not self.complete:
                self.trainer._line += ("UNFREEZING (optimizer " + str(self.optimizer_index) + ")")
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
    def __init__(self,
                 metric,
                 subset='valid',
                 cooldown=5,
                 limit=5,
                 factor=0.5,
                 max_mode=False):

        self.cooldown = cooldown
        self.limit = limit
        self.factor = factor

        self.since_last = 0
        self.since_best = 0

        self.best_metric = None
        self.subset = subset
        self.metric = metric
        self.max_mode = max_mode

        self.optimizer_index = 1


    def on_init(self):
        self.trainer._lr_reduce=True


    def on_epoch_end(self):
        if (self.trainer._mode == 'validating' and
            self.trainer._subset == self.subset and
            self.trainer._lr_reduce):

            if self.best_metric is None:
                self.best_metric = self.trainer._metrics[self.subset][self.metric]
                self.since_best = 0

            else:
                new_metric = self.trainer._metrics[self.subset][self.metric]

                if ((new_metric > self.best_metric and self.max_mode) or
                    (new_metric < self.best_metric and not self.max_mode)):

                    self.best_metric = new_metric
                    self.since_best = 0

            if self.since_last >= self.cooldown and self.since_best >= self.limit:
                self.trainer._line += 'REDUCING lr'
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
        for optim_index in range(len(self.trainer._optimizers)):
            for group_index in range(len(self.trainer._optimizers[optim_index].optimizer.param_groups)):
                self.trainer._optimizers[optim_index].optimizer.param_groups[group_index]['lr'] = (
                    self.lrs[optim_index][group_index] * self.period_f(self.trainer._progress))

    def on_epoch_end(self):
        for optim_index in range(len(self.lrs)):
            for group_index in range(len(self.trainer._optimizers[optim_index].optimizer.param_groups)):
                self.trainer._optimizers[optim_index].optimizer.param_groups[group_index]['lr'] = (
                    self.lrs[optim_index][group_index])


# class SwitchBetweenOptimizers(Callback):
#     '''
#     The callback allows to switch between the optimizers depending on the
#     iteration index.
#     '''
#     def __init__(self, optimizer_steps):
#         self.optimizer_steps = optimizer_steps
#         self.iteration = 0
#         self.cycle_len = sum(optimizer_steps)
#
#         self.optimizer_steps = [0]
#         for index in range(0, len(self.optimizer_steps) - 1):
#             self.optimizer_steps.append(
#                 self.optimizer_steps[index] + optimzer_steps[index])
#
#     def on_batch_begin(self):
#         if self.iteration >= self.cycle_len:
#             self.iteration = 0
#
#         for index in range(len(self.trainer._optimizers)):
#             self.trainer._optimizers[index].is_active = False
#
#         optimizer_index = 0
#         while self.iteration > self.optimizer_steps[optimizer_index]:
#             self.trainer._optimizers[optimizer_index].is_active = False
#             optimizer_index += 1
