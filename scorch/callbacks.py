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
import types
import pickle
import copy

from . import internal

class Callback():
    '''
    Callback basic class.

    Callback has the following methods:

    * __init__(self) -- constructor

    * on_train_begin(self) -- method that is executed when training starts
        (in the beginning of ```trainer.train```)

    * on_train_end(self) -- method that is executed when training starts
        (in the end of ```trainer.train```)

    * on_epoch_begin(self) -- method that is executed when the epoch starts
        (in the beginning of ```train_one_epoch```, ```validate_one_epoch```,
        ```predict```)

    * on_epoch_end(self) -- method that is executed when the epoch starts
        (in the end of ```train_one_epoch```, ```validate_one_epoch```,
        ```predict```)

    * on_batch_begin(self) -- method that is executed when the batch starts
        (in the beginning of training cycle body in ```train_one_epoch```,
        ```validate_one_epoch```, ```predict```)

    * on_batch_end(self) -- method that is executed when the batch starts
        (in the end of training cycle body in ```train_one_epoch```,
        ```validate_one_epoch```, ```predict```)

    * set_trainer(self, trainer) -- method that links the trainer to the callback.
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
    stored in a directory ```predictions``` and in directory specified in
    ```trainer._predictions_dir```. Batches are processed with the
    specified function ```f```. The directory is flushed during the
    ```__init__``` and the result is saved when the batch is
    finished (when on_batch_end is triggered).
    '''
    def __init__(self, f=None):
        '''
        Constructor

        Args:
            f (function): function to process the
                predictions.

            dir (string): directory where the results should be
                stored.
        '''
        self.f = f
        self.index = 0

        if not os.path.exists('./predictions'):
            os.makedirs('./predictions')


    def on_batch_end(self):
        if self.trainer._mode == 'predicting':
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
            torch.save(res, os.path.join('./predictions', str(self.index) + '.pth.tar'))

            if hasattr(self.trainer, "_predictions_dir"):
                torch.save(res, os.path.join(self.trainer._predictions_dir,
                    str(self.index) + 'pth.tar'))

            self.index += 1


class ShuffleDataset(Callback):
    '''
    Callback to shuffle datasets before the epoch starts
    (on_epoch_begin). Test set is never shuffled.
    '''
    def __init__(self, shuffle_valid=False):
        '''
        Constructor.

        Args:
            shuffle_valid (bool): shuffle validation dataset too if
                True. If False only the training dataset will be
                shuffled. Note that the test set is never shuffled.
        '''
        self.shuffle_valid = shuffle_valid

    def on_epoch_begin(self):
        if (self.trainer._mode == 'training' or
            (self.trainer._mode == 'validation' and
             self.shuffle_valid)):

            self.trainer._ds_wrapper.shuffle()


class ComputeMetrics(Callback):
    '''
    This callback computes metrics when the validation is
    performed. The metrics are updated on batch end. The parameter
    ```steps_to_compute``` specifies on how many batches the
    metrics are computed. When the epoch ends -- final metrics
    are computed. The history is flushed when the
    epoch starts.

    Note: list of metrics should contain callables. Each of the
        callables my return either one value or two values. If two
        values are returned (or two numpy arrays of the same shape) --
        the first value is treated as enumerator(s), the second is treated as a
        denominator(s).
        When the metric is requested -- the new enumerator(s) and
        denomirator(s) are computed. Overall enumerators and denominators
        are updated (new values added to the accumulated ones).
        After that there are two options
        for computing average for each of the metrics in arrays.
        First option is to first sum all the accumulated enumerators
        and denominators and then to divide one by another
        (case of ```divide_first``` flag for the metric
        is set to False). The second option is to first divide
        each of the enumerators by its denominator and then average
        (case of ```divide_first``` for the metric is set to True).
    '''
    def __init__(self,
                 metrics=None,
                 divide_first=None,
                 steps_to_compute=1):

        '''
        Constructor.

        Args:
            metrics (list): list of callables for metrics
                computation. The callable should return two values:
                enumerator and denominator.

            steps_to_compute (int): how many steps to perform
                before metrics update.
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
                    res = self.metrics[index](self.outputs, self.targets)
                    if isinstance(res, (list, tuple)):
                        if len(res) == 2:
                            enumerators = numpy.array(res[0])
                            denominators = numpy.array(res[1])
                    else:
                        enumerators = numpy.array(res)
                        denominators = numpy.ones(enumerators.shape)

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
    This callback makes checkpoints during an experiment. Two checkpoints
    are stored: the best one (with "_best" postfix) (created only in case the
    needed metrics has already been computed) and the last one
    (with "_last" postfix). You should specify the
    metrics that will be monitored to select the best model.

    The checkpoint is computed when epoch starts (on_epoch_begin).

    The checkpoints are saved in the directory ```checkpoints``` and
    in a directory specified in ```trainer._checkpoints_dir```. If the
    ```checkpoints``` directory does not exist -- it will be created.
    '''
    def __init__(self,
                 metric,
                 subset='valid',
                 max_mode=False,
                 name='checkpoint'):

        '''
        Constructor.

        Args:
            metric (str): name of the metric to monitor

            subset (hashable): name of the subset on which the metric will be
                monitored

            max_mode (bool): if True, the higher the metric -- the better the
                model.

            name (str): name of the checkpoint file.
        '''
        self.best_metric = None
        self.metric = metric
        self.max_mode = max_mode
        self.name = name
        self.subset = subset

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')


    # @staticmethod
    # def create_dir(fname):
    #     dirname = os.path.dirname(fname)
    #     if not os.path.exists(dirname):
    #         os.makedirs(dirname)


    def on_epoch_begin(self):

        is_best = False

        if self.trainer._mode == 'training':
            if hasattr(self.trainer, '_metrics'):
                if self.subset in self.trainer._metrics:
                    if self.metric in self.trainer._metrics[self.subset]:
                        if self.best_metric is None:
                            self.best_metric = (
                                self.trainer._metrics[self.subset][self.metric])
                            is_best = True

                        if ((self.best_metric < self.trainer._metrics[self.subset][self.metric] and
                             self.max_mode) or
                            (self.best_metric > self.trainer._metrics[self.subset][self.metric] and
                             not self.max_mode)):

                             self.best_metric = self.trainer._metrics[self.subset][self.metric]
                             is_best = True


            self.trainer.save(os.path.join(
                './checkpoints',
                self.name + '_latest.pth.tar'))

            if is_best:
                self.trainer.save(os.path.join(
                    './checkpoints',
                    self.name + '_best.pth.tar'))

            if hasattr(self.trainer, '_checkpoints_dir'):
                self.trainer.save(os.path.join(
                    self.trainer._checkpoints_dir,
                    self.name + '_latest.pth.tar'))

                if is_best:
                    self.trainer.save(os.path.join(
                        self.trainer._checkpoints_dir,
                        self.name + '_best.pth.tar'))


class WriteToTensorboard(Callback):
    '''
    Callback to write the metrics to the TensorboardX. When the epoch starts
    (on_epoch_begin), it uploads the computer metrics to the TnsorboardX.

    It also writes the predictions to the tensorboard when the ```predict```
    method of the Trainer is called and visualization function is specified.

    Visaulization function (passed as ```f``` to the construtor)
    takes as inputs: one input, target and output per sample and returns the
    dictionary with the outputs for visualization. This dictionary may contain
    the following keys:

    {
        "images": dict with numpy images,
        "texts": dict with texts,
        "audios": dict with numpy audios,
        "figures": dict with matplotlib figures,
        "graphs": dict with onnx graphs,
        "embeddings": dict with embeddings
    }
    Each of the dicts should have the following structure:
    {image_name: image} for images. For example, the following syntax will work:

    ```
    {"images": {"input": input_fig,
                "output": ouput_fig,
                "target": target_fig}}
    ```

    '''
    def __init__(self,
                 f=None,
                 write_flag=True,
                 name='checkpoint'):
        self.tb_writer = tensorboardX.SummaryWriter()
        self.f = f
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
    This callback saves all the information about training process of the
    model. It is important the training process understanding and for the
    experiment reproducibility.

    The information is stored in the directory ```./logs/time```, where
    time is a string representation of the timestamp when the experiment has
    started.

    During the initialization, the Logger creates all the necessary directories
    for storing information. It also saves the bash command that has triggered
    the Trainer creation in the text file called "bash_command.txt", it saves
    all the contents of the directory from where the command was called in the
    archive "snapshot.zip" (except for ```checkpoints```, ```logs```,
    ```predictions``` and ```runs``` directories). It crestes the
    directories ```./logs/<timestamp>/checkpoints```
    and ```./logs/<timestamp>/predictions``` and sves paths to these directories
    to the ```trainer._checkpoints_dir``` and ```trainer._predictions_dir```.

    The following information is logged during the training process:

    * loss value is saved after each batch in loss.txt file in the checkpoint
        directory

    * metrics values are stored in metrics.txt file (if metrics are specified)
        in the checkpoint directory

    * images, audios, figures, texts are stored in the corresponding directories
        in the checkpoint directory if the processing function ```f``` is
        specified.

    '''
    def __init__(self,
                 f=None,
                 write_flag=True,
                 name='checkpoint'):

        '''
        Constructor.

        Args:
            f (callable): processing function to be used during prediction
                process.

            write_flag (bool): if False -- the visualization is not performed.

            name (str): name of the experiment
        '''

        self.f = f
        self.write_flag = write_flag
        self.name = name

        self.root_path = os.path.join(
            './logs',
            self.name,
            str(datetime.datetime.now()).replace('\s', '-')
        )

        os.makedirs(self.root_path)

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


    def on_init(self):
        checkpoints_dir = os.path.join(self.root_path, 'checkpoints')
        predictions_dir = os.path.join(self.root_path, 'predictions')

        os.makedirs(checkpoints_dir)
        os.makedirs(predictions_dir)

        self.trainer._checkpoints_dir = checkpoints_dir
        self.trainer._predictions_dir = predictions_dir


    def on_epoch_begin(self):
        if self.trainer._mode == 'training':
            if hasattr(self.trainer, '_metrics'):
                print(self.trainer._metrics)
            with open(os.path.join(self.root_path, 'metrics.txt'), 'a+') as fout:
                if hasattr(self.trainer, '_metrics'):
                    fout.write(
                        str(self.trainer._epoch) + '\t' +
                        str(self.trainer._metrics) + '\n')


    def save_image(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.png')
        if len(content.shape) == 3:
            content = content.swapaxes(0, 2).swapaxes(0, 1)
        os.makedirs('/'.join(fname.split('/')[:-1]))
        skimage.io.imsave(
            fname,
            content)

    def save_text(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.txt')
        os.makedirs('/'.join(fname.split('/')[:-1]))
        with open(fname, 'w+') as fout:
            fout.write(content)

    def save_audio(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.wav')
        os.makedirs('/'.join(fname.split('/')[:-1]))
        scipy.io.wavfile.write(
            fname,
            44100,
            content)

    def save_figure(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.png')
        os.makedirs('/'.join(fname.split('/')[:-1]))
        content.savefig(fname)
        # make pickle dump


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
    it. It temporarily turns off LR reducing and, instead of it, performs
    unfreezing.
    '''
    def __init__(self,
                 metric,
                 subset='valid',
                 cooldown=5,
                 limit=5,
                 max_mode=False):

        '''
        Constructor.

        Args:
            metric (str) : name of the metric to monitor.

            cooldown (int) : minimal amount of epochs between two learning rate changes.

            limit (int) : amount of epochs since the last improvement of maximum of
                the monitored metric before the learning rate change.

            max_mode (bool) : if True then the higher is the metric the better.
                Otherwise the lower is the metric the better.
        '''

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
    This callback performs Learning Rate reducing when the plateau
    is reached.
    '''
    def __init__(self,
                 metric,
                 subset='valid',
                 cooldown=5,
                 limit=5,
                 factor=0.5,
                 max_mode=False):

        '''
        Constructor.

        Args:
            metric (str) : name of the metric to monitor.

            cooldown (int) : minimal amount of epochs between two learning rate changes.

            limit (int) : amount of epochs since the last improvement of maximum of
                the monitored metric before the learning rate change.

            max_mode (bool) : if True then the higher is the metric the better.
                Otherwise the lower is the metric the better.
        '''

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
    in the beginning. During the epoch the value of Learning Rate is attenuated
    by the value of the ```cycle``` function. If you are using per-epoch
    scheduling (```ReduceLROnPlateau``` for instance) -- you will need to
    list ```CyclicLR``` callback in the list of callbacks AFTER other scheduling
    callbacks. Otherwise the per-epoch scheduling will not work.

    When epoch starts (on_epoch_begin), the values of the learning rates
    are saved.

    When batch starts (on_batch_begin), the function computes the attenuation
    coefficient for the learning rates saved during epoch start and sets the
    learning rate of all th optimizers to the attenuated value.

    When epoch ends (on_epoch_end), the values of the initial learning rates are
    restored.
    '''

    def __init__(self, cycle):
        '''
        Constructor.

        Args:
            cycle (callable): A function or callable that takes as input one
                variable (float between 0 and 1, fraction of completed steps in
                the epoch) and computes the coefficient for the learning rate
                for the next step.
        '''
        self.cycle = cycle


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
                    self.lrs[optim_index][group_index] * self.cycle(self.trainer._progress))

    def on_epoch_end(self):
        for optim_index in range(len(self.lrs)):
            for group_index in range(len(self.trainer._optimizers[optim_index].optimizer.param_groups)):
                self.trainer._optimizers[optim_index].optimizer.param_groups[group_index]['lr'] = (
                    self.lrs[optim_index][group_index])


class ExponentialWeightAveraging(Callback):
    '''
    This callback performs weight averaging during the training. The callback
    duplicates a model and updates weights of it in the following way:
    $$\tilde{w}_i = \gamma \tilde{w}_i + (1.0 - \gamma) w_i,$$
    where $\tilde{w}$ is a weight of an averaged model, $\gamma$ is a parameter
    of an exponential moving average, $w_i$ is a weight of an optimized model.

    This callbacks does nothing until specified epoch. After this epoch it
    begins tracking of the averaged model. During validation and testing, the
    averaged model is used.
    '''

    def __init__(self,
                 gamma=0.99,
                 epoch_start=10):
        self.gamma = gamma
        self.epoch_start = epoch_start

    def on_epoch_begin(self):
        if (self.trainer._epoch >= self.epoch_start and
            not hasattr(self, 'averaged_model') and
            self.trainer._mode == 'training'):

            self.averaged_model = copy.deepcopy(self.trainer._model)

        if (hasattr(self, 'averaged_model') and (
                self.trainer._mode == 'valid' or
                self.trainer._mode == 'test')):
            self.trainable_model = self.trainer._model
            self.trainer._model = self.averaged_model

    def on_epoch_end(self):
        if (hasattr(self, 'averaged_model') and (
                self.trainer._mode == 'valid' or
                self.trainer._mode == 'test')):

            self.trainer._model = self.trainable_model


    def on_batch_end(self):
        if (hasattr(self, 'averaged_model') and
            self.trainer._mode == 'training'):

            avg_pars = self.averaged_model.parameters()
            trn_pars = self.trainer._model.parameters()

            for avg_par, trn_par in zip(avg_pars, trn_pars):
                avg_par = avg_par * (1.0 - self.gamma) + trn_par * self.gamma



            #avg_par += (
            #    (1.0 - self.gamma) * trn_par)
            #avg_par /= (2.0 - self.gamma)


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
