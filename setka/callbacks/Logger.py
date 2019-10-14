from .Callback import Callback

import os
import datetime
import sys
import zipfile
import skimage.io
import scipy.io.wavfile

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
                 name='checkpoint',
                 log_dir='./',
                 ignore_list=('checkpoints', 'logs', 'predictions', 'runs')):

        # '''
        # Constructor.
        #
        # Args:
        #     f (callable): processing function to be used during prediction
        #         process.
        #
        #     write_flag (bool): if False -- the visualization is not performed.
        #
        #     name (str): name of the experiment
        # '''

        self.f = f
        self.write_flag = write_flag
        self.name = name

        self.root_path = os.path.join(
            self.log_dir,
            'logs',
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
            if (file not in ignore_list and
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
        if self.trainer._mode == 'train':
            with open(os.path.join(self.root_path, 'metrics.txt'), 'a+') as fout:
                if hasattr(self.trainer, '_metrics'):
                    fout.write(
                        str(self.trainer._epoch) + '\t' +
                        str(self.trainer._metrics) + '\n')


    def save_image(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.png')
        if len(content.shape) == 3:
            content = content.swapaxes(0, 2).swapaxes(0, 1)
        if not os.path.exists('/'.join(fname.split('/')[:-1])):
            os.makedirs('/'.join(fname.split('/')[:-1]))
        skimage.io.imsave(
            fname,
            content)

    def save_text(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.txt')
        if not os.path.exists('/'.join(fname.split('/')[:-1])):
            os.makedirs('/'.join(fname.split('/')[:-1]))
        with open(fname, 'w+') as fout:
            fout.write(content)

    def save_audio(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.wav')
        if not os.path.exists('/'.join(fname.split('/')[:-1])):
            os.makedirs('/'.join(fname.split('/')[:-1]))
        scipy.io.wavfile.write(
            fname,
            44100,
            content)

    def save_figure(self, name, content, epoch):
        fname = os.path.join(self.root_path, name + '_' + str(epoch) + '.png')
        if not os.path.exists('/'.join(fname.split('/')[:-1])):
            os.makedirs('/'.join(fname.split('/')[:-1]))
        content.savefig(fname)


    @staticmethod
    def get_one(input, item_index):
        if isinstance(input, (list, tuple)):
            one = []
            for list_index in range(len(input)):
                one.append(input[list_index][item_index])
            return one

        elif isinstance(input, dict):
            one = {}
            for key, value in input.items():
                one[key] = value
            return one

        else:
            one = input[item_index]
            return one


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

        if self.trainer._mode == 'train':
            with open(os.path.join(self.root_path, 'loss.txt'), 'a+') as fout:
                fout.write(str(self.trainer._epoch) + '\t' +
                           str(self.trainer._loss.detach().cpu().item()) + '\n')

        if self.trainer._mode == 'test' and self.write_flag and (self.f is not None):
            for index in range(len(self.trainer._ids)):

                one_input = self.get_one(self.trainer._input, index)
                one_output = self.get_one(self.trainer._output, index)

                res = self.f(one_input, one_output)
                id = self.trainer._ids[index]

                self.show(res, id)

    def on_epoch_end(self):
        line = '  '.join([str(k) + ': ' + str(v) for k, v in self.trainer.status.items()])
        with open(os.path.join(self.root_path, 'log.txt'), 'a+') as fout:
            fout.write(line + '\n')
