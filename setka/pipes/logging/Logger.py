import fnmatch
import os
import datetime
import sys
from subprocess import Popen, PIPE
import zipfile

import skimage.io
import scipy.io.wavfile

import torch
from setka.pipes.Pipe import Pipe


def get_process_output(command):
    if not isinstance(command, (list, tuple)):
        command = command.split(' ')
    process = Popen(command, stdout=PIPE, shell=True)
    output, err = process.communicate()
    exit_code = process.wait()
    return exit_code, output.decode()


def check_list(path, masks):
    for mask in masks:
        if fnmatch.fnmatch(path, mask):
            return False
    return True


def collect_snapshot_list(command_root_dir, ignore_list, full_path=True):
    results = []
    for file in os.listdir(command_root_dir):
        if check_list(file, ignore_list) and file[0] != '.':
            if os.path.isdir(os.path.join(command_root_dir, file)):
                for root, _, files in os.walk(os.path.join(command_root_dir, file)):
                    for sub_file in files:
                        if check_list(os.path.relpath(os.path.join(root, sub_file), command_root_dir), ignore_list):
                            results.append(os.path.join(root, sub_file))
            else:
                results.append(os.path.join(command_root_dir, file))

    if full_path:
        return results
    else:
        return [os.path.relpath(f, command_root_dir) for f in results]


class Logger(Pipe):
    """
    This pipe saves all the information about training process of the
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
    ```predictions``` and ```runs``` directories). It creates the
    directories ```./logs/<timestamp>/checkpoints```
    and ```./logs/<timestamp>/predictions``` and saves paths to these directories
    to the ```trainer._checkpoints_dir``` and ```trainer._predictions_dir```.

    The following information is logged during the training process:

    * loss value is saved after each batch in loss.txt file in the checkpoint
        directory

    * metrics values are stored in metrics.txt file (if metrics are specified)
        in the checkpoint directory

    * images, audios, figures, texts are stored in the corresponding directories
        in the checkpoint directory if the processing function ```f``` is
        specified.

    Args:
        f (callable): function for test samples visualization. If set to None, test will not be visualized.
        name (str): name of the experiment (will be used as a a name of the log folder)
        log_dir (str): path to the directory, where the logs are stored.
        ignore_list (list of str): folders to not to include to the snapshot.
    """
    def __init__(self, f=None, name='experiment', log_dir='runs', make_snapshot=True,
                 ignore_list=[
                     '*.zip*',
                     '*.pth*',
                     '*__pycache__*',
                     '*.ipynb_checkpoints*',
                     '*.jpg',
                     '*.jpeg',
                     '*.png',
                     '*.wav',
                     '*.mp4',
                     '*.bmp',
                     '*.mov',
                     '*.mp3',
                     '*.csv',
                     '*.txt',
                     '*.json',
                     '*.tar.gz',
                     '*.zip',
                     '*.gzip',
                     '*.7z',
                     '*.ipynb',
                     '*.coredump',
                     '*data*',
                     'logs/*',
                     'runs/*'],
                 full_snapshot_path=False, collect_environment=True):

        super(Logger, self).__init__()
        self.root_path = None
        self.f = f
        self.name = name
        self.log_dir = log_dir
        self.make_snapshot = make_snapshot
        self.full_snapshot_path = full_snapshot_path
        self.collect_environment = collect_environment
        self.ignore_list = ignore_list

    def on_init(self):
        self.root_path = os.path.join(self.log_dir, self.name,
                                      str(self.trainer.creation_time).replace(' ', '_').replace(':', '-'))

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)

        with open(os.path.join(self.root_path, 'bash_command.txt'), 'w+') as fout:
            fout.write(' '.join(sys.argv))

        if self.make_snapshot:
            command_root_dir = os.getcwd()
            with zipfile.ZipFile(os.path.join(self.root_path, 'snapshot.zip'), 'w') as snapshot:
                snapshot_list = collect_snapshot_list(command_root_dir, self.ignore_list, self.full_snapshot_path)
                for file in snapshot_list:
                    snapshot.write(file)
            print('Made snapshot of size {:.2f} MB'.format(
                os.path.getsize(os.path.join(self.root_path, 'snapshot.zip')) / (1024 * 1024)))

        if self.collect_environment:
            is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
            if is_conda:
                print('Collecting environment using conda...', end=' ')
                code, res = get_process_output('conda env export')
            else:
                print('Collecting environment using pip...', end=' ')
                code, res = get_process_output('pip list')

            print('FAILED' if code != 0 else 'OK')
            with open(os.path.join(self.root_path, 'environment.txt'), 'w') as f:
                f.write(res)

        predictions_dir = os.path.join(self.root_path, 'predictions')
        
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)

    def before_epoch(self):
        """
        Dumps metrics to the log file.
        """
        if self.trainer._mode == 'train':
            with open(os.path.join(self.root_path, 'metrics.txt'), 'a+') as fout:
                if hasattr(self.trainer, '_metrics'):
                    fout.write(str(self.trainer._epoch - 1) + '\t' + str(self.trainer._metrics) + '\n')

    @staticmethod
    def make_dirs(fname):
        dir_name = os.sep.join(fname.split(os.sep)[:-1])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def save_image(self, name, content, epoch, ext='png'):
        fname = os.path.join(self.root_path, str(epoch) + '_' + name)
        if len(fname.split(os.sep)[-1].split('.')) == 1:
            fname = fname + '.' + ext
        if len(content.shape) == 3:
            content = content.swapaxes(0, 2).swapaxes(0, 1)
        self.make_dirs(fname)
        skimage.io.imsave(fname, content)

    def save_text(self, name, content, epoch, ext='txt'):
        fname = os.path.join(self.root_path, str(epoch) + '_' + name)
        if len(fname.split(os.sep)[-1].split('.')) == 1:
            fname = fname + '.' + ext
        self.make_dirs(fname)
        with open(fname, 'w+') as fout:
            fout.write(content)

    def save_audio(self, name, content, epoch, ext='wav'):
        fname = os.path.join(self.root_path, str(epoch) + '_' + name)
        if len(fname.split(os.sep)[-1].split('.')) == 1:
            fname = fname + '.' + ext
        self.make_dirs(fname)
        scipy.io.wavfile.write(fname, 44100, content)

    def save_figure(self, name, content, epoch, ext='png'):
        fname = os.path.join(self.root_path, str(epoch) + '_' + name)
        if len(fname.split(os.sep)[-1].split('.')) == 1:
            fname = fname + '.' + ext
        self.make_dirs(fname)
        content.savefig(fname)

    def save_file(self, name, content, epoch, ext='bin'):
        fname = os.path.join(self.root_path, str(epoch) + '_' + name)
        if len(fname.split(os.sep)[-1].split('.')) == 1:
            fname = fname + '.' + ext

        self.make_dirs(fname)

        with open(fname, 'wb+') as fout:
            # print(content)
            content.seek(0)
            fout.write(content.read())

    def show(self, to_show, id):
        type_writers = {
            'images': self.save_image,
            'texts': self.save_text,
            'audios': self.save_audio,
            'figures': self.save_figure,
            'files': self.save_file
        }

        for type in type_writers:
            if type in to_show:
                for desc in to_show[type]:
                    kwargs = {
                        'name': os.path.join(type, str(id), desc),
                        'content': to_show[type][desc],
                        'epoch': str(self.trainer._epoch)
                    }

                    type_writers[type](**kwargs)

    def after_batch(self):
        """
        Writes the loss to the loss log (in case of train mode).
        Also performs visualisation in case of test mode.
        """
        if self.trainer._mode == 'train':
            with open(os.path.join(self.root_path, 'loss.txt'), 'a+') as fout:
                fout.write(str(self.trainer._epoch) + '\t' +
                           str(self.trainer._loss.detach().cpu().item()) + '\n')

        if self.trainer._mode == 'test' and (self.f is not None):
            for index in range(len(self.trainer._ids)):
                one_input = self.trainer.collection_op.split_index(self.trainer._input, index)[0]
                one_output = self.trainer.collection_op.split_index(self.trainer._output, index)[0]
                res = self.f(one_input, one_output)
                id = self.trainer._ids[index]
                self.show(res, id)

    # @staticmethod
    # def format(v):
    #     if isinstance(v, torch.Tensor):
    #         return str(v.item())
    #     return str(v)

    def after_epoch(self):
        """
        Writes the trainer status to the log file.
        """
        line = '  '.join([str(k) + ': ' + str(v) for k, v in self.trainer.status.items()])
        with open(os.path.join(self.root_path, 'log.txt'), 'a+') as fout:
            fout.write(line + '\n')
