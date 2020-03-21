import sys
import os

import torch.utils.tensorboard as TB
from setka.pipes.Pipe import Pipe


class TensorBoard(Pipe):
    """
    pipe to write the progress to the TensorBoard. When the epoch starts
    (before_epoch), it uploads computed metrics on previous epoch to the TensorBoard.
    It also writes the predictions to the TensorBoard when the ```predict```
    method of the Trainer is called and visualization function is specified.
    Visualization function (passed as ```f``` to the constructor)
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
    {
        "images": {
            "input": input_fig,
            "output": output_fig,
            "target": target_fig
        }
    }
    ```
    Args:
        f (callable): function to visualize the network results.
        name (str): name of the experiment.
        log_dir (str): path to the directory for "tensorboard --logdir" command.
    """

    def __init__(self, f=None, log_dir='runs', name='experiment_name'):
        super(TensorBoard, self).__init__()
        self.f = f
        self.log_dir = log_dir
        self.name = name
        self.tb_writer = None

    def before_epoch(self):
        """
        Writes scalars (metrics) of the previous epoch.
        """
        self.tb_writer = TB.SummaryWriter(log_dir=os.path.join(self.log_dir, self.name))

        if self.trainer._mode == 'train' and hasattr(self.trainer, '_metrics'):
            for subset in self.trainer._metrics:
                for metric_name in self.trainer._metrics[subset]:
                    self.tb_writer.add_scalar(
                        f'{metric_name}/{subset}',
                        self.trainer._metrics[subset][metric_name],
                        self.trainer._epoch - 1
                    )

    def show(self, to_show, id):
        type_writers = {
            'images': self.tb_writer.add_image,
            'texts': self.tb_writer.add_text,
            'audios': self.tb_writer.add_audio,
            'figures': (lambda x, y, z: self.tb_writer.add_figure(x, y, z)),
            'graphs': self.tb_writer.add_graph,
            'embeddings': self.tb_writer.add_embedding
        }

        for type in type_writers:
            if type in to_show:
                for desc in to_show[type]:
                    type_writers[type](str(id) + '/' + desc, to_show[type][desc], str(self.trainer._epoch))

    def after_batch(self):
        """
        Writes the figures to the tensorboard when the trainer is in the test mode.
        """
        if self.trainer._mode == 'test' and (self.f is not None):
            for index in range(len(self.trainer._ids)):
                one_input = self.trainer.collection_op.split_index(self.trainer._input, index)[0]
                one_output = self.trainer.collection_op.split_index(self.trainer._output, index)[0]

                res = self.f(one_input, one_output)
                id = self.trainer._ids[index]

                self.show(res, id)

        if self.trainer._mode == 'train':
            self.tb_writer.add_scalar('loss/summary', self.trainer._loss.detach().cpu(), self.trainer._iteration)
            if hasattr(self.trainer, '_loss_values') and len(self.trainer._loss_values) > 1:
                for key in self.trainer._loss_values:
                    self.tb_writer.add_scalar(f'loss/{key}', self.trainer._loss_values[key], self.trainer._iteration)

    def after_epoch(self):
        """
        Destroys TensorBoardWriter
        """
        self.tb_writer.close()
        del self.tb_writer
