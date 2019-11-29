from .Pipe import Pipe

import torch.utils.tensorboard as TB
import sys
import os

class TensorBoard(Pipe):
    '''
    pipe to write the progress to the TensorBoard. When the epoch starts
    (before_epoch), it uploads the computer metrics to the TensorBoard.

    It also writes the predictions to the TensorBoard when the ```predict```
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

    Args:
        f (callbale): function to visualize the network results.

        write_flag (bool): if True -- the results of f will be written to the tensorboard

        name (str): name of the experiment.

        log_dir (str): path to the directory, where the logs will be stored.

    '''
    def __init__(self,
                 f=None,
                 write_flag=True,
                 name='checkpoint',
                 log_dir='./',
                 priority={'after_batch': 10}):
        self.log_dir = os.path.join(log_dir, 'runs')
        self.f = f
        self.write_flag = write_flag
        self.name = name

    def before_epoch(self):
        '''
        Initializes the TensorBoardWriter.
        '''
        self.tb_writer = TB.SummaryWriter(log_dir=self.log_dir)
        
        if self.trainer._mode == 'train' and self.write_flag:

            if hasattr(self.trainer, '_metrics'):
                data = {}
                for subset in self.trainer._metrics:
                    for metric_name in self.trainer._metrics[subset]:
                        if metric_name not in data:
                            data[metric_name] = {}
                        data[metric_name][subset] = (
                            self.trainer._metrics[subset][metric_name])


                for metric_name in data:
                    self.tb_writer.add_scalars(
                            self.name + '/' + metric_name,
                            data[metric_name],
                            self.trainer._epoch)
                    
    
    def after_epoch(self):
        '''
        Destroys TensorBoardWriter
        '''

        self.tb_writer.close()
        del self.tb_writer
        

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

    @staticmethod
    def get_one(input, item_index):
        if isinstance(input, (list, tuple)):
            one = []
            for list_index in range(len(input)):
                one.append(input[list_index][item_index])
            return one
        elif isinstance(input, dict):
            one = {}
            for dict_key in input:
                one[dict_key] = input[dict_key][item_index]
            return one
        else:
            one = input[item_index]
            return one


    def after_batch(self):
        '''
        Writes the figures to the tensorboard when the trainer is in the test mode.
        '''
        if self.trainer._mode == 'test' and self.write_flag and (self.f is not None):
            for index in range(len(self.trainer._ids)):

                one_input = self.get_one(self.trainer._input, index)
                one_output = self.get_one(self.trainer._output, index)

                res = self.f(one_input, one_output)
                id = self.trainer._ids[index]

                self.show(res, id)

        self.tb_writer.add_scalar(
            self.name + '/loss',
            self.trainer._loss.detach().cpu(),
            self.trainer._epoch)
