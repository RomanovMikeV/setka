from .Pipe import Pipe

import os
import torch

class SaveResult(Pipe):
    '''
    pipe for saving predictions of the model. The results are
    stored in a directory ```predictions``` and in directory specified in
    ```trainer._predictions_dir```. Batches are processed with the
    specified function ```f```. The directory is flushed during the
    ```__init__``` and the result is saved when the batch is
    finished (when after_batch is triggered).

    Args:
        f (callable): function to process the predictions.
        dir (string): location where the predictions will be saved
    '''
    def __init__(self,
                 f=None,
                 dir='./'):
        self.f = f
        self.index = 0
        self.dir = dir
        self.root_dir = os.path.join(self.dir, './predictions')

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    @staticmethod
    def get_one(input, item_index):
        if isinstance(input, (list, tuple)):
            one = []
            for list_index in range(len(input)):
                one.append(input[list_index][item_index])
            return one

        else:
            one = input[item_index]
            return one



    def after_batch(self):
        if self.trainer._mode == 'test':
            res = {}
            for index in range(len(self.trainer._ids)):


                one_input = self.get_one(self.trainer._input, index)
                one_output = self.get_one(self.trainer._output, index)

                if self.f is not None:
                    res[self.trainer._ids[index]] = self.f(
                        one_input,
                        one_output)
                else:
                    res[self.trainer._ids[index]] = one_output

            torch.save(res, os.path.join(self.root_dir, str(self.index) + '.pth.tar'))

            self.index += 1