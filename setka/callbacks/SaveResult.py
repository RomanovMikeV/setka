from .Callback import Callback

import os
import torch

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



    def on_batch_end(self):
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

            torch.save(res, os.path.join('./predictions', str(self.index) + '.pth.tar'))

            if hasattr(self.trainer, "_predictions_dir"):
                torch.save(res, os.path.join(self.trainer._predictions_dir,
                    str(self.index) + 'pth.tar'))

            self.index += 1