import tqdm

from .Callback import Callback

class ProgressBar(Callback):
    def __init__(self, filter=None):
        self.set_priority(-100)
        try:
            self.pbar = tqdm.tqdm_notebook(leave=False)
        except:
            self.pbar = tqdm.tqdm(leave=False, ascii=True, )


    @staticmethod
    def format(val):
        res = str(val)
        if isinstance(val, int):
            res = '{0:5d}'.format(val)
        if isinstance(val, float):
            res = '{0:.2e}'.format(val)
        if isinstance(val, str):
            res = '{0:5s}'.format(val)
        return res


    def on_epoch_end(self):
        self.status_string = '  '.join([str(k) + ': ' + self.format(v) for k, v in self.trainer.status.items()])
        if hasattr(self, 'status_string'):
            self.pbar.write(self.status_string)

        self.pbar.reset()


    def on_batch_end(self):

        self.pbar.total = self.trainer._n_iterations
        self.pbar.update()

        self.status_string = '  '.join([str(k) + ': ' + self.format(v) for k, v in self.trainer.status.items()])

        self.pbar.set_description(self.status_string)

    def on_train_end(self):
        self.pbar.close()