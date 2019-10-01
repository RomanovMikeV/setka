import time
import tqdm

from .Callback import Callback

class ProgressBar(Callback):
    def __init__(self, filter=None):
        self.set_priority(-100)
        try:
            self.pbar = tqdm.tqdm_notebook(
                range(100),
                leave=False,
                bar_format='{n}/|/{percentage:3.0f}% [{elapsed}>{remaining}] {rate_fmt} {postfix}')
        except:
            self.pbar = tqdm.tqdm(
                range(100),
                ascii=True,
                leave=False,
                bar_format='{percentage:3.0f}% [{elapsed}>{remaining}] {rate_fmt} {postfix}'
            )

            print(self.pbar.bar_format)


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

    def on_epoch_begin(self):
        self.pbar.clear()
        self.pbar.n = 0
        self.pbar.start_t = time.time()

    def on_epoch_end(self):
        self.status_string = '  '.join([str(k) + ': ' + self.format(v) for k, v in self.trainer.status.items()])
        if hasattr(self, 'status_string'):
            if isinstance(self.pbar, tqdm._tqdm_notebook.tqdm_notebook):
                print(self.status_string)
            else:
                self.pbar.write(self.status_string)

    def on_batch_end(self):

        self.pbar.n = int(float(self.trainer._epoch_iteration) / float(self.trainer._n_iterations) * 100.0)
        self.pbar.refresh()

        self.status_string = '  '.join([str(k) + ': ' + self.format(v) for k, v in self.trainer.status.items()])

        self.pbar.set_postfix_str(self.status_string)

    def on_train_end(self):
        self.pbar.close()