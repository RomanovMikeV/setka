import time
import tqdm

from .Pipe import Pipe

class ProgressBar(Pipe):
    '''
    This pipe shows progress of the training.
    '''
    def __init__(self, mode='bash'):
        self.set_priority(-100)
        self.mode = mode


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

    def before_epoch(self):
        '''
        Initializes the progressbar
        '''
        if self.mode == 'notebook':
            self.pbar = tqdm.tqdm_notebook(
                range(100),
                leave=False,
                bar_format='{n}/|/{percentage:3.0f}% [{elapsed}>{remaining}] {rate_fmt} {postfix}')
        elif self.mode == 'bash':
            self.pbar = tqdm.tqdm(
                range(100),
                ascii=True,
                leave=False,
                bar_format='{percentage:3.0f}% [{elapsed}>{remaining}] {rate_fmt} {postfix}')
        
        self.pbar.clear()
        self.pbar.n = 0
        self.pbar.start_t = time.time()

        
    def after_epoch(self):
        '''
        Clears the progressbar, keeps the status message.
        '''
        self.status_string = '  '.join([str(k) + ': ' + self.format(v) for k, v in self.trainer.status.items()])
        
        
        #print(self.status_string)
        if hasattr(self, 'status_string'):
            self.pbar.write(self.status_string)
        
        self.pbar.clear()
        del self.pbar

    def after_batch(self):
        '''
        Updates the progressbar.
        '''

        self.pbar.n = int(float(self.trainer._epoch_iteration) / float(self.trainer._n_iterations) * 100.0)
        self.pbar.refresh()

        self.status_string = '  '.join([str(k) + ': ' + self.format(v) for k, v in self.trainer.status.items()])

        self.pbar.set_postfix_str(self.status_string)
