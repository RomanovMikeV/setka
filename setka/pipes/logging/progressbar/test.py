import numpy as np
import os
import time

from stage_bar import StageProgressBar
from theme_parser import progress_str, recursive_substituion
from theme import main_theme


config = main_theme()            
w = os.get_terminal_size()[0]
vals = {
    'COMMON': {},
    'cur_epoch': 9,
    'n_epochs': 50,
    'cur_stage': 'train',
    'cur_subset': 'train',
    'cur_iter': 253,
    'max_iter': 1000,
    'time_str': '[1:42:53>2:56:00]',
    'percentage_completed': 0.0,
    'progress_bar': progress_str(30, 0.0),
    'batch_time': 0.0653,
    'data_time': 0.054,
    'loss': [
        {'loss_name': 'Loss', 'loss_value': np.random.rand()},
        {'loss_name': 'L1-loss', 'loss_value': np.random.rand()},
        {'loss_name': 'grad-loss', 'loss_value': np.random.rand()},
        {'loss_name': 'eigen', 'loss_value': np.random.rand()}
    ],
    'metrics': [
        {'metric_name': 'd1', 'metric_value': np.random.rand()},
        {'metric_name': 'd2', 'metric_value': np.random.rand()},
        {'metric_name': 'd3', 'metric_value': np.random.rand()},
        {'metric_name': 'rel', 'metric_value': np.random.rand()},
        {'metric_name': 'log10', 'metric_value': np.random.rand()}
    ]
}


def rval():
    val = np.random.rand()
    if np.random.rand() > 0.7:
        val *= 200
    return val


def update_vals(vals, i, N_ITERS):
    if i % 10 == 0:
        for j in range(len(vals['loss'])):
            vals['loss'][j]['loss_value'] = rval()
        for j in range(len(vals['metrics'])):
            vals['metrics'][j]['metric_value'] = rval()
    vals['cur_iter'] = i
    vals['percentage_completed'] = float(i) * 100 / N_ITERS 
    vals['progress_bar'] = progress_str(30, float(i) / N_ITERS)

    
def animate_stage(N_ITERS, stage, sleep=0.01):
    vals['max_iter'] = N_ITERS
    vals['cur_stage'] = stage
    
    pbar = StageProgressBar()
    for i in range(1, N_ITERS+1):
        update_vals(vals, i, N_ITERS)
        
        pbar.update(vals)
        time.sleep(sleep)


if __name__ == '__main__':
    print('-' * w)
    print(recursive_substituion('epoch_global', vals, config))

    animate_stage(500, 'train')
    animate_stage(200, 'valid')
    vals['metrics'] = []
    vals['loss'] = []
    animate_stage(200, 'test')
    print('-' * w)
