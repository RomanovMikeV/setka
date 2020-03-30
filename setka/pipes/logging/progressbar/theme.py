from .theme_parser import FormatableList, fullstr_formatter, value_format, nline_format
from copy import deepcopy


def main_theme():
    config = {
        'colors': {
            'epoch_str': 'red',
            'cur_stage': {
                'train': 'green',
                'valid': 'yellow',
                'test': 'magenta'
            },
            'loss_name': 'cyan',
            'metric_name': 'blue'
        },
        'formatters': {
            'progress_str': fullstr_formatter,
            'loss_str': fullstr_formatter,
            'metrics_str': fullstr_formatter,
            'detailed_time_str': fullstr_formatter,
            'loss_value': value_format,
            'metric_value': value_format,
            'percentage_completed': '{:.1f}',
            'batch_time': '{:.4f}',
            'data_time': '{:.4f}',
            'pbar_time': '{:.2f}',
        },
        
        'stage_header': '* Stage: {cur_stage}, subset: {cur_subset}',
        'stage_info': '{progress_str}{loss_str}{metrics_str}{detailed_time_str}',
        'epoch_global': '{epoch_str}',
        
        'progress_str': '{stage_prefix}Iteration: {cur_iter}/{max_iter} {time_str} {percentage_completed}% {progress_bar}',
        'detailed_time_str': '{stage_prefix}Batch time: {batch_time}s. Data time: {data_time}s.',
        'epoch_str': 'Epoch: {cur_epoch:3d} /{n_epochs:3d}',
        'stage_prefix': ' ' * 5,
    }

    config['loss_str'] = FormatableList('{loss_name}: {loss_value}', 'loss', config, prefix=config['stage_prefix'])
    config['metrics_str'] = FormatableList('{metric_name}: {metric_value}', 'metrics', config, prefix=config['stage_prefix'])
    return config


def adopt2ipython(config):
    new_config = deepcopy(config)
    for fmt_key in config['formatters']:
        if config['formatters'][fmt_key] == fullstr_formatter:
            new_config['formatters'][fmt_key] = nline_format
    return new_config
