from copy import deepcopy
import math
import os
import re
import string
from termcolor import colored


class FormatableList:
    def __init__(self, single_pattern, key, config, prefix='', delimiter=' | '):
        self.prefix = prefix
        self.delimiter = delimiter
        self.single_pattern = single_pattern
        self.config = config
        self.key = key

    def arguments(self):
        return get_argument_names(self.single_pattern)

    def format(self, **vals):
        if len(vals[self.key]) == 0:
            return ''
        
        lst = []
        for cur_dict in vals[self.key]:
            cur_dict['COMMON'] = vals['COMMON'] 
            lst.append(colored_subst(self.single_pattern, cur_dict, self.config))
        
        return self.prefix + self.delimiter.join(lst)


def get_argument_names(str_pattern):
    if isinstance(str_pattern, str):
        return [t[1] for t in string.Formatter().parse(str_pattern) if t[1] is not None]
    elif isinstance(str_pattern, FormatableList):
        return []


def colored_subst(str_pattern, vals, config):
    vals_subst = deepcopy(vals)
    for arg in get_argument_names(str_pattern):
        if arg in config['formatters']:
            fmtr = config['formatters'][arg]
            if isinstance(fmtr, str):
                vals_subst[arg] = fmtr.format(vals[arg])
            else:
                vals_subst[arg] = fmtr(vals[arg], vals['COMMON'])

    for arg in get_argument_names(str_pattern):
        if arg in config['colors']:
            clr = None
            if isinstance(config['colors'][arg], dict):
                if vals_subst[arg] in config['colors'][arg]:
                    vals_subst[arg] = colored(vals_subst[arg], config['colors'][arg][vals[arg]])
            else:
                if config['colors'][arg] != 'default':
                    vals_subst[arg] = colored(vals_subst[arg], config['colors'][arg])
    return str_pattern.format(**vals_subst)


def recursive_substituion(key, vals, config):
    subst = {'COMMON': vals['COMMON']}
    if isinstance(config[key], FormatableList):
        subst[config[key].key] = vals[config[key].key]
    
    for arg in get_argument_names(config[key]):
        subst[arg] = recursive_substituion(arg, vals, config) if arg in config else vals[arg]
        
    return colored_subst(config[key], subst, config)


def progress_str(width, state):
    filled = int(round(width * state)) - 1
    return '[' + '='* filled + '>' + ' ' * (width - filled - 1) + ']' 


def escape_ansi(line):
    return re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]').sub('', line)


def fullstr_formatter(val, common):
    if val == '': # If string is empty we don`t need to pad it
        return val
    
    # Calculate desired line length
    str_len = len(escape_ansi(val))
    common['MAX_LEN'] = max(common['MAX_LEN'], str_len)
    w = min(common['CONSOLE_WIDTH'], common['MAX_LEN'])
    
    # Make padding or trim it in case of overflow
    pad = max(w - str_len, 0)
    val = val + ' ' * pad
    if str_len > w:
        # TODO: Trimming shouldn`t affect special ANSI sequences
        # TODO: anyway, add \u001b[0m in the end of line
        val = val[:-str_len + w]
    return val + '\n'


def nline_format(val, common):
    return val + '\n'


def nlines(text):
    return text.count('\n') + 1


def value_format(val, common):
    pow_max = common['MAX_ABS_POWER']
    if isinstance(val, (list, tuple)):
        res = []
        for index in range(len(val)):
            res.append(value_format(val[index], common))
        return res
    
    if (val > 10 ** pow_max) or (val < 10 ** (-pow_max)):
        return '{:.3e}'.format(val)
    else:
        return '{:.5f}'.format(val)


def max_str_len(text):
    return max([len(s) for s in escape_ansi(text).split('\n')])
