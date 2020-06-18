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

### NEW ENGINE

import numpy
import copy
import re

status = {
    'Info': '5/10 [######9     ] 59%',
    'Time': {'D': 0.9999999999999999999999,
             'B': 0.1238476123864126345187245,
             'AvgD': 0.812341348176234987162349817643,
             'AvgB': 0.8561283745187634581726345},
    'Loss': {'L1': 112398746192834619827364,
             'L2': 19234618623.0,
             'L2.5': 1,
             'L3': 0.8126347152,
             'L4': 10,
             'L5': numpy.inf},
    'Metrics': {'One': 0.888888,
                'Two': 123987.0,
                'Three': 1.0}
}


def textcolor(style=None, color=None):
    if color is None:
        color = 0
    else:
        color_code = 30 + color

    if style is None:
        style_code = 0
    else:
        style_code = style
    return '\033[' + str(style_code) + ';' + str(color_code) + 'm', '\033[' + str(0) + ';' + str(0) + 'm'


def format_status(inp):
    if isinstance(inp, (dict)):
        for key in inp:
            inp[key] = format_status(inp[key])

    if isinstance(inp, (list, tuple)):
        for index in range(len(inp)):
            inp[index] = format_status(inp[index])

    if isinstance(inp, int):
        if abs(inp) > 10 ** 6:
            return '{:.3e}'.format(inp)
        else:
            return '{:d}'.format(inp)

    if isinstance(inp, float):
        if abs(inp) > 10 ** 6:
            return '{:.3e}'.format(inp)
        elif abs(inp) < 10 ** -6:
            return '{:.3e}'.format(inp)
        else:
            return '{:.6f}'.format(inp)

    return inp


def colorize_string(string, colors, padding=0):
    indice = []
    for color in colors:
        indice.append(colors[0])

    substrings = []
    last_index = 0
    for color in colors:
        index = color[0] + padding
        substrings.append(string[last_index:index])
        substrings.append(color[1])
        last_index = index
    substrings.append(string[last_index:])
    return ''.join(substrings)


def view_status(inp, display_len=80):
    separator = ' | '
    strings = ['']
    colors = [[]]
    color_index = 0

    maxlen = 0
    for key in inp:
        maxlen = max(len(str(key)), maxlen)

    for key in inp:

        start, end = textcolor(style=1, color=color_index + 1)
        colors[-1].append((len(strings[-1]), start))
        strings[-1] += ('{:>' + str(maxlen) + 's} ').format(key)
        colors[-1].append((len(strings[-1]), end))

        if isinstance(inp[key], (list, tuple)):
            strings[-1] += separator.join(inp[key])

        elif isinstance(inp[key], dict):
            pos = len(strings[-1])
            subres = []

            for subkey in inp[key]:
                start, end = textcolor(style=3, color=color_index + 1)
                colors[-1].append((pos, start))
                colors[-1].append((pos + len(subkey), end))
                subres.append(subkey + ': ' + str(inp[key][subkey]))
                pos = pos + len(subkey) + len(': ') + len(str(inp[key][subkey])) + len(separator)
            strings[-1] += separator.join(subres)

        else:
            strings[-1] += str(inp[key])

        strings.append('')
        colors.append([])

        color_index += 1
        color_index %= 6

    new_strings = []
    new_colors = []
    for index in range(len(strings)):
        string = strings[index]
        str_colors = colors[index]
        position = 0
        color_index = 0
        padding = 0
        while len(string) > 0:
            splitter_location = -1

            if len(string) > display_len:
                splitter_location = string[:display_len].rfind(' | ')

            split_colors = []

            if splitter_location > 0:
                string_end = splitter_location
            else:
                string_end = min(display_len, len(string))
            while color_index < len(colors[index]) and colors[index][color_index][0] - position < string_end - padding:
                split_colors.append(list(colors[index][color_index]))
                split_colors[-1][0] -= position
                color_index += 1

            if len(string) < display_len:
                to_print = string
                to_print = to_print + ' ' * (display_len - len(to_print))
                new_strings.append(colorize_string(to_print, split_colors, padding=padding))
                break

            elif splitter_location > 0:
                to_print = string[:splitter_location]
                to_print = to_print + ' ' * (display_len - len(to_print))
                new_strings.append(colorize_string(to_print, split_colors, padding=padding))
                split_colors = []
                string = ' ' * (maxlen + 1) + string[splitter_location + 3:]
                position += splitter_location + 3 - padding
                padding = maxlen + 1

            else:
                to_print = string[:string_end]
                to_print = to_print + ' ' * (display_len - len(to_print))
                new_strings.append(colorize_string(to_print, split_colors, padding=padding))
                split_colors = []
                string = ' ' * (maxlen + 1) + string[string_end:]
                position += string_end - padding
                padding = maxlen + 1

    new_strings.append('=' * display_len)
    return '\n'.join(new_strings)

