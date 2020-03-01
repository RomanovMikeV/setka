import os
from .theme_parser import recursive_substituion, max_str_len, nlines
from .theme import main_theme


class StageProgressBar:
    def __init__(self, width_function=None, config=None, max_float_verbose_power=2.0):
        self.width_function = width_function
        self.config = config
        if self.config is None:
            self.config = main_theme()
            
        self.common = {
            'CONSOLE_WIDTH': self.width_function(),
            'MAX_LEN': 0,
            'MAX_ABS_POWER': max_float_verbose_power
        }
        self.last_vals = None
        self.finalized = False
        self.started = False
        
    def __str__(self):
        return recursive_substituion('stage_info', self.last_vals, self.config)
    
    def __del__(self):
        self.finalize()
    
    def update(self, vals):
        if self.finalized:
            return
        
        self.common['CONSOLE_WIDTH'] = self.width_function()
        vals['COMMON'] = self.common
        self.last_vals = vals
        cur_info = str(self)
        self.common['MAX_LEN'] = max(self.common['MAX_LEN'], max_str_len(cur_info))
        
        if not self.started:
            self.started = True
            print(recursive_substituion('stage_header', self.last_vals, self.config))

        print(cur_info, end='')
        print('\033[' + str(nlines(cur_info)) + 'A')
        
    def finalize(self):
        if not self.finalized:
            print(str(self))
