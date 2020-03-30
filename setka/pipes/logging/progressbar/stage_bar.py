import os
from setka.pipes.logging.progressbar.theme_parser import recursive_substituion, max_str_len, nlines
from setka.pipes.logging.progressbar.theme import main_theme

try:
    from IPython.display import display, update_display
except:
    pass


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        module = get_ipython().__class__.__module__
        if module == "google.colab._shell":
            return True
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


class StageProgressBar:
    def __init__(self, width_function=None, config=None, display_id=0, max_float_verbose_power=2.0, is_ipython=None):
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

        self.is_ipython = isnotebook() if is_ipython is None else is_ipython
        self.display_id = display_id

    def __str__(self):
        return recursive_substituion('stage_info', self.last_vals, self.config)

    def display(self, content):
        if not self.is_ipython:
            print(content, end='')
            print('\033[' + str(nlines(content)) + 'A')
        else:
            update_display({'text/plain': content}, display_id=self.display_id, raw=True)
    
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
            if self.is_ipython:
                display({'text/plain': ''}, display_id=self.display_id, raw=True)

        self.display(cur_info)

    def finalize(self):
        if (not self.finalized) and (not self.is_ipython):
            print(str(self))
