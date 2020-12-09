import os
from setka.pipes.logging.progressbar.theme_parser import view_status, format_status
# from setka.pipes.logging.progressbar.theme import main_theme

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

def nlines(text):
    return text.count('\n') + 1


class StageProgressBar:
    def __init__(self, width_function=None, display_id=0, is_ipython=None):
        self.width_function = width_function
            

        self.last_vals = None
        self.finalized = False
        self.started = False

        self.is_ipython = isnotebook() if is_ipython is None else is_ipython
        self.display_id = display_id
        
        self.width = self.width_function()

    def __str__(self):
        status = format_status(self.last_vals)
        to_view = view_status(status, display_len=self.width)
        return to_view

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
        
        self.width = self.width_function()
        self.last_vals = vals
        cur_info = str(self)

        if not self.started:
            self.started = True
            if self.is_ipython:
                display({'text/plain': ''}, display_id=self.display_id, raw=True)

        self.display(cur_info)

    def finalize(self):
        if (not self.finalized) and (not self.is_ipython):
            print(str(self))
