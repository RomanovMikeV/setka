from .Pipe import Pipe

import gc

class GarbageCollector(Pipe):
    '''
    Probably should be merged with the Dataset handler (?).
    '''
    def __init__(self):
        gc.enable()

    def after_epoch(self):
        gc.collect()