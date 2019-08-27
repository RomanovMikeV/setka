from .Callback import Callback

import gc

class GarbageCollector(Callback):
    def __init__(self):
        gc.enable()

    def on_epoch_end(self):
        gc.collect()