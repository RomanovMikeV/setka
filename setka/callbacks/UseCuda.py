from .Callback import Callback
import torch

class UseCuda(Callback):
    @staticmethod
    def move2cuda(coll):
        if isinstance(coll, dict):
            for key in coll:
                coll[key] = coll[key].cuda()
            return coll
                
        elif isinstance(coll, (list, tuple)):
            new_coll = []
            for val in coll:
                new_coll.append(val.cuda())
            return new_coll
                
        else:
            coll = coll.cuda()
            return coll
        
    def __init__(self):
        pass
    
    def on_epoch_begin(self):
        self.trainer._model.cuda()
    
    def on_batch_begin(self):
        self.move2cuda(self.trainer._input)
