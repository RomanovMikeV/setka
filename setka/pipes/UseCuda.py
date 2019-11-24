from .Pipe import Pipe
import torch

class UseCuda(Pipe):
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
    
    def before_epoch(self):
        self.trainer._model.cuda()
    
    def before_batch(self):
        self.move2cuda(self.trainer._input)
