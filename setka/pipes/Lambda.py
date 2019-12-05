from .Pipe import Pipe

class Lambda(Pipe):
    '''
    This pipe will help you to rapidly create simple pipes.

    You may specify pipes that you want to use in the following manner:
    ```
    Lambda(before_epoch = foo)
    ```
    where foo is a python function.
    '''
    def __init__(self, **kwargs):

        for key in kwargs:
            setattr(self, key, kwargs[key])