from .Callback import Callback

class Lambda(Callback):
    '''
    This callback will help you to rapidly create simple callbacks.

    You may specify callbacks that you want to use in the following manner:
    ```
    Lambda(on_epoch_begin = foo)
    ```
    where foo is a python function.
    TODO: test it
    '''
    def __init__(self, **kwargs):

        for key in kwargs:
            setattr(self, key, kwargs[key])