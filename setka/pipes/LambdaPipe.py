from .Pipe import Pipe


class LambdaPipe(Pipe):
    """
    This pipe will help you to rapidly create simple pipes.

    You may specify pipes that you want to use in the following manner:
    ```
    Lambda(before_epoch = foo)
    ```
    where foo is a python function.
    """
    def __init__(self, **kwargs):
        super(LambdaPipe, self).__init__()

        for key in kwargs:
            setattr(self, key, kwargs[key])
