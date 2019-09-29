class OptimizerSwitch():
    '''
    This class contains optimizer and the module that this
    optimizer should optimize. The OptimizerSwitch tells trainer
    if it should switch the modules to the training mode and back
    to evaluation mode and if it should perform optimization for
    this module's parameters.
    '''

    def __init__(self, train_module, optimizer, is_active=True, recurse=True, **kwargs):
        '''
        Constructs the OptimizerSwitch instance.

        Args:
            train_module (torch.nn.Module):

            optimizer (torch.optim.Optimizer):

            is_active (bool):

            **kwargs:

        Example:
        ```
        OptimizerSwitch(
            net.params(),
            torch.optim.Adam,
            lr=3.0e-4,
            is_active=False)
        ```
        '''

        self.optimizer = optimizer(train_module.parameters(recurse=recurse), **kwargs)
        self.module = train_module
        self.active = is_active