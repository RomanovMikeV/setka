class OptimizerSwitch():
    '''
    Contains optimizer and the module that this optimizer should optimize together with active state flag.

    :param train_module: torch.Module to optimize
    :param optimizer: optimizer to use
    :param is_active: optimizer is active if set to True
    :param recurse: retrieve torch.Module parameters recursively if True.
    '''

    def __init__(self, train_module, optimizer, is_active=True, recurse=True, **kwargs):
        self.optimizer = optimizer(train_module.parameters(recurse=recurse), **kwargs)
        self.module = train_module
        self.active = is_active

