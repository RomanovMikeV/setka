class Scheduler:
    '''
    Proxy for learning rate scheduling to use with setka.base.Optimizer.
    Args:
        scheduler_c (torch.optim.lr_scheduler.Scheduler): class of the scheduler to use.
        *args: list of arguments supported by scheduler_c (without optimizer instance)
        monitor: callable -- a function that returns the value that the scheduler will monitor.
        **kwargs: dict of arguments supported by scheduler_c
    '''
    def __init__(self, scheduler_c, monitor=None, *args, **kwargs):
        self.scheduler_c = scheduler_c
        self.args = args
        self.kwargs = kwargs
        self.monitor = monitor

    def build(self, optimizer):
        self._scheduler = self.scheduler_c(optimizer, *self.args, **self.kwargs)
        return self
