class OptimizerSwitch:
    """
    Contains optimizer and the module that this optimizer should optimize together with active state flag.

    Arguments:
        train_module (torch.Module): module to optimize
        optimizer: optimizer class to set up for given module
        is_active (bool): Whether optimizer
        schedulers (optional, list of scheduler classes or tuples): Schedulers to apply after optimizer step.
            Single class or list of scheduler classes, which takes optimizer on initialization and has .step() method
            implemented. If optimizer should take additional metric value on .step() method call, specify a tuple
            (SchedulerClass, 'metric_name'). Metric 'metric_name' should be computed by setka.pipes.ComputeMetrics pipe
            E.g.:
                schedulers=[torch.optim.lr_scheduler.CyclicLR,
                            (torch.optim.lr_scheduler.ReduceLROnPlateau, 'accuracy')]
    """

    def __init__(self, train_module, optimizer, schedulers=None, is_active=True, recurse=True, **kwargs):
        self.optimizer = optimizer(train_module.parameters(recurse=recurse), **kwargs)
        self.module = train_module
        self.active = is_active
        self.schedulers = schedulers
        if self.schedulers is not None:
            if not isinstance(self.schedulers, list):
                self.schedulers = [self.schedulers]

            for i in range(len(self.schedulers)):
                if isinstance(self.schedulers[i], tuple):
                    # Format: (scheduler, metric_name)
                    self.schedulers[i] = (self.schedulers[i][0](self.optimizer), self.schedulers[i][1])
                else:
                    self.schedulers[i] = self.schedulers[i](self.optimizer)

    def schedulers_step(self):
        if self.schedulers is None:
            return None

        for i in range(len(self.schedulers)):
            if isinstance(self.schedulers[i], tuple):
                self.schedulers[i].step(self.trainer._metrics[self.schedulers[i][1]])
            else:
                self.schedulers[i].step()
