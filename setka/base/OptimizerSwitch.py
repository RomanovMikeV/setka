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

    def __init__(self, train_module, optimizer, schedulers=[], is_active=True, recurse=True, **kwargs):
        self.optimizer = optimizer(train_module.parameters(recurse=recurse), **kwargs)
        self.module = train_module
        self.active = is_active
        self.schedulers = schedulers

        for key in self.schedulers:
            if not isinstance(self.schedulers[key], list):
                self.schedulers[key] = [self.schedulers[key]]
            # print(self.schedulers[key])
            for i in range(len(self.schedulers[key])):
                if isinstance(self.schedulers[key][i], (tuple, list)):
                    # Format: (scheduler, subset, metric_name)
                    self.schedulers[key][i] = list(self.schedulers[key][i])
                    self.schedulers[key][i][0] = self.schedulers[key][i][0](self.optimizer)
                    # if len(self.schedulers[key][i]) == 3:
                    #     self.schedulers[key][i] = (
                    #         self.schedulers[key][i][0](self.optimizer),
                    #         self.schedulers[key][i][1],
                    #         self.schedulers[key][i][2])
                    # elif len(self.schedulers[key][i]) == 4:
                    #     self.schedulers[key][i] = (
                    #         self.schedulers[key][i][0](self.optimizer),
                    #         self.schedulers[key][i][1],
                    #         self.schedulers[key][i][2],
                    #         self.schedulers[key][i][3])
                else:
                    self.schedulers[key][i] = self.schedulers[key][i](self.optimizer)

        print(self.schedulers)

    def schedulers_step(self):
        if self.schedulers is None:
            return None

        for i in range(len(self.schedulers)):
            if (isinstance(self.schedulers[i], tuple) and
                hasattr(self.trainer, '_metrics') and
                self.schedulers[i][1] in self.trainer._metrics and
                self.schedulers[i][2] in self.trainer._metrics[self.schedulers[i][1]]):

                if len(self.schedulers[i]) == 3:
                    self.schedulers[i].step(self.trainer._metrics[self.schedulers[i][1]][self.schedulers[i][2]])
                elif len(self.schedulers[i]) == 4:
                    self.schedulers[i].step(self.trainer._metrics[self.schedulers[i][1]][self.schedulers[i][2]], )

            else:
                self.schedulers[i].step()
