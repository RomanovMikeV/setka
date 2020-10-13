class Optimizer:
    """
    Contains optimizer and the module that this optimizer should optimize together with active state flag.

    Arguments:
        train_module (torch.Module): module to optimize
        optimizer: optimizer class to set up for given module
        is_active (bool): Whether optimizer
        iter_schedulers (optional, list of schedulers): list of setka.base.Schedulers that will be called after each
            iteration. May be useful for Cycling Learning Rate schedulers
        epoch_schedulers (optional, list of schedulers): list of setka.base.Schedulers that will be called after each
            epoch. May be useful for ReduceLROnPlateau schedulers
    """

    def __init__(self, train_module, optimizer, iter_schedulers=[], epoch_schedulers=[], is_active=True, recurse=True,
                 **kwargs):
        self.optimizer = optimizer(train_module.parameters(recurse=recurse), **kwargs)
        self.module = train_module
        self.active = is_active
        self.epoch_schedulers = epoch_schedulers
        self.iter_schedulers = iter_schedulers

        self._epoch_schedulers = []
        for scheduler in self.epoch_schedulers:
            self._epoch_schedulers.append(scheduler.build(self.optimizer))

        self._iter_schedulers = []
        for scheduler in self.iter_schedulers:
            self._iter_schedulers.append(scheduler.build(self.optimizer))

        # for key in self.schedulers:
        #     if not isinstance(self.schedulers[key], list):
        #         self.schedulers[key] = [self.schedulers[key]]
        #     # print(self.schedulers[key])
        #     for i in range(len(self.schedulers[key])):
        #         if isinstance(self.schedulers[key][i], (tuple, list)):
        #             # Format: (scheduler, subset, metric_name)
        #             self.schedulers[key][i] = list(self.schedulers[key][i])
        #             self.schedulers[key][i][0] = self.schedulers[key][i][0](self.optimizer)
        #             # if len(self.schedulers[key][i]) == 3:
        #             #     self.schedulers[key][i] = (
        #             #         self.schedulers[key][i][0](self.optimizer),
        #             #         self.schedulers[key][i][1],
        #             #         self.schedulers[key][i][2])
        #             # elif len(self.schedulers[key][i]) == 4:
        #             #     self.schedulers[key][i] = (
        #             #         self.schedulers[key][i][0](self.optimizer),
        #             #         self.schedulers[key][i][1],
        #             #         self.schedulers[key][i][2],
        #             #         self.schedulers[key][i][3])
        #         else:
        #             self.schedulers[key][i] = self.schedulers[key][i](self.optimizer)
        #
        # print(self.schedulers)

    def step_iter_schedulers(self):
        for scheduler in self._iter_schedulers:
            if scheduler.monitor is None:
                scheduler._scheduler.step()
            else:
                scheduler._scheduler.step(scheduler.monitor())


    def step_epoch_schedulers(self):
        for scheduler in self._epoch_schedulers:
            if scheduler.monitor is None:
                scheduler._scheduler.step()
            else:
                scheduler._scheduler.step(scheduler.monitor())

    # def schedulers_step(self):
    #     if self.schedulers is None:
    #         return None
    #
    #     for i in range(len(self.schedulers)):
    #         if (isinstance(self.schedulers[i], tuple) and
    #             hasattr(self.trainer, '_metrics') and
    #             self.schedulers[i][1] in self.trainer._metrics and
    #             self.schedulers[i][2] in self.trainer._metrics[self.schedulers[i][1]]):
    #
    #             if len(self.schedulers[i]) == 3:
    #                 self.schedulers[i].step(self.trainer._metrics[self.schedulers[i][1]][self.schedulers[i][2]])
    #             elif len(self.schedulers[i]) == 4:
    #                 self.schedulers[i].step(self.trainer._metrics[self.schedulers[i][1]][self.schedulers[i][2]], )
    #
    #         else:
    #             self.schedulers[i].step()
