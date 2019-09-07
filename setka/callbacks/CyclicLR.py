from .Callback import Callback

class CyclicLR(Callback):
    '''
    This callback allows cyclic learning rate. It takes the learning rate that
    is set to the optimizer in the beginning of the epoch and
    in the end of the epoch the learning rate is set to the same value as it was
    in the beginning. During the epoch the value of Learning Rate is attenuated
    by the value of the ```cycle``` function. If you are using per-epoch
    scheduling (```ReduceLROnPlateau``` for instance) -- you will need to
    list ```CyclicLR``` callback in the list of callbacks AFTER other scheduling
    callbacks. Otherwise the per-epoch scheduling will not work.

    When epoch starts (on_epoch_begin), the values of the learning rates
    are saved.

    When batch starts (on_batch_begin), the function computes the attenuation
    coefficient for the learning rates saved during epoch start and sets the
    learning rate of all th optimizers to the attenuated value.

    When epoch ends (on_epoch_end), the values of the initial learning rates are
    restored.
    '''

    def __init__(self, cycle):
        '''
        Constructor.

        Args:
            cycle (callable): A function or callable that takes as input one
                variable (float between 0 and 1, fraction of completed steps in
                the epoch) and computes the coefficient for the learning rate
                for the next step.
        '''
        self.cycle = cycle


    def on_epoch_begin(self):
        self.lrs = []
        for optimizer in self.trainer._optimizers:
            self.lrs.append([])
            for index in range(len(optimizer.optimizer.param_groups)):
                self.lrs[-1].append(
                    optimizer.optimizer.param_groups[index]['lr'])


    def on_batch_begin(self):
        for optim_index in range(len(self.trainer._optimizers)):
            for group_index in range(len(self.trainer._optimizers[optim_index].optimizer.param_groups)):
                self.trainer._optimizers[optim_index].optimizer.param_groups[group_index]['lr'] = (
                    self.lrs[optim_index][group_index] * self.cycle(
                        self.trainer._epoch_iteration / self.trainer._n_iterations))

    def on_epoch_end(self):
        for optim_index in range(len(self.lrs)):
            for group_index in range(len(self.trainer._optimizers[optim_index].optimizer.param_groups)):
                self.trainer._optimizers[optim_index].optimizer.param_groups[group_index]['lr'] = (
                    self.lrs[optim_index][group_index])