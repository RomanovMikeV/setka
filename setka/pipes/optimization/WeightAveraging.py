from setka.pipes.Pipe import Pipe
import copy


class WeightAveraging(Pipe):
    """
    This pipe performs weight averaging during the training. The pipe
    duplicates a model and updates weights of it in the following way:
    $$\tilde{w}_i = \gamma \tilde{w}_i + (1.0 - \gamma) w_i,$$
    where $\tilde{w}$ is a weight of an averaged model, $\gamma$ is a parameter
    of an exponential moving average, $w_i$ is a weight of an optimized model.

    This pipes does nothing until specified epoch. After this epoch it
    begins tracking of the averaged model. During validation and testing, the
    averaged model is used.

    Args:
        gamma (float): factor of Exponential Moving Average.

        epoch_start (int): epoch when the averaging starts.

        interval (int): interval between the iterations when the model is averaged.

    """

    def __init__(self, gamma=0.99, epoch_start=10, interval=10):
        super(WeightAveraging, self).__init__()
        self.gamma = gamma
        self.epoch_start = epoch_start
        self.interval = interval

    def before_epoch(self):
        """
        Copies the current model, changes the model to the average model in case the Trainer is in
        'valid' or 'test' modes.
        """
        if (self.trainer._epoch >= self.epoch_start and
            not hasattr(self, 'averaged_model') and
            self.trainer._mode == 'train'):

            self.averaged_model = copy.deepcopy(self.trainer._model)

        if (hasattr(self, 'averaged_model') and (
                self.trainer._mode == 'valid' or
                self.trainer._mode == 'test')):
            self.trainable_model = self.trainer._model
            self.trainer._model = self.averaged_model

    def after_epoch(self):
        """
        Sets the model from averaged to trained if the trainer is in 'valid' or 'test' modes.
        """
        if (hasattr(self, 'averaged_model') and (
                self.trainer._mode == 'valid' or
                self.trainer._mode == 'test')):

            self.trainer._model = self.trainable_model

    def after_batch(self):
        """
        If trainer is in 'train' mode, the averaging of the models is performed.
        """
        if (hasattr(self, 'averaged_model') and
            self.trainer._mode == 'train'):

            avg_pars = self.averaged_model.parameters()
            trn_pars = self.trainer._model.parameters()

            if self.trainer._iteration % self.interval ==0:
                for avg_par, trn_par in zip(avg_pars, trn_pars):
                    avg_par = avg_par * (1.0 - self.gamma) + trn_par * self.gamma
