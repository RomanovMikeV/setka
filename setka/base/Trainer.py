import collections
import numpy

class Trainer():
    '''
    Trainer can train and validate your network. It can also
    predict results.

    It contains the following methods:

    * __init__ -- class constructor, where you specify everything
        that you will need during the training procedure.

    * train_one_epoch -- performs training for one epoch

    * validate_one_epoch -- performs validation for one epoch

    * predict -- predicts results

    * save -- dumps network's and optimizer's states to the file

    * load -- loads network's and optimizer's states from the file

    '''

    def __init__(self,
                 callbacks=[],
                 seed=0,
                 max_threads=4,
                 deterministic_cuda=False,
                 benchmark_cuda=True,
                 silent=False):
        '''
        Class constructor.

        Args:
            model (base.Network) -- model to train

            optimizers (list of base.OptimizerSwitch) -- optimizer
                to use during training. The optimisers whose
                is_active flag is set to True will be used in the
                optimization procedure (there may be many of them)

            criterion (function) -- criterion (loss) function that
                needs to be optimized

            callbacks (list of callbacks.Callback) -- callbacks
                that extend functionality of the trainer.

            seed (int) -- seed for random number generators to use.

            deterministic_cuda (bool) -- flag that specifies if
                the cuda should behave deterministically
                (takes more time to perform computations)

            silent (bool) -- do not show output


        Args:
            model (base.Network): the model for the socket.

            callbacks (list): List of setka.callbacks to use
                during training.

            seed (int): seed to initialize the random value
                generators. 0 by default.

            deterministic_cuda (bool): whether to use deterministic
                CUDA backend. If True, the computations are slower,
                but deterministic.

            batch_to_metrics (int): how many batches to accumulate before
            metrics computation. Default is 1. If None -- use all batches for
            metrics computation.

            silent (bool): no outputs if True.
        '''
        # self.setup_environment(seed=seed,
        #                        max_threads=max_threads,
        #                        deterministic_cuda=deterministic_cuda,
        #                        benchmark_cuda=benchmark_cuda)

        self._callbacks = callbacks

        for callback in self._callbacks:
            callback.trainer = self

        self.status = collections.OrderedDict()

        self._epoch = 0
        self.status['epoch'] = 0
        self._iteration = 0
        self.status['iteration'] = 0

        self._best_metrics = None

        self._stop_epoch = False
        self.run_callbacks('on_init')


    def run_callbacks(self, stage):
        priorities = []
        for callback in self._callbacks:

            if hasattr(callback, 'priority'):
                if isinstance(callback.priority, dict):
                    if stage in callback.priority:
                        priorities.append(-callback.priority[stage])
                    else:
                        priorities.append(0)
                else:
                    priorities.append(-callback.priority)
            else:
                priorities.append(0)
        priorities = numpy.array(priorities)

        order = priorities.argsort(kind='stable')

        for index in order:
            getattr(self._callbacks[index], stage)()


    def one_epoch(self,
                  mode='valid',
                  subset='valid',
                  n_iterations=None):

        '''
        Trains a model for one epoch.

        Args:
            dataset (base.DataSet): dataset instance to train on.

            subset (hashable): identifier of the dataset's subset
                which willbe used for training.

            batch_size (int): batch size to use during training.

            num_workers (int): number of workers to use in
                torch.utils.data.DataLoader.

            max_iterations (int): maximum amount of iterations to
                perform during one epoch. If None -- training
                will be performed until the end of the subset.
        '''

        self.status['mode'] = mode
        self._mode = mode

        self.status['subset'] = subset
        self._subset = subset

        if mode == 'train':
            self.status['epoch'] += 1
            self._epoch += 1

        self._epoch_iteration = 0

        self.run_callbacks('on_epoch_begin')
        
        if n_iterations is not None:
            n_iterations = min(self._n_iterations, n_iterations)
        else:
            n_iterations = self._n_iterations
        
        for i in range(n_iterations):

            if mode == 'train':
                self._iteration += 1
                self.status["iteration"] += 1

            self._epoch_iteration += 1

            self.run_callbacks('on_batch_begin')
            self.run_callbacks('on_batch_run')
            self.run_callbacks("on_batch_end")

            if hasattr(self, "_stop_epoch"):
                if self._stop_epoch:
                    self._stop_epoch = False
                    return

        self.run_callbacks("on_epoch_end")


    def get_optimizers_states(self):
        '''
        Gets the optimizers states.

        Returns:
            list with optimizers states
        '''
        res = []
        for optimizer in self._optimizers:
            res.append(optimizer.optimizer.state_dict())
        return res


    def set_optimizers_states(self, states):
        '''
        Sets the optimizers states.

        Args:
            states: list of states for each of the optimizer.
        '''
        for opt_index in range(len(states)):
            self._optimizers[opt_index].optimizer.load_state_dict(states[opt_index])


    def get_optimizers_flags(self):
        '''
        Gets optimizers active flags.

        Returns:
            list with optimizer active flags.
        '''
        res = []
        for optimizer in self._optimizers:
            res.append(optimizer.active)
        return res


    def set_optimizers_flags(self, flags):
        '''
        Sets optimizers active flags.

        Args:
             flags (list): list with optimizers active flags.
        '''
        for opt_index in range(len(flags)):
            self._optimizers[opt_index].active = flags[opt_index]
