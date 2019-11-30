import collections
import numpy
import pandas
import datetime

class Trainer():
    '''

    Trainer is a class that takes control over the training procedure. It is the main module of the whole Setka.

    :param pipes: list or tuple of pipes for a pipeline.
    :param train_flow: callbacks in the order in which they are called when the trainer.run_training is called
    '''

    def __init__(self,
                 pipes=[],
                 train_flow=['before_train', 'on_train', 'after_train'],
                 epoch_flow=['before_epoch', 'on_epoch', 'after_epoch'],
                 batch_flow=['before_batch', 'on_batch', 'after_batch']):

        self.creation_time = datetime.datetime.now()

        self._train_flow = train_flow
        self._batch_flow = batch_flow
        self._epoch_flow = epoch_flow

        self._pipes = pipes

        for pipe in self._pipes:
            pipe.trainer = self

        self.status = collections.OrderedDict()

        self._epoch = 0
        self.status["epoch"] = 0
        self._iteration = 0
        self.status["iteration"] = 0

        # self._best_metrics = None

        self._run_pipes('on_init')


    def _traverse_pipes(self, stage, action='run'):
        priorities = []
        for pipe in self._pipes:
            if hasattr(pipe, 'priority'):
                if isinstance(pipe.priority, dict):
                    if stage in pipe.priority:
                        priorities.append(-pipe.priority[stage])
                    else:
                        priorities.append(0)
                else:
                    priorities.append(-pipe.priority)
            else:
                priorities.append(0)
        priorities = numpy.array(priorities)

        order = priorities.argsort(kind='stable')

        res = []
        for index in order:
            if hasattr(self._pipes[index], stage):
                if action == 'run':
                    getattr(self._pipes[index], stage)()
                elif action == 'view':
                    doc = 'Whoopsy... No description provided'
                    name = self._pipes[index].__class__.__name__ + '.' + getattr(self._pipes[index], stage).__name__
                    if hasattr(getattr(self._pipes[index], stage), '__doc__'):
                        if getattr(self._pipes[index], stage).__doc__ is not None:
                            doc = getattr(self._pipes[index], stage).__doc__

                    res.append((priorities[index], name, ' '.join(doc.split())))

        return res


    def _run_pipes(self, stage):
        self._traverse_pipes(stage, action='run')


    def _view_pipes(self, stage):
        self._traverse_pipes(stage, action='view')


    def _traverse_train(self,
                  n_epochs=None,
                  action='view'):

        if action == 'view':
            n_epochs = 1

        if n_epochs is not None:
            self._n_epochs = n_epochs
            self.status["n_epochs"] = n_epochs

        res = []
        for stage in self._train_flow:
            res.extend(self._traverse_pipes(stage, action=action))

        return res


    def _traverse_epoch(self,
                  mode='valid',
                  subset='valid',
                  n_iterations=None,
                  action='view'):

        if action == 'view':
            n_iterations = 1

        if mode == 'train':
            self._epoch += 1
            self.status['epoch'] += 1

        self._epoch_iteration = 0

        self._mode = mode
        self._subset = subset

        self.status["mode"] = mode
        self.status["subset"] = subset

        self._n_iterations = n_iterations

        res = []
        for stage in self._epoch_flow:
            res.extend(self._traverse_pipes(stage, action=action))
        return res


    def _traverse_batch(self, action='view'):

        if action != 'view':
            if self._mode == 'train':
                self._iteration += 1
                self.status['iteration'] += 1

            self._epoch_iteration += 1

        res = []
        for stage in self._batch_flow:
            res.extend(self._traverse_pipes(stage, action=action))

        return res


    def run_train(self, n_epochs=None):
        self._traverse_train(n_epochs=n_epochs, action='run')


    def run_epoch(self,
                  mode='valid',
                  subset='valid',
                  n_iterations=None):
        self._traverse_epoch(mode=mode, subset=subset, n_iterations=n_iterations, action='run')


    def run_batch(self):
        self._traverse_batch(action='run')


    def view_train(self):
        res = pandas.DataFrame(self._traverse_train(action='view'))
        res.columns = ['priority', 'action', 'description']
        return res


    def view_epoch(self):
        res = pandas.DataFrame(self._traverse_epoch(action='view'))
        res.columns = ['priority', 'action', 'description']
        return res


    def view_batch(self):
        res = pandas.DataFrame(self._traverse_batch(action='view'))
        res.columns = ['priority', 'action', 'description']
        return res


    def view_pipeline(self):
        res = []
        for index in range(len(self._pipes)):
            res.append([self._pipes[index].__class__.__name__])

        return pandas.DataFrame(res)
