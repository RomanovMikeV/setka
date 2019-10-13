import copy
import torch
import torch.utils

from .Callback import Callback

class ComputeMetrics(Callback):
    '''
    This callback computes metrics when the validation is
    performed. The metrics are updated on batch end. The parameter
    ```steps_to_compute``` specifies on how many batches the
    metrics are computed. When the epoch ends -- final metrics
    are computed. The history is flushed when the
    epoch starts.

    Note: list of metrics should contain callables. Each of the
        callables my return either one value or two values. If two
        values are returned (or two numpy arrays of the same shape) --
        the first value is treated as enumerator(s), the second is treated as a
        denominator(s).
        When the metric is requested -- the new enumerator(s) and
        denomirator(s) are computed. Overall enumerators and denominators
        are updated (new values added to the accumulated ones).
        After that there are two options
        for computing average for each of the metrics in arrays.
        First option is to first sum all the accumulated enumerators
        and denominators and then to divide one by another
        (case of ```divide_first``` flag for the metric
        is set to False). The second option is to first divide
        each of the enumerators by its denominator and then average
        (case of ```divide_first``` for the metric is set to True).
    '''
    def __init__(self,
                 metrics=None,
                 divide_first=None,
                 steps_to_compute=1):

        '''
        Constructor.

        Args:
            metrics (list): list of callables for metrics
                computation. The callable should return two values:
                enumerator and denominator.

            steps_to_compute (int): how many steps to perform
                before metrics update.
        '''

        self.steps_to_compute = steps_to_compute
        self.metrics = metrics
        self.names = []

        if self.metrics is None:
            self.names = []
            self.metrics = []

        else:
            for index in range(len(self.metrics)):
                self.names.append(self.metrics[index].__name__)

        if divide_first is None:
            self.divide_first = [True] * len(self.metrics)
        else:
            self.divide_first = divide_first

        self.steps = 0

        self.avg_values = {}

        # self.inputs = []
        # self.outputs = []

        # self.enumerators = []
        # self.denominators = []


    def reset(self):
        self.enumerators = []
        self.denominators = []

        for metric in self.metrics:
            self.enumerators.append(None)
            self.denominators.append(None)


    def on_epoch_begin(self):
        # clear all cache
        self.reset()

        if hasattr(self, 'inputs'):
            del self.inputs

        if hasattr(self, 'outputs'):
            del self.outputs

        self.inputs = []
        self.outputs = []

        self.steps = 0


    @staticmethod
    def preprocess_collection(input):
        if isinstance(input[0], (list, tuple)):
            res = []
            for list_index in range(len(input[0])):
                one = []
                for index in range(len(input)):
                    one.append(input[index][list_index])
                one = torch.cat(one, dim=0)

                res.append(one)

            return res

        elif isinstance(input[0], dict):
            res = {}
            for key in input[0]:
                one = []
                for index in range(len(input)):
                    one.append(input[index][key])
                one = torch.cat(one, dim=0)

                res[key] = one

            return res

        else:
            return torch.cat(input, dim=0)


    def evaluate(self):
        self.inputs = self.preprocess_collection(self.inputs)
        self.outputs = self.preprocess_collection(self.outputs)

        for index in range(len(self.metrics)):
            res = self.metrics[index](self.outputs, self.inputs)

            if isinstance(res, (list, tuple)):
                if len(res) == 2:
                    if isinstance(res[0], torch.Tensor):
                        enum = res[0].detach()
                    else:
                        enum = torch.tensor(res[0])

                    if isinstance(res[1], torch.Tensor):
                        denom = res[1].detach()
                    else:
                        denom = torch.tensor(res[1])
            else:
                if isinstance(res, torch.Tensor):
                    enum = res.detach()
                else:
                    enum = torch.tensor(res)
                denom = torch.ones(enum.shape)

            enum = enum.detach().cpu().numpy()
            denom = denom.detach().cpu().numpy()

            if self.enumerators[index] is None:
                self.enumerators[index] = enum
                self.denominators[index] = denom
            else:
                self.enumerators[index] += enum
                self.denominators[index] += denom

        self.avg_values.clear()
        for index in range(len(self.enumerators)):
            if self.divide_first[index]:
                self.avg_values[self.names[index]] = (
                    (self.enumerators[index] /
                     (self.denominators[index] + 1.0e-12)).mean())
            else:
                self.avg_values[self.names[index]] = (
                        self.enumerators[index].sum() /
                        (self.denominators[index].sum() + 1.0e-12))

        del self.inputs
        del self.outputs

        self.inputs = []
        self.outputs = []

        self.steps = 0

        for x in self.avg_values:
            self.trainer.status[x] = self.avg_values[x]


    @staticmethod
    def detach_collection(input):
        if isinstance(input, (list, tuple)):
            return [x.detach() for x in input]
        elif isinstance(input, dict):
            return {k: v for k, v in input.items()}
        else:
            return input.detach()


    def on_batch_end(self):
        if self.trainer._mode in ('train', 'valid'):

            self.steps += 1

            self.outputs.append(self.detach_collection(self.trainer._output))

            self.inputs.append(self.detach_collection(self.trainer._input))

            if self.steps >= self.steps_to_compute:
                self.evaluate()


    def on_epoch_end(self):
        if self.trainer._mode == 'valid':
            if self.steps != 0:
                self.evaluate()

            if not hasattr(self.trainer, '_metrics'):
                self.trainer._metrics = {}
            self.trainer._metrics[self.trainer._subset] = copy.deepcopy(self.avg_values)
        self.avg_values.clear()
        self.reset()
