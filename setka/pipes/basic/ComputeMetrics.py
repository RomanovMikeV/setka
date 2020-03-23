import copy
import torch
import torch.utils

from setka.pipes.Pipe import Pipe
from setka.base import CollectionOperator


class ComputeMetrics(Pipe):
    """
    This pipe computes metrics when the validation is
    performed. The metrics are updated on batch end. The parameter
    ```steps_to_compute``` specifies on how many batches the
    metrics are computed. When the epoch ends -- final metrics
    are computed. The history is flushed when the
    epoch starts.

    Note: list of metrics should contain callable. Each of the
        callable may return either one value or two values. If two
        values are returned (or two numpy arrays of the same shape) --
        the first value is treated as enumerator(s), the second is treated as a
        denominator(s).
        When the metric is requested -- the new enumerator(s) and
        denominator(s) are computed. Overall enumerators and denominators
        are updated (new values added to the accumulated ones).
        After that there are two options
        for computing average for each of the metrics in arrays.
        First option is to first sum all the accumulated enumerators
        and denominators and then to divide one by another
        (case of ```divide_first``` flag for the metric
        is set to False). The second option is to first divide
        each of the enumerators by its denominator and then average
        (case of ```divide_first``` for the metric is set to True).

    Args:
        metrics (list of callable, required): list of metric functions to compute.
        divide_first (list of bool, not required): list of flags indicating that the
            division should be performed before the reduce.
        steps_to_compute (int): indicates how often the metrics values should be updated
    """
    def __init__(self, metrics, divide_first=None, steps_to_compute=1):
        super(ComputeMetrics, self).__init__()
        self.steps_to_compute = steps_to_compute
        self.metrics = metrics
        self.names = []
        self.eps = 1e-12

        for index in range(len(self.metrics)):
            self.names.append(self.metrics[index].__name__)

        if divide_first is None:
            self.divide_first = [True] * len(self.metrics)
        else:
            self.divide_first = divide_first

        self.steps = 0
        self.avg_values = {}
        self.enumerators = None
        self.denominators = None
        self.inputs = None
        self.outputs = None

    def reset(self):
        self.enumerators = [None] * len(self.metrics)
        self.denominators = [None] * len(self.metrics)

    def before_epoch(self):
        """
        Initializes the storage for the metrics.
        """
        self.reset()

        if hasattr(self, 'inputs'):
            del self.inputs

        if hasattr(self, 'outputs'):
            del self.outputs

        self.inputs = []
        self.outputs = []

        self.steps = 0
        self.trainer._avg_metrics = {}

    def evaluate(self):        
        self.inputs = self.trainer.collection_op.collate_fn(self.inputs)
        self.outputs = self.trainer.collection_op.collate_fn(self.outputs)

        for index in range(len(self.metrics)):
            with torch.no_grad():
                res = self.metrics[index](self.outputs, self.inputs)

            if isinstance(res, (list, tuple)):
                if len(res) != 2:
                    raise ValueError("Metric should return list or tuple of length 2: "
                                     "numerator and denominator of result")

                enum = torch.as_tensor(res[0]).detach().cpu().numpy()
                denom = torch.as_tensor(res[1]).detach().cpu().numpy()
            else:
                enum = torch.as_tensor(res).detach().cpu().numpy()
                denom = torch.ones(enum.shape)

            if self.enumerators[index] is None:
                self.enumerators[index] = enum
                self.denominators[index] = denom
            else:
                self.enumerators[index] += enum
                self.denominators[index] += denom

        self.avg_values.clear()

        for index in range(len(self.enumerators)):
            if self.divide_first[index]:
                self.avg_values[self.names[index]] = float((self.enumerators[index] / (self.denominators[index] + self.eps)).mean())
            else:
                self.avg_values[self.names[index]] = float(self.enumerators[index].sum() / (self.denominators[index].sum() + self.eps))

        del self.inputs
        del self.outputs

        self.inputs = []
        self.outputs = []
        self.steps = 0

        for x in self.avg_values:
            self.trainer.status[x] = self.avg_values[x]
            self.trainer._avg_metrics[x] = self.avg_values[x]

    def after_batch(self):
        """
        Updates storage and evaluates the metrics.
        """
        if self.trainer._mode in ('train', 'valid'):
            self.steps += 1
            self.outputs.extend(
                self.trainer.collection_op.split(
                    self.trainer.collection_op.detach(self.trainer._output)))
            self.inputs.extend(
                self.trainer.collection_op.split(
                    self.trainer.collection_op.detach(self.trainer._input)))
        
            if self.steps >= self.steps_to_compute:
                self.evaluate()

    def after_epoch(self):
        """
        Stores final metrics in the 'self.trainer._metrics' and resets the
        pipe.
        """
        if self.trainer._mode == 'valid':
            if self.steps != 0:
                self.evaluate()

            if not hasattr(self.trainer, '_metrics'):
                self.trainer._metrics = {}
            self.trainer._metrics[self.trainer._subset] = copy.deepcopy(self.avg_values)
        self.avg_values.clear()
        self.reset()
        self.trainer._avg_metrics = {}
