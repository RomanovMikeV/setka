import torch

class OptimizerSwitch():
    '''
    This class contains optimizer and the module that this optimizer should
    optimize.
    '''
    def __init__(self, train_module, optimizer, is_active=True, **kwargs):
        self.optimizer = optimizer(train_module.parameters(), **kwargs)
        self.module = train_module
        self.active = is_active


class DataSet():
    '''
    Dataset index class, where all the information about the dataset is
    collected. This class should have only the constructor, where the dataset
    structure is described. It is recommended to store the class info here
    instead of the DataSet class as in the other case the amount of memery
    consumed by dataset info may triple.

    You may redefine in this class the following methods:
    ```get_len```, ```get_item```.
    '''

    def __init__(self):
        '''
        Class constructor. Please overload this function definition when making
        your dataset.
        '''
        pass

    def get_len(self, mode='train'):
        '''
        Function that gets length of dataset's subset specified in ```mode```.
        By default is

        ```
        return len(self.inputs[mode])
        ```

        Args:
            mode (str): subset of the dataset to get the length of.

        Returns:
            int, number of elements in the subset.
        '''

        return len(self.data[mode])

    def get_item(self, index, mode='train'):
        '''
        Function that gets the item from the dataset's subset specified in
        ```mode```.

        By default it is:
        ```
        datum = self.data[mode][index]
        target = self.labels[mode][index]

        id = mode + "_" + str(index)

        return [datum], [target], id
        ```

        Args:
            index: index of the item to be loaded

            mode (str): subset of the dataset to get the length of.

        Returns:
            list of inputs to the neural network for the item.

            list of targets for the item.

            id (str is preferred) of the item.
        '''

        datum = self.data[mode][index]
        target = self.labels[mode][index]

        id = mode + "_" + str(index)

        return [datum], [target], id


class Network(torch.nn.Module):
    '''
    Base class for your models.

    Your networks should be a subclass of this class.
    In your models you should redefine ```__init__``` function
    to construct the model and ```forward``` function.
    '''
    def __init__(self):
        '''
        Class constructor.

        You should define your network elements here. Network elements should
        be ```torch.nn.Module``` subclasses' objects.

        Args:
            This method may also take extra keyword arguments specified with
            ```--network-args``` keyword in the bash or specified in
            ```network_args``` parameter of a function call.
        '''
        super(Network, self).__init__()


    def forward(self, input):
        '''
        Forward propagation function for your network. By default this function
        returns the input.

        Args:
            input (list): A list of batches of inputs for your Network (most
                likely, you want a list of ```torch.Tensors```).
                Note that those inputs come in batches.

        Returns:
            list: A list of outputs of Network for specified inputs.
        '''

        return input

    def __call__(self, input):
        '''
        Call should duplicate the forward propagation function
        (in some versions of pytorch the __call__ is needed, while in the
        others it is enough to have just forward function).

        Args:
            input (list): Same as ```forward``` function arguments.

        Returns:
            The same result as ```forward``` function.
        '''
        return self.forward(input)


class Socket():
    '''
    Socket base class for your networks. The object of this class
    knows how to deal with your network. It contains such fields and methods as
    optimizers, criterion, metrics, process_result, visualize, scheduling.
    '''
    def __init__(self, model):
        '''
        Class constructor. This is one of the most important
        things that you have to redefine. In this function you should define:

        ```self.optimizers``` -- a list of OptimizerSwitches that will be used
        during the training procedure.

        Args:
            model (base.Network): the model for the socket.
        '''

        self.model = model

        self.epoch = 0
        self.iteration = 0

        self.metrics_valid = {}
        self.metrics_train = {}

    def criterion(self, preds, target):
        '''
        Function that computes the criterion, also known as loss function
        for your network. You should define this function for your network
        in order to proceed.

        Note that the loss function should be computed using only pytorch
        operations as you should be able to do backpropagation.

        Args:
            preds: list of outputs returned by your model.

            target: list of target values for your model.

        Returns:
            value for the criterion function.
        '''
        return self.criterion_f(input, target)

    def metrics(self, preds, target):
        '''
        Optional.

        Function that computes metrics for your model.
        Note that there should be a main metric based
        on which framework will decide which checkpoint
        to mark as best.

        Arguments are the same as in the ```criterion method```.

        By default returns criterion value as main metric.

        Args:
            preds (list): list of outputs returned by your model.

            target (list): list of target values for your model.

        Returns:
            dictionary with metrics for your model.
        '''

        return {'main': self.criterion(preds, target)}

    def process_result(self, input, output, id):
        '''
        Optional.

        Function that processes the outputs of the network and forms
        the understandable result of the network. Used only during the
        testing procedure.

        The results of this function will be packed in pth.tar files
        and saved as inference results of the network.

        By default returns {id: output}.

        Args:
            input (list): List of inputs for the network for one item.

            output (list): list of outputs from the network for one item.

            id (str): an id of the sample. The string is preferred.

        Returns:
            dictionary, containing inference results of the network to be stored
            as a result. It is useful to use the id of the sample as a key.
        '''

        return {id: output}

    def visualize(self, input, output, id):
        '''
        Optional.

        Function that is used during the testing session of the epoch.
        It visualizes how the network works. You may use matplotlib for
        visualization.

        By default returns empty dict.

        Args:
            input (list): List of inputs for the network for one item.

            output (list): list of outputs from the network for one item.

            id (str): an id of the sample. The string is preferred.

        Returns:
            dictionary, containing the following fields:

            ```figures```
            (the value should be the dictionary with matplotlib figures),

            ```images``` (the value should be the dictionary with
            numpy arrays that can be treated as images),

            ```texts``` (the value should be the dictionary with strings),

            ```audios```, ```graphs``` and ```embeddings```.
        '''
        return {}

    def scheduling(self):
        '''
        Optional.

        Function that is called every iteration. You may use it to
        switch between optimizers, change learning rate and do whatever
        you want your model to do from iteration to iteration.

        Has no inputs or outputs, usually just changes parameters of training.

        Does nothing by default.
        '''
        pass
