import copy


class Dataset:
    """
    Dataset class, where all the information about the dataset is
    collected. It contains all the information about the dataset.
    The dataset may be split into the subsets (that are usually
    referred to as ```train```, ```valid``` and ```test```, but
    you may make much more different modes for your needs).

    The DataSet has the following methods:
        * __init__ -- constructor, where the index collection
            is ususally performed. You need to define it.
            In the original constuctor the ```subset``` is set
            to ```train```.

        * getlen -- function that gets the size of the subset of the dataset.
            This function gets as argument the subset ID (hashable).
            You need to define this function in your class.

        * getitem -- function that retrieves the item form the dataset.
            This function gets as arguments the subset ID (hashable) and the
            ID of the element in the subset (index).
            You need to define this function in your class.


        * getitem -- function that selects

        * __len__ -- function that is called when the ```len()```
            operator is called and returns the volume of the
            current subset. Predefined, you do not need to redefine it.

        * __getitem__ -- function that gets the item of the dataset
            by its index. Predefined, you do not need to redefine it.
    """

    def __init__(self):
        pass

    def getitem(self, subset, index):
        pass

    def getlen(self, subset):
        pass

    def __getitem__(self, *args):
        args = args[0]
        if type(args) is tuple:
            assert(len(args) <= 2)
            assert(len(args) > 0)

            return self.getitem(args[0], args[1])
        else:
            if not hasattr(self, 'subset'):
                sliced_dataset = copy.copy(self)
                sliced_dataset.subset = args
                return sliced_dataset

            else:
                return self.getitem(self.subset, args)

    def __len__(self):
        assert(hasattr(self, 'subset'))
        return self.getlen(self.subset)
