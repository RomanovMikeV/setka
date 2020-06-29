import torch
import numpy as np
import collections.abc as container_abcs
from itertools import compress


class CollectionOperator:
    """
    Class performs operations with single dataset objects and batches of objects.
    Supports following operations:
        * Collation of single elements into batch
        * Split batch into single data elements
        * Split one element from batch using it index
        * Split group of elements from batch using binary mask
        * Split group of elements from batch using their indices
        * Detach all torch.Tensors from current graph
        * Transfer all torch.Tensors from batch to specific device or dtype

    This class support operations with python structures of arbitary nesting depth, performs all operations recursively
    Support is guaranteed for the following structures for data element content:
        * All sequence classes (e.g. list, tuple)
        * All abc.Mapping classes (e.g. dict, OrderedDict)
        * Torch tensors (torch.Tensor)
        * Numpy arrays (np.ndarray)
        * Python numeric variables (e.g. float, int)

    Attributes:
        int_classes (list): list of types with integer data type
        string_classes (list): list of types that contain string data
        leaf_types (list): list of types

    For further details consider taking a look examples/CollectionOperator.ipynb

    Arguments:
        soft_collate_fn (bool, optional): If true and it is impossible to stack data element attribute into single batch
            (e.g. each element attribute in list has different shape), no errors are thrown. Instead, element attribute
            represented as list of tensors in batch. If false, collate_fn behaviour is equal to pytorch DataLoader.
        collate_fn_conversions (bool optional): Affects collate_fn behaviour. If true, elements are converted to
            torch.Tensor whenever it is possible. If false, container types are preserved
    """
    int_classes = (int,)
    string_classes = (str,)
    leaf_types = (torch.Tensor, np.ndarray)
    primitives = (int, float, str)

    def __init__(self, soft_collate_fn=False, collate_fn_conversions=True):
        self.soft_collate_fn = soft_collate_fn
        self.collate_fn_conversions = collate_fn_conversions

    def collate_fn(self, batch, mode='concat'):
        """
        By given sequence of data elements returns stacked batch.

        Arguments:
            batch: sequence of elements

        Returns:

        :return:
        """
        # Significant code part persisted from
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        #
        # Change: if container with tensors stacking is impossible, collate_fn doesn`t raise
        # error. Instead, return list of tensors
        
        new_batch = []
        for item in batch:
            if item is not None:
                new_batch.append(item)
        batch=new_batch
        
        elem = batch[0]
        elem_type = type(elem)

        if isinstance(elem, torch.Tensor):
            try:
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                return torch.stack(batch, dim=0, out=out)
            except Exception as e:
                if self.soft_collate_fn:
                    return list(batch)
                else:
                    raise e
        elif elem_type.__module__ == 'numpy':
            if type(elem).__name__ == 'ndarray':
                return self.collate_fn([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch) if self.collate_fn_conversions else batch
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float32) if self.collate_fn_conversions else batch
        elif isinstance(elem, CollectionOperator.int_classes):
            return torch.tensor(batch) if self.collate_fn_conversions else batch
        elif isinstance(elem, CollectionOperator.string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return elem_type({key: self.collate_fn([d[key] for d in batch]) for key in elem})
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.collate_fn(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return elem_type([self.collate_fn(samples) for samples in transposed])

        raise TypeError('Not understood: ' + type(elem).__name__)

    @staticmethod
    def _is_leaf(element):
        if isinstance(element, CollectionOperator.leaf_types):
            return True
        if isinstance(element, container_abcs.Sequence) and isinstance(element[0], CollectionOperator.primitives):
            return True
        return False

    @staticmethod
    def _get_ik(el, i, key):
        return el[i][key] if key is not None else el[i]

    @staticmethod
    def _set_ik(el, i, key, val):
        if key is not None:
            el[i][key] = val
        else:
            el[i] = val

    @staticmethod
    def _apply_mask(el, mask):
        if isinstance(mask, int):
            return el[mask]
        dt = mask.dtype
        if (not isinstance(mask, torch.Tensor)) or (dt not in [torch.long, torch.bool]) or len(mask.shape) > 1:
            raise ValueError('Mask should be 1D long or bool tensor')
        if isinstance(el, torch.Tensor):
            return el[mask]
        mask = mask.detach().cpu().numpy()
        if isinstance(el, np.ndarray):
            return el[mask]
        # Element assumed to be list
        if dt == torch.bool:
            return list(compress(el, list(mask)))
        return list(map(el.__getitem__, mask))

    @staticmethod
    def _split(elements, result, key=None, index=None):
        if CollectionOperator._is_leaf(elements):
            if index is None:
                assert len(elements) == len(result)
                for i in range(len(elements)):
                    CollectionOperator._set_ik(result, i, key, elements[i])
            else:
                masked = CollectionOperator._apply_mask(elements, index)
                CollectionOperator._set_ik(result, 0, key, masked)
        elif isinstance(elements, container_abcs.Mapping):
            for i in range(len(result)):
                CollectionOperator._set_ik(result, i, key, {})
            for subkey in elements:
                CollectionOperator._split(
                    elements[subkey],
                    [CollectionOperator._get_ik(result, i, key) for i in range(len(result))],
                    subkey, index)
        elif isinstance(elements, container_abcs.Sequence):
            for i in range(len(result)):
                CollectionOperator._set_ik(result, i, key, [None] * len(elements))
            for i in range(len(elements)):
                CollectionOperator._split(
                    elements[i],
                    [CollectionOperator._get_ik(result, j, key) for j in range(len(result))],
                    i, index)
        else:
            raise ValueError('Cannot split type: ' + str(type(elements)))

    @staticmethod
    def split(elements, batch_size=None):
        """
        By given batch of elements splits it into sequence of separate elements

        Arguments:
             elements: input batch

        Returns:
            seq: sequence of separated batch elements
        """
        if batch_size is None:
            cur = elements
            while not CollectionOperator._is_leaf(cur):
                if isinstance(cur, container_abcs.Sequence):
                    cur = cur[0]
                elif isinstance(cur, container_abcs.Mapping):
                    cur = cur[list(cur.keys())[0]]
                else:
                    raise ValueError('Couldn`t determine batch size from collection')
            batch_size = len(cur)
            
        result = [None] * batch_size
        CollectionOperator._split(elements, result)
        return result

    @staticmethod
    def split_index(elements, index):
        """
        By given batch of elements return single element with specified index
        """
        result = [None]
        CollectionOperator._split(elements, result, index=index)
        return result

    @staticmethod
    def detach(elements):
        """
        Detaches all tensor elements in collection, returns detached data
        """
        if isinstance(elements, (tuple, list)):
            output = []
            for i in range(len(elements)):
                output.append(CollectionOperator.detach(elements[i]))
            return type(elements)(output)
        if isinstance(elements, dict):
            output = {}
            for k in elements:
                output[k] = CollectionOperator.detach(elements[k])
            return output
        if isinstance(elements, torch.Tensor):
            return elements.detach()

        return elements

    @staticmethod
    def to(elements, *args, **kwargs):
        """
        Change parameters (e.g. device, data type) of tensors from data collection
        """
        if isinstance(elements, (tuple, list)):
            output = []
            for i in range(len(elements)):
                output.append(CollectionOperator.to(elements[i], *args, **kwargs))
            return type(elements)(output)
        if isinstance(elements, dict):
            output = {}
            for k in elements:
                output[k] = CollectionOperator.to(elements[k], *args, **kwargs)
            return output
        if isinstance(elements, torch.Tensor):
            return elements.to(*args, **kwargs)

        return elements
