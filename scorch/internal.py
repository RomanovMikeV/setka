import numpy
import torch
from torch._six import string_classes, int_classes
import collections
import re

_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""


def default_collate(batch):
    r"""
    Alternative to the default pytorch collator that ignores Nones if
    they appear in the batch.
    """

    new_batch = []
    for index in range(len(batch)):
        if batch[index] is not None:
            new_batch.append(batch[index])
    batch = new_batch

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def show(tb_writer, to_show, epoch):

    type_writers = {
        'images': tb_writer.add_image,
        'texts': tb_writer.add_text,
        'audios': tb_writer.add_audio,
        'figures': (lambda x, y, z: tb_writer.add_figure(x, y, z)),
        'graphs': tb_writer.add_graph,
        'embeddings': tb_writer.add_embedding}

    for type in type_writers:
        if type in to_show:
            for desc in to_show[type]:
                type_writers[type](desc, to_show[type][desc], str(epoch))


class DataSetWrapper():
    '''
    This is the wrapper for a dataset class.

    This class should have the following methods:
    __init__, __len__, __getitem__.
    '''

    def __init__(self, dataset, mode='train'):
        self.dataset = dataset
        self.mode = mode

        self.order = numpy.arange(self.dataset.get_len(mode))

        if self.mode == 'train':
            self.order = numpy.random.permutation(self.dataset.get_len(self.mode))

    def __len__(self):
        return self.dataset.get_len(self.mode)

    def __getitem__(self, index):
        real_index = self.order[index]
        return self.dataset.get_item(real_index, self.mode)

    def shuffle(self):
        if self.mode == 'train':
            self.order = numpy.random.permutation(self.dataset.get_len(self.mode))
