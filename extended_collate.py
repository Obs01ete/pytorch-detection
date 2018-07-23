import torch
import collections
import re
from torch._six import string_classes, int_classes
from torch.utils.data.dataloader import numpy_type_map, _use_shared_memory

def extended_collate(batch, depth=0, collate_first_n=2):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Dmitry Khzibullin: iteratively collate only first 2 items: image and target.
    """

    depth += 1

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
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
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = [v for v in zip(*batch)]
        if depth == 1: # collate image and target only
            num_first = collate_first_n
        else:
            num_first = len(transposed)
        transposed_process = transposed[:num_first]
        transposed_noprocess = transposed[num_first:]
        collated = [extended_collate(samples, depth=depth) for samples in transposed_process]
        merged = [*collated, *transposed_noprocess]
        return merged
    else:
        return batch

