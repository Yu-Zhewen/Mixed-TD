import functools
import numpy as np


def get_conv2d_macs(module, ofm_size):
    macs = ofm_size[0] * ofm_size[2] * ofm_size[3] * get_conv2d_params(module)
    return macs

def get_conv2d_params(module):
    return module.weight.nelement()

def get_low_rank_macs(low_rank_wrapper, rank=None):
    if rank == None:
        rank = low_rank_wrapper.rank
    low_rank_macs = low_rank_wrapper.get_per_rank_macs() * rank
    return low_rank_macs

def get_factors(n):
    return np.sort(list(set(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))

class DecompositionWrapper():
    def __init__(self, original_module, original_ofm_size):
        self.original_module = original_module
        self.original_ofm_size = original_ofm_size

        assert self.original_module.groups == 1

    def get_original_module_macs(self):
        return get_conv2d_macs(self.original_module, self.original_ofm_size)

    def get_original_module_params(self):
        return get_conv2d_params(self.original_module)