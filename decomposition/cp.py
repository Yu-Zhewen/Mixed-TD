import itertools
import tensorly as tl
import torch
import torch.nn as nn

from decomposition.common import DecompositionWrapper
from tensorly.base import unfold
from tensorly.cp_tensor import cp_to_tensor, CPTensor
from tensorly.decomposition import parafac
from torch import Tensor

def enumerate_cp_candidates(original_module, original_ofm_size):
    in_channels = original_module.in_channels
    out_channels = original_module.out_channels

    candidates = []
    candidates.append(CPDecompositionWrapper(original_module, original_ofm_size, group_c=1, group_f=1))

    #for group in get_factors(in_channels)[1:]:
    #    candidates.append(CPDecompositionWrapper(original_module, original_ofm_size, group_c=group, group_f=1))
    #
    #for group in get_factors(out_channels)[1:]:
    #    candidates.append(CPDecompositionWrapper(original_module, original_ofm_size, group_c=1, group_f=group))

    return candidates

def initialize_cp(tensor, rank, svd='numpy_svd', random_state=None):
    rng = tl.check_random_state(random_state)

    try:
        svd_fun = tl.SVD_FUNS[svd]
    except KeyError:
        message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                    svd, tl.get_backend(), tl.SVD_FUNS)
        raise ValueError(message)

    factors = []
    for mode in range(tl.ndim(tensor)):
        U, S, V = svd_fun(unfold(tensor, mode), n_eigenvecs=min(rank, tensor.shape[mode]))
        if mode == 0:
            idx = min(rank, tl.shape(S)[0])
            U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

        if tensor.shape[mode] < rank:
            random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
            U = tl.concatenate([U, random_part], axis=1)

        factors.append(U[:, :rank])

    kt = CPTensor((None, factors))

    return kt

class CPDecomposition(nn.Module):
    def __init__(self, low_rank_conv1, low_rank_conv2, low_rank_conv3, low_rank_conv4):
        super(CPDecomposition, self).__init__()
        self.low_rank_conv1 = low_rank_conv1
        self.low_rank_conv2 = low_rank_conv2
        self.low_rank_conv3 = low_rank_conv3
        self.low_rank_conv4 = low_rank_conv4


    def forward(self, x: Tensor) -> Tensor:
        intermediate_1 = self.low_rank_conv1(x)
        intermediate_2 = self.low_rank_conv2(intermediate_1)
        intermediate_3 = self.low_rank_conv3(intermediate_2)
        out = self.low_rank_conv4(intermediate_3)
        return out

class CPDecompositionWrapper(DecompositionWrapper):
    def __init__(self, original_module, original_ofm_size, group_c=1, group_f=1):
        super(CPDecompositionWrapper, self).__init__(original_module, original_ofm_size)
        self.group_c = group_c 
        self.group_f = group_f
        self.seed = 0

    def get_max_rank(self):
        dims = list(self.original_module.weight.size())
        dims[0] = dims[0] // self.group_f
        dims[1] = dims[1] // self.group_c
        dims.remove(max(dims))
        max_rank = int(np.prod(dims))
        return max_rank

    def get_rank_dims(self):
        return [0]

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[2] \
                        * self.original_ofm_size[3] \
                        * (self.original_module.out_channels * self.group_c
                            + self.original_module.kernel_size[1] * self.group_c * self.group_f
                            + self.original_module.kernel_size[0] * self.group_c * self.group_f * self.original_module.stride[1]
                            + self.original_module.in_channels * self.group_f * self.original_module.stride[0] * self.original_module.stride[1])

        return per_rank_macs
    
    def get_per_rank_params(self):
        per_rank_macs = self.original_module.out_channels * self.group_c\
                        + self.original_module.kernel_size[1] * self.group_c * self.group_f \
                        + self.original_module.kernel_size[0] * self.group_c * self.group_f \
                        + self.original_module.in_channels * self.group_f

        return per_rank_params

    def initialise_low_rank_module(self, rank):
        assert rank > 0
        self.rank = rank

        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        int(self.group_c*self.group_f*self.rank), 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0, 
                                        groups=self.group_c, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(int(self.group_c*self.group_f*self.rank), 
                                        int(self.group_c*self.group_f*self.rank),
                                        kernel_size=(self.original_module.kernel_size[0], 1), 
                                        stride=(self.original_module.stride[0], 1), 
                                        padding=(self.original_module.padding[0], 0), 
                                        groups=int(self.group_c*self.group_f*self.rank), 
                                        bias=False, 
                                        dilation=1)
        
        self.low_rank_conv3 = nn.Conv2d(int(self.group_c*self.group_f*self.rank),
                                        int(self.group_c*self.group_f*self.rank), 
                                        kernel_size=(1, self.original_module.kernel_size[1]), 
                                        stride=(1, self.original_module.stride[1]), 
                                        padding=(0, self.original_module.padding[1]), 
                                        groups=int(self.group_c*self.group_f*self.rank), 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv4 = nn.Conv2d(int(self.group_c*self.group_f*self.rank), 
                                        self.original_module.out_channels, 
                                        kernel_size=1,
                                        stride=1, 
                                        padding=0, 
                                        groups=self.group_f, 
                                        bias=(self.original_module.bias!=None), 
                                        dilation=1)

    def generate_low_rank_weight(self):
        original_weight = self.original_module.weight.detach().clone()
        tl.set_backend('pytorch')

        original_weight = torch.reshape(original_weight,(self.group_f,
                                                        self.original_module.out_channels // self.group_f,
                                                        self.group_c,   
                                                        self.original_module.in_channels // self.group_c,
                                                        self.original_module.kernel_size[0],
                                                        self.original_module.kernel_size[1]))    
        original_weight = torch.permute(original_weight, (0,2,1,3,4,5))                                                                                                                                                                                                                                                                  

        a2 = torch.empty((self.group_f, self.group_c, self.original_module.in_channels // self.group_c, self.rank))
        a3 = torch.empty((self.group_f, self.group_c, self.original_module.kernel_size[0], self.rank))
        a4 = torch.empty((self.group_f, self.group_c, self.original_module.kernel_size[1], self.rank))
        a1 = torch.empty((self.group_f, self.group_c, self.original_module.out_channels // self.group_f, self.rank))
        approximated_weight_array = torch.empty((self.group_f, self.group_c, self.original_module.out_channels // self.group_f, self.original_module.in_channels // self.group_c, self.original_module.kernel_size[0], self.original_module.kernel_size[1]))
        for i, j in itertools.product(range(self.group_f), range(self.group_c)):
            cp_tensor = initialize_cp(original_weight[i,j], self.rank, svd="truncated_svd", random_state=self.seed)
            cp_tensor = parafac(original_weight[i,j], rank=self.rank, init=cp_tensor, svd="truncated_svd",random_state=self.seed)
            a1[i,j], a2[i,j], a3[i,j], a4[i,j] = cp_tensor[1]
            approximated_weight_array[i,j] = cp_to_tensor(cp_tensor)
        assert not approximated_weight_array.isnan().any()

        low_rank_weight_array1 = torch.permute(a2, (0, 1, 3, 2))
        low_rank_weight_array1 = torch.reshape(low_rank_weight_array1, (int(self.group_f*self.group_c*self.rank),
                                                                    self.original_module.in_channels // self.group_c, 
                                                                    1,
                                                                    1))
        low_rank_weight_array2 = torch.permute(a3, (0, 1, 3, 2))
        low_rank_weight_array2 = torch.reshape(low_rank_weight_array2, (int(self.group_f*self.group_c*self.rank),
                                                                    1, 
                                                                    self.original_module.kernel_size[0],
                                                                    1))
        low_rank_weight_array3 = torch.permute(a4, (0, 1, 3, 2))
        low_rank_weight_array3 = torch.reshape(low_rank_weight_array3, (int(self.group_f*self.group_c*self.rank),
                                                                    1, 
                                                                    1,
                                                                    self.original_module.kernel_size[1]))
        low_rank_weight_array4 = torch.permute(a1, (0, 2, 1, 3))
        low_rank_weight_array4 = torch.reshape(low_rank_weight_array4, (self.original_module.out_channels, 
                                                                    int(self.group_c*self.rank),
                                                                    1,
                                                                    1))

        if hasattr(self, "low_rank_conv1"):
            self.low_rank_conv1.weight.data.copy_(low_rank_weight_array1)
            self.low_rank_conv2.weight.data.copy_(low_rank_weight_array2)
            self.low_rank_conv3.weight.data.copy_(low_rank_weight_array3)
            self.low_rank_conv4.weight.data.copy_(low_rank_weight_array4)

            if self.original_module.bias != None:
                self.low_rank_conv4.bias.data.copy_(self.original_module.bias.detach().clone())

        approximated_weight_array = torch.permute(approximated_weight_array, (0,2,1,3,4,5))   
        approximated_weight_array = torch.reshape(approximated_weight_array, (self.original_module.out_channels,
                                                                            self.original_module.in_channels,
                                                                            self.original_module.kernel_size[0],
                                                                            self.original_module.kernel_size[1]))
        return approximated_weight_array

    def export_decomposition(self):
        decomposed_module = CPDecomposition(self.low_rank_conv1, self.low_rank_conv2, self.low_rank_conv3, self.low_rank_conv4)
        if self.original_module.weight.is_cuda:
            decomposed_module.cuda()
        return decomposed_module