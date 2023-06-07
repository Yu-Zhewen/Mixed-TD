import torch
import torch.nn as nn

from decomposition.common import DecompositionWrapper, get_factors
from torch import Tensor


def generate_low_rank_wrapper(scheme, original_module, original_ofm_size, groups=-1):
    if scheme == 0:
        return StaticLowRankScheme0(original_module, original_ofm_size)
    elif scheme == 1:
        return StaticLowRankScheme1(original_module, original_ofm_size)
    elif scheme == 2:
        return StaticLowRankScheme2(original_module, original_ofm_size)
    elif scheme == 3:
        return StaticLowRankScheme3(original_module, original_ofm_size)
    elif scheme == 4:
        if groups == 1:
            return StaticLowRankScheme1(original_module, original_ofm_size)
        elif groups == original_module.in_channels:
            return StaticLowRankScheme3(original_module, original_ofm_size)
        else:
            return StaticLowRankScheme4(original_module, original_ofm_size, groups)
    elif scheme == 5:
        if groups == original_module.out_channels:
            return StaticLowRankScheme0(original_module, original_ofm_size)
        else: 
            return StaticLowRankScheme5(original_module, original_ofm_size, groups)
    else:
        assert False

def enumerate_scheme_candidates(original_module, original_ofm_size):
    in_channels = original_module.in_channels
    out_channels = original_module.out_channels
    candidates = []
    candidates.append(generate_low_rank_wrapper(2, original_module, original_ofm_size))
    for group in get_factors(in_channels):
        candidates.append(generate_low_rank_wrapper(4, original_module, original_ofm_size, group))
    for group in get_factors(out_channels):
        candidates.append(generate_low_rank_wrapper(5, original_module, original_ofm_size, group))

    return candidates

class StaticLowRankDecomposition(nn.Module):
    def __init__(self, low_rank_conv1, low_rank_conv2):
        super(StaticLowRankDecomposition, self).__init__()
        self.low_rank_conv1 = low_rank_conv1
        self.low_rank_conv2 = low_rank_conv2

    def forward(self, x: Tensor) -> Tensor:
        intermediate = self.low_rank_conv1(x)
        out = self.low_rank_conv2(intermediate)
        return out

class StaticLowRankDecompositionWrapper(DecompositionWrapper):
    def get_rank_dims(self):
        return [0]

    def decompose_weight(self, unfolded_weight, rank_slice=None):
        if rank_slice == None:
            rank_slice = list(range(self.rank))

        assert len(rank_slice) > 0
        [u, s, vh] = torch.linalg.svd(unfolded_weight, full_matrices=False)

        u_lowrank = u[ : , : , rank_slice]
        s_lowrank = s[ : , rank_slice]
        vh_lowrank = vh[ : , rank_slice, : ]

        u_s_sqrt = torch.zeros_like(u_lowrank)
        vh_s_sqrt = torch.zeros_like(vh_lowrank)

        for i in range(self.groups):
            s_sqrt_diag = torch.diag(torch.sqrt(s_lowrank[i]))
            u_s_sqrt[i] = u_lowrank[i] @ s_sqrt_diag
            vh_s_sqrt[i] = s_sqrt_diag @ vh_lowrank[i]

        return u_s_sqrt, vh_s_sqrt
    
    def generate_low_rank_weight(self, rank_slice=None, quantization_method=0, weight_quantizer=None, weight_width=8, quantise_first=0, model_name=None, conv_layer_index=None):
        if rank_slice == None:
            rank_slice = list(range(self.rank))

        assert len(rank_slice) > 0
        original_weight = self.original_module.weight.detach().clone()
        unfolded_weight = self.unfold_original_weight(original_weight)
        u_s_sqrt, vh_s_sqrt = self.decompose_weight(unfolded_weight, rank_slice)
        low_rank_weight_array1, low_rank_weight_array2 = self.fold_low_rank_weight(u_s_sqrt, vh_s_sqrt)
                
        if hasattr(self, "low_rank_conv1"): # without low-rank module, no need to store the weight, save memory
            self.low_rank_conv1.weight.data.copy_(low_rank_weight_array1)
            self.low_rank_conv2.weight.data.copy_(low_rank_weight_array2)
            if self.original_module.bias != None:
                self.low_rank_conv2.bias.data.copy_(self.original_module.bias.detach().clone())

        approximated_weight_array = self.fold_original_weight(u_s_sqrt @ vh_s_sqrt)
        return approximated_weight_array

    def export_decomposition(self):
        decomposed_module = StaticLowRankDecomposition(self.low_rank_conv1, self.low_rank_conv2)
        if self.original_module.weight.is_cuda:
            decomposed_module.cuda()
        return decomposed_module
                                               
class StaticLowRankScheme2(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme2, self).__init__(original_module, original_ofm_size)
        self.groups = 1
        self.scheme = 2

    def get_max_rank(self):
        max_rank = min(self.original_module.out_channels * self.original_module.kernel_size[0], 
                        self.original_module.in_channels * self.original_module.kernel_size[1])
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[2] \
                        * self.original_ofm_size[3] \
                        * (self.original_module.out_channels * self.original_module.kernel_size[0] 
                            + self.original_module.in_channels * self.original_module.kernel_size[1] * self.original_module.stride[0])

        return per_rank_macs

    def get_per_rank_params(self):
        per_rank_params = self.original_module.out_channels \
                            * self.original_module.kernel_size[0] \
                            + self.original_module.in_channels * self.original_module.kernel_size[1]
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = torch.permute(original_weight_array, (0, 2, 1, 3))
        unfolded_weight_array = torch.reshape(unfolded_weight_array, (1,
                                                                   self.original_module.out_channels * self.original_module.kernel_size[0],
                                                                   self.original_module.in_channels * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = torch.reshape(unfolded_weight_array, (self.original_module.out_channels,
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.in_channels,
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = torch.permute(approximated_weight_array, (0, 2, 1, 3))

        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        self.rank = rank
        assert rank > 0

        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.rank, 
                                        kernel_size=(1, self.original_module.kernel_size[1]), 
                                        stride=(1, self.original_module.stride[1]), 
                                        padding=(0, self.original_module.padding[1]), 
                                        groups=1, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=(self.original_module.kernel_size[0], 1), 
                                        stride=(self.original_module.stride[0], 1), 
                                        padding=(self.original_module.padding[0], 0), 
                                        groups=1, 
                                        bias=(self.original_module.bias!=None), 
                                        dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0

        low_rank_weight_array1 = torch.reshape(vh_s_sqrt, (rank, 
                                                        self.original_module.in_channels, 
                                                        1, 
                                                        self.original_module.kernel_size[1]))

        low_rank_weight_array2 = torch.reshape(u_s_sqrt, (self.original_module.out_channels,
                                                       self.original_module.kernel_size[0], 
                                                       rank, 
                                                       1))
        low_rank_weight_array2 = torch.permute(low_rank_weight_array2, (0, 2 ,1, 3))

        return low_rank_weight_array1, low_rank_weight_array2

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0

        vh_s_sqrt = torch.reshape(low_rank_weight_array1, (1,
                                                        rank,
                                                        self.original_module.in_channels * self.original_module.kernel_size[1]))
        u_s_sqrt = torch.permute(low_rank_weight_array2, (0, 2 ,1, 3))
        u_s_sqrt = torch.reshape(u_s_sqrt, (1,
                                         self.original_module.out_channels * self.original_module.kernel_size[0],
                                         rank))
        return u_s_sqrt, vh_s_sqrt

class StaticLowRankScheme4(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size, groups):
        super(StaticLowRankScheme4, self).__init__(original_module, original_ofm_size)
        self.groups = groups
        self.scheme = 4

    def get_max_rank(self):
        max_rank = min(self.original_module.out_channels, 
                        int(self.original_module.in_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[2] \
                        * self.original_ofm_size[3] \
                        * self.groups \
                        * (self.original_module.out_channels 
                            + int(self.original_module.in_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_macs
    
    def get_per_rank_params(self):
        per_rank_params = self.groups \
                            * (self.original_module.out_channels 
                                + int(self.original_module.in_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = torch.reshape(original_weight_array, (self.original_module.out_channels,
                                                                   self.groups,
                                                                   int(self.original_module.in_channels / self.groups),
                                                                   self.original_module.kernel_size[0],
                                                                   self.original_module.kernel_size[1]))
        unfolded_weight_array = torch.permute(unfolded_weight_array, (1, 0, 2, 3, 4))
        unfolded_weight_array = torch.reshape(unfolded_weight_array, (self.groups,
                                                                   self.original_module.out_channels,
                                                                   int(self.original_module.in_channels / self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = torch.reshape(unfolded_weight_array, (self.groups,
                                                                       self.original_module.out_channels,
                                                                       int(self.original_module.in_channels / self.groups),
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = torch.permute(approximated_weight_array, (1, 0, 2, 3, 4))
        approximated_weight_array = torch.reshape(approximated_weight_array, (self.original_module.out_channels,
                                                                           self.original_module.in_channels,
                                                                           self.original_module.kernel_size[0],
                                                                           self.original_module.kernel_size[1]))

        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        assert rank > 0
        self.rank = rank

        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.groups * self.rank, 
                                        kernel_size=self.original_module.kernel_size, 
                                        stride=self.original_module.stride, 
                                        padding=self.original_module.padding, 
                                        groups=self.groups, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.groups * self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0, 
                                        groups=1, 
                                        bias=(self.original_module.bias!=None), 
                                        dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0
        low_rank_weight_array1 = torch.reshape(vh_s_sqrt, (self.groups * rank, 
                                                        int(self.original_module.in_channels/self.groups), 
                                                        self.original_module.kernel_size[0], 
                                                        self.original_module.kernel_size[1]))

        low_rank_weight_array2 = torch.reshape(u_s_sqrt, (self.groups,
                                                       self.original_module.out_channels,
                                                       rank,
                                                       1, 
                                                       1))
        low_rank_weight_array2 = torch.permute(low_rank_weight_array2, (1, 0, 2, 3, 4))
        low_rank_weight_array2 = torch.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                                     self.groups * rank,
                                                                     1,
                                                                     1))

        return low_rank_weight_array1, low_rank_weight_array2

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0

        vh_s_sqrt = torch.reshape(low_rank_weight_array1, (self.groups,
                                                        rank, 
                                                        int(self.original_module.in_channels/self.groups)*self.original_module.kernel_size[0]*self.original_module.kernel_size[1]))
        u_s_sqrt = torch.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                       self.groups,
                                                       rank,
                                                       1,
                                                       1))
        u_s_sqrt = torch.permute(u_s_sqrt, (1, 0, 2, 3, 4))
        u_s_sqrt = torch.reshape(u_s_sqrt, (self.groups,
                                         self.original_module.out_channels,
                                         rank))       

        return u_s_sqrt, vh_s_sqrt

class StaticLowRankScheme5(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size, groups):
        super(StaticLowRankScheme5, self).__init__(original_module, original_ofm_size)
        self.groups = groups
        self.scheme = 5

    def get_max_rank(self):
        max_rank = min(self.original_module.in_channels, 
                        int(self.original_module.out_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])                          
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[2] \
                        * self.original_ofm_size[3] \
                        * self.groups \
                        * (self.original_module.in_channels * self.original_module.stride[0] * self.original_module.stride[1]
                            + int(self.original_module.out_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_macs
    
    def get_per_rank_params(self):
        per_rank_params = self.groups \
                            * (self.original_module.in_channels 
                                + int(self.original_module.out_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = torch.reshape(original_weight_array, (self.groups,
                                                                   int(self.original_module.out_channels / self.groups), 
                                                                   self.original_module.in_channels,
                                                                   self.original_module.kernel_size[0],
                                                                   self.original_module.kernel_size[1]))
        unfolded_weight_array = torch.permute(unfolded_weight_array, (0, 2, 1, 3, 4))
        unfolded_weight_array = torch.reshape(unfolded_weight_array, (self.groups,
                                                                   self.original_module.in_channels,
                                                                   int(self.original_module.out_channels / self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = torch.reshape(unfolded_weight_array, (self.groups,
                                                                       self.original_module.in_channels,
                                                                       int(self.original_module.out_channels / self.groups),
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = torch.permute(approximated_weight_array, (0, 2, 1, 3, 4))
        approximated_weight_array = torch.reshape(approximated_weight_array, (self.original_module.out_channels,
                                                                           self.original_module.in_channels,
                                                                           self.original_module.kernel_size[0],
                                                                           self.original_module.kernel_size[1]))
                    
        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        assert rank > 0
        self.rank = rank
        
        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.groups * self.rank, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0, 
                                        groups=1, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.groups * self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=self.original_module.kernel_size, 
                                        stride=self.original_module.stride, 
                                        padding=self.original_module.padding, 
                                        groups=self.groups, 
                                        bias=(self.original_module.bias!=None), dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0
        low_rank_weight_array1 = torch.reshape(u_s_sqrt, (self.groups, 
                                                       self.original_module.in_channels, 
                                                       rank, 
                                                       1, 
                                                       1))
        low_rank_weight_array1 = torch.permute(low_rank_weight_array1, (0, 2, 1, 3, 4))
        low_rank_weight_array1 = torch.reshape(low_rank_weight_array1, (self.groups*rank, 
                                                                     self.original_module.in_channels, 
                                                                     1, 
                                                                     1))

        low_rank_weight_array2 = torch.reshape(vh_s_sqrt, (self.groups, 
                                                        rank,
                                                        int(self.original_module.out_channels/self.groups),
                                                        self.original_module.kernel_size[0], 
                                                        self.original_module.kernel_size[1]))
        low_rank_weight_array2 = torch.permute(low_rank_weight_array2, (0, 2, 1, 3, 4))
        low_rank_weight_array2 = torch.reshape(low_rank_weight_array2, (self.original_module.out_channels, 
                                                                     rank,
                                                                     self.original_module.kernel_size[0], 
                                                                     self.original_module.kernel_size[1]))

        return low_rank_weight_array1, low_rank_weight_array2  

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0
        u_s_sqrt = torch.reshape(low_rank_weight_array1,(self.groups,
                                                     rank,
                                                     self.original_module.in_channels,
                                                     1,
                                                     1))
        u_s_sqrt = torch.permute(u_s_sqrt, (0, 2, 1, 3, 4))
        u_s_sqrt = torch.reshape(u_s_sqrt, (self.groups, 
                                         self.original_module.in_channels, 
                                         rank))

        vh_s_sqrt = torch.reshape(low_rank_weight_array2, (self.groups,
                                                        int(self.original_module.out_channels/self.groups), 
                                                        rank, 
                                                        self.original_module.kernel_size[0],
                                                        self.original_module.kernel_size[1]))  
        vh_s_sqrt = torch.permute(vh_s_sqrt, (0, 2, 1, 3, 4))  
        vh_s_sqrt = torch.reshape(vh_s_sqrt, (self.groups, 
                                           rank,
                                           int(self.original_module.out_channels/self.groups)*self.original_module.kernel_size[0]*self.original_module.kernel_size[1]))    
        return u_s_sqrt, vh_s_sqrt   

class StaticLowRankScheme0(StaticLowRankScheme5):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme0, self).__init__(original_module, original_ofm_size, original_module.out_channels)
        self.scheme = 0

class StaticLowRankScheme1(StaticLowRankScheme4):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme1, self).__init__(original_module, original_ofm_size, 1)
        self.scheme = 1

class StaticLowRankScheme3(StaticLowRankScheme4):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme3, self).__init__(original_module, original_ofm_size, original_module.in_channels)
        self.scheme = 3

class Linear_Conv2d(nn.Module):
    def __init__(self, linear_module):
        super(Linear_Conv2d, self).__init__()
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features

        self.conv_module = nn.Conv2d(linear_module.in_features, 
                                     linear_module.out_features, 
                                     kernel_size=1, 
                                     stride=1, 
                                     padding=0, 
                                     groups=1, 
                                     bias=(linear_module.bias!=None), 
                                     dilation=1)

        self.conv_module.weight.data = torch.reshape(linear_module.weight.data,
                                                    (linear_module.out_features, 
                                                    linear_module.in_features, 
                                                    1, 
                                                    1))
        if linear_module.bias != None:
            self.conv_module.bias.data = linear_module.bias.data

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, self.in_features, 1, 1)
        out = self.conv_module(x)
        new_shape = list(shape[:-1]) + [self.out_features]
        out = out.view(new_shape)
        return out