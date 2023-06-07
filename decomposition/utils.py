import copy
import numpy as np
import scipy.stats as stats
import torch.nn as nn

from decomposition.common import *
from decomposition.cp import *
from decomposition.svd import *


def svd_target_module_filter(model_name, module_name, module):
    if model_name in ["resnet18", "repvgga0"]:
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (3, 3) and module.in_channels != 3:
                assert module.groups == 1
                return True
        return False

def get_rank_by_compression_ratio(low_rank_wrapper, compression_ratio):
    assert compression_ratio > 0 and compression_ratio <= 1, "Compression ratio should be in (0,1]"
    return max(1, int(low_rank_wrapper.get_original_module_macs()*compression_ratio/low_rank_wrapper.get_per_rank_macs()))

def enumerate_decomposition_candidates(original_module, original_ofm_size, method):
    low_rank_candidates = []
    if "SVD" in method:
        low_rank_candidates += enumerate_scheme_candidates(original_module, original_ofm_size)
    if "CP" in method and original_module.kernel_size[0] > 1:
        low_rank_candidates += enumerate_cp_candidates(original_module, original_ofm_size)    
    return low_rank_candidates

def build_decomposition_candidates(model_name, original_model, module_name, target_module, method=["SVD"]):                       
    sweep_list = [0.3, 0.7, 0.05]
    baseline_macs =  get_conv2d_macs(target_module, target_module.ofm_size)
    candidate_wrappers = []
    candidates = []
    
    for scheme_wrapper in enumerate_decomposition_candidates(target_module, target_module.ofm_size, method):
        rank_list = []  
        for ratio in reversed(np.arange(sweep_list[0], sweep_list[1]+sweep_list[2], sweep_list[2])):
            rank = get_rank_by_compression_ratio(scheme_wrapper, ratio)
            if rank not in rank_list:
                scheme_wrapper.rank = rank   
                rank_list.append(rank)

        filtered_rank_list = []
        if len(rank_list) > 1:
            rank_list = [int(2*rank_list[0])-rank_list[1]] + rank_list
            if int(2*rank_list[-1])-rank_list[-2] > 0: 
                rank_list = rank_list + [int(2*rank_list[-1])-rank_list[-2]]

            for i in range(len(rank_list)-1):
                max_factors_num = 0
                max_factors_n = None
                for n in range(rank_list[i+1], rank_list[i]):
                    factors_num = len(get_factors(n))
                    if factors_num > max_factors_num:
                        max_factors_num = factors_num
                        max_factors_n = n
                filtered_rank_list.append(max_factors_n)
            rank_list = filtered_rank_list

        for rank in rank_list:
            scheme_wrapper.rank = rank
            low_rank_macs = get_low_rank_macs(scheme_wrapper) 
            if low_rank_macs > baseline_macs:
                continue
            candidate_wrappers.append((scheme_wrapper, rank))
            wrapper_copy = copy.deepcopy(scheme_wrapper)
            wrapper_copy.initialise_low_rank_module(rank)
            wrapper_copy.generate_low_rank_weight()
            new_candidate = wrapper_copy.export_decomposition()
            new_candidate.macs = low_rank_macs
            candidates.append(new_candidate)
        print(scheme_wrapper, rank_list)

    new_candidate = copy.deepcopy(target_module)
    new_candidate.macs = baseline_macs
    candidate_wrappers.append((-1, -1))
    candidates.append(new_candidate)

    return candidate_wrappers, candidates 

def wrapper_to_config(low_rank_wrapper):
    if isinstance(low_rank_wrapper, StaticLowRankDecompositionWrapper):
        return {"method": "SVD", "scheme": int(low_rank_wrapper.scheme), "group": int(low_rank_wrapper.groups), "rank": int(low_rank_wrapper.rank)}
    elif isinstance(low_rank_wrapper, CPDecompositionWrapper):
        assert low_rank_wrapper.group_c == 1
        assert low_rank_wrapper.group_f == 1
        return {"method": "CP", "rank": int(low_rank_wrapper.rank)}
    else:
        assert False, "Unknown low rank wrapper"

def config_to_wrapper(original_module, original_ofm_size, config):
    if config["method"] == "SVD":
        low_rank_wrapper = generate_low_rank_wrapper(config["scheme"], original_module, original_ofm_size, config["group"])
        low_rank_wrapper.initialise_low_rank_module(config["rank"])
    elif config["method"] == "CP":
        low_rank_wrapper = CPDecompositionWrapper(original_module, original_ofm_size, 1, 1)
        low_rank_wrapper.initialise_low_rank_module(config["rank"])
    else:
        assert False, "Unknown low rank wrapper"

    return low_rank_wrapper

def config_encoder(config):
    if config == -1:
        code = [-1, -1, -1, -1]
    elif config["method"] == "SVD":
        # 83 86 68 ASCII code for "SVD"
        code = [838668, config["scheme"], config["group"], config["rank"]]
    elif config["method"] == "CP":
        # 67 80 ASCII code for "CP"
        code = [6780, config["rank"], -1, -1]
    else:
        assert False, "Unknown config"

    assert len(code) == 4, "Code length should be 4"
    return code

def config_decoder(code):
    assert len(code) == 4, "Code length should be 4"

    if code[0] == -1:
        config = -1
    elif code[0] == 838668:
        config = {"method": "SVD", "scheme": code[1], "group": code[2], "rank": code[3]}
    elif code[0] == 6780:
        config = {"method": "CP", "rank": code[1]}
    else:
        assert False, "Unknown config"

    return config

def generate_random_sample(candidates, mean=0.8, std_dev=0.15, step=0.05):
    range_list = [(x, x+step) for x in np.arange(0, 1, step)]
    range_dict = {}
    baseline_macs = candidates[-1].macs
    for i, candidate in enumerate(candidates):
        macs_ratio = candidate.macs / baseline_macs

        bFound = False
        for range_bin in range_list:
            if range_bin[0] < macs_ratio and macs_ratio <= range_bin[1]:
                if range_bin not in range_dict.keys():
                    range_dict[range_bin] = []
                range_dict[range_bin].append(i)
                bFound = True
                break
        assert bFound

    range_dict = dict(sorted(range_dict.items()))
    range_prob = []

    for range_bin in range_dict.keys():
        range_prob.append(stats.norm.cdf(range_bin[1], loc=mean, scale=std_dev)-stats.norm.cdf(range_bin[0], loc=mean, scale=std_dev))                                                                     
    
    range_prob = range_prob / np.sum(range_prob)
    return range_dict, range_prob