import numpy as np
import os
import random
import torch

from decomposition.utils import *
from fpga_utils import launch_optimiser
from regressor import random_forest_regressor
from utils import *

class EvolutionaryAgent:
    def __init__(self, id, base_model, model_name, perf_mode, target_compression_ratio, cand_mappings, search_loader, val_loader, macs_latency_bias):
        self.agent_id = id
        random_input = torch.randn(base_model.input_size)
        if torch.cuda.is_available():
            random_input = random_input.cuda()
        self.random_input = random_input
        
        self.model = base_model
        self.model_name = model_name

        self.val_acc_cache = {}
        self.perf_cache = {}

        self.acc_query_count = 0
        self.perf_query_count = 0

        assert perf_mode in ["fpgaconvnet_run", "fpgaconvnet_sample_predict", "fpgaconvnet_macs"]

        if perf_mode == "fpgaconvnet_sample_predict":
            # launch optimiser in the first few generations
            perf_mode = "fpgaconvnet_run"

        self.perf_mode = perf_mode
        self.baseline_perf = self.get_model_performance()

        self.target_perf = self.baseline_perf * target_compression_ratio
        self.target_macs = self.get_model_performance(mode='fpgaconvnet_macs') * (target_compression_ratio-macs_latency_bias)

        self.min_accuracy = 5.0
        self.cand_mappings = cand_mappings

        self.search_loader = search_loader
        self.search_data_iter = iter(search_loader)
        self.val_loader = val_loader

        self.predictor = None
        self.pred_X = []
        self.pred_Y = []

    def report_status(self):
        status = {
            "val_acc_cache": self.val_acc_cache,
            "perf_cache": self.perf_cache,
            "acc_query_count": self.acc_query_count,
            "perf_query_count": self.perf_query_count
        }
        return status

    def apply_sample(self, sample):
        return replace_modules(self.model, {k: v[1] for k, v in sample.items()})

    def revert_sample(self, sample):
        return replace_modules(self.model, {v[1]: k for k, v in sample.items()})

    def get_model_accuracy_proxy(self, fix_batch=False):
        self.model.eval()
        with torch.no_grad():
            if fix_batch and hasattr(self, "search_images"):
                search_images, search_targets = self.search_images, self.search_targets
            else:
                try:
                    search_images, search_targets = next(self.search_data_iter)
                except StopIteration:
                    self.search_data_iter = iter(self.search_loader)
                    search_images, search_targets = next(self.search_data_iter)
                if torch.cuda.is_available():
                    search_images = search_images.cuda(non_blocking=True)
                    search_targets = search_targets.cuda(non_blocking=True)
                if fix_batch:
                    self.search_images, self.search_targets = search_images, search_targets
            outputs = self.model(search_images)
            acc1, acc5 = accuracy(outputs, search_targets, topk=(1, 5))
            self.acc_query_count += 1
            return acc1.item()

    def get_model_valid_accuracy(self, sample):
        hash_x = "_".join([str(x[0]) for x in sample.values()])
        if hash_x in self.val_acc_cache.keys():
            return self.val_acc_cache[hash_x]
        else:
            self.apply_sample(sample)
            val_top1, val_top5 = validate(self.val_loader, self.model, nn.CrossEntropyLoss(), verbose=False)
            val_top1 = val_top1.avg.item()
            val_top5 = val_top5.avg.item()
            self.revert_sample(sample)
            self.val_acc_cache[hash_x] = (val_top1, val_top5)
            return val_top1, val_top5

    def prepare_predictor_features(self, cfg):
        features = []
        for c in cfg:
            features.extend(c)
        macs, params = calculate_macs_params(self.model, self.random_input, False, False)
        features.extend([macs, params])

        return features

    def get_model_performance(self, mode=None, cfg=None):
        if mode == None:
            mode = self.perf_mode
        if mode not in self.perf_cache.keys():
            print("Create perf cache for mode: ", mode)
            self.perf_cache[mode] = {}
        if cfg != None:
            hash_cfg = "_".join([str(x) for x in cfg])
            if hash_cfg in self.perf_cache[mode].keys():
                return self.perf_cache[mode][hash_cfg]
        if mode == "fpgaconvnet_macs":
            macs, params = calculate_macs_params(self.model, self.random_input, False, False)
            perf = macs
        elif mode == "fpgaconvnet_run":
            model_name = self.model_name if cfg is None else self.model_name+""+"_mtd"
            model_name += "_"+str(self.agent_id) # prevent conflict between runs
            output_path = os.path.join("/tmp/fpgaconvnet", model_name)
            report = launch_optimiser(self.model, model_name, self.random_input, output_path, silence=(cfg != None))
            throughput = report["network"]["performance"]["throughput"]
            if cfg != None:
                features = self.prepare_predictor_features(cfg)
                self.pred_X.append(features)
                self.pred_Y.append(throughput)
            perf = 1/throughput
        elif mode == "fpgaconvnet_predict":
            if self.predictor == None:
                self.build_predictor()
            features = self.prepare_predictor_features(cfg)
            throughput = self.predictor.predict([features])[0]
            perf = 1/throughput
        if cfg != None:
            self.perf_cache[mode][hash_cfg] = perf
        if mode == self.perf_mode:
            self.perf_query_count += 1
        return perf

    def build_predictor(self):
        self.predictor = random_forest_regressor(self.pred_X, self.pred_Y)

    def encode_cfg(self, wrapper_info):
        if wrapper_info[0] != -1:
            wrapper_info[0].rank = wrapper_info[1]
            cfg = wrapper_to_config(wrapper_info[0])
            cfg = config_encoder(cfg) 
        else:
            cfg = config_encoder(-1)
        return cfg

    def filter_candidates(self, tolerance, reverse):
        baseline_accuracy = self.get_model_accuracy_proxy(fix_batch=True)
        for module, (candidate_wrappers, candidates) in self.cand_mappings.items():
            good_candidates = []
            current_wrapper = None
            i_range = range(len(candidates)-1)
            if reverse:
                i_range = reversed(i_range)
            for i in i_range:
                if current_wrapper == candidate_wrappers[i][0]:
                    good_candidates.append(i)
                    continue
                self.apply_sample({module: (self.encode_cfg(candidate_wrappers[i]), candidates[i])})
                accuracy = self.get_model_accuracy_proxy(fix_batch=True)
                if accuracy >= baseline_accuracy - tolerance:
                    good_candidates.append(i)
                    current_wrapper = candidate_wrappers[i][0]
                self.revert_sample({module: (self.encode_cfg(candidate_wrappers[i]), candidates[i])})
            good_candidates.append(len(candidates)-1)
            self.cand_mappings[module] = ([candidate_wrappers[i] for i in good_candidates], [candidates[i] for i in good_candidates])
        del self.search_images, self.search_targets

    def measure_sample(self, sample):
        # run different proxy tasks: macs, accuracy, throughput
        self.apply_sample(sample)
        macs = self.get_model_performance(mode="fpgaconvnet_macs", cfg=[x[0] for x in sample.values()])
        if macs <= self.target_macs:
            accuracy = self.get_model_accuracy_proxy()
            if accuracy >= self.min_accuracy:
                if self.perf_mode == "fpgaconvnet_macs":
                    efficiency = macs
                else:
                    efficiency = self.get_model_performance(cfg=[x[0] for x in sample.values()])
                self.revert_sample(sample)
                return [efficiency <= self.target_perf, accuracy, efficiency]
            else:
                self.revert_sample(sample)
                return [False, accuracy, "NA"]
        else:
            self.revert_sample(sample)
            return [False, "NA", "NA"]

    def random_sample(self, weights=None):
        while True:
            sample = {}
            for module, (candidate_wrappers, candidates) in self.cand_mappings.items():
                if weights is None:
                    index = random.choice(range(len(candidates)))
                else:
                    index = np.random.choice(range(len(candidates)), p=weights[module])
                cfg = self.encode_cfg(candidate_wrappers[index])
                sample[module] = (cfg, candidates[index])

            results = self.measure_sample(sample)
            if results[0]:
                return sample, results[1], results[2]
    
    def mutate_sample(self, sample, mutate_prob, weights=None):
        while True:
            new_sample = {}
            indices = []
            for module, (candidate_wrappers, candidates) in self.cand_mappings.items():
                if random.random() < mutate_prob:
                    if weights is None:
                        index = random.choice(range(len(candidates)))
                    else:
                        index = np.random.choice(range(len(candidates)), p=weights[module])
                    cfg = self.encode_cfg(candidate_wrappers[index])
                    new_sample[module] = (cfg, candidates[index])
                    indices.append(index)
                else:
                    new_sample[module] = sample[module]

            results = self.measure_sample(new_sample)
            if results[0]:
                return new_sample, results[1], results[2]    

    def crossover_sample(self, sample1, sample2):
        while True:
            new_sample = {}
            indices = []
            for module, (candidate_wrappers, candidates) in self.cand_mappings.items():
                index = random.choice([0,1])
                new_sample[module] = [sample1[module], sample2[module]][index]
                indices.append(index)
            
            results = self.measure_sample(new_sample)
            if results[0]:
                return new_sample, results[1], results[2] 