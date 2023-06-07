import argparse
import copy
import csv
import hashlib
import json
import numpy as np
import os
import pickle
import random
import time
import torch

from decomposition.utils import *
from evo_agent import EvolutionaryAgent
from utils import *


def search_main():
    parser = argparse.ArgumentParser(description='Mixed-TD Search')

    parser.add_argument('--gpu', default=None, type=int)

    parser.add_argument('--batch_size', default='128', type=int)
    parser.add_argument('--workers', default='4', type=int)

    parser.add_argument('--output_path', default=None, type=str)

    parser.add_argument('--model_name', default='resnet18', type=str,
                        choices=["resnet18", "repvgga0"])
    parser.add_argument('--method', default=['SVD', 'CP'], type=str, nargs='+') 

    parser.add_argument('--perf_mode', default="fpgaconvnet_sample_predict", type=str,
                        choices=["fpgaconvnet_macs", "fpgaconvnet_run",  "fpgaconvnet_sample_predict"])

    args = parser.parse_args()

    args.search_size = 50000
    args.max_steps = 500
    args.start_step = 0

    if args.model_name == "resnet18":
        args.required_compression_ratio = 0.65
        if args.method == ["CP"]:
            args.sample_bias = 0.2
            args.macs_latency_bias = 0.3
        else:
            args.sample_bias = 0.1
            args.macs_latency_bias = 0.23
    elif args.model_name == "repvgga0":
        args.required_compression_ratio = 0.75
        if args.method == ["CP"]:
            args.sample_bias = 0.20
            args.macs_latency_bias = 0.35
        else:
            args.sample_bias = 0.15
            args.macs_latency_bias = 0.23

    if args.output_path == None:
        args.output_path = os.getcwd() + "/output"
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.resume_checkpoint = None
    if args.resume_checkpoint is not None:
        with open(args.resume_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
        args.start_step = checkpoint["step"] + 1

    print(args)

    data_begin = time.time()

    agent_id = hashlib.md5(args.output_path.encode()).hexdigest()
    print("Agent id: " + str(agent_id))

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model = load_model(args.model_name)
    random_input = torch.randn(model.input_size)        
    train_loader, search_loader, val_loader = prepare_dataloader(model, args.batch_size, args.workers, args.search_size)
    if torch.cuda.is_available():
        print("Using gpu " + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        model.cuda()
        random_input = random_input.cuda()
    annoate_feature_map_size(model, random_input)

    model_begin = time.time()
    print("Data loading time: ", model_begin - data_begin)

    agent_model = copy.deepcopy(model)
    cand_mappings = {}
    if torch.cuda.is_available():
        agent_model.cuda()

    wrapper_mapping = {}
    lr_module_cache = {}
    conv_layer_index = 0
    for name, module in agent_model.named_modules(): 
        if svd_target_module_filter(args.model_name, name, module):
            candiate_wrappers, candidates = build_decomposition_candidates(args.model_name, agent_model, name, module, method=args.method)                   
            for wrap, cand in zip(candiate_wrappers, candidates):
                wrapper_mapping[cand] = wrap
                cfg = EvolutionaryAgent.encode_cfg(None,wrap)
                cfg = "_".join([str(x) for x in cfg])
                lr_module_cache[(module, cfg)] = cand
            cand_mappings[module] = (candiate_wrappers, candidates)

            conv_layer_index += 1

    search_begin = time.time()
    print("Model loading time: ", search_begin - model_begin)

    agent_model.eval()
    model.eval()

    agent = EvolutionaryAgent(agent_id, agent_model, args.model_name, args.perf_mode, args.required_compression_ratio, cand_mappings, search_loader, val_loader, args.macs_latency_bias)
    agent.lr_module_cache = lr_module_cache
    agent.wrapper_mapping = wrapper_mapping
    if args.model_name == "resnet18":
        agent.filter_candidates(tolerance=1.0, reverse=False)
    elif args.model_name == "repvgga0":
        agent.filter_candidates(tolerance=5.0, reverse=True)

    init_prob = {}
    for module, (candiate_wrappers, candidates) in agent.cand_mappings.items():
        dist_mean = args.required_compression_ratio-args.sample_bias-args.macs_latency_bias
        range_dict, range_prob = generate_random_sample(candidates, mean=dist_mean)
        init_prob[module] = [0] * len(candidates)
        for i, (k,v) in enumerate(range_dict.items()):
            for j in v:
                init_prob[module][j] = range_prob[i] / len(v)

    children_size = 100
    parents_size = 25
    mutation_size = 50
    mutate_prob = 0.2
    crossover_size = 50
    valid_cache = {}
    assert children_size == mutation_size + crossover_size # new children per generation

    best_top1 = -1
    best_info = None
    total_step_time = 0
    for i in range(args.start_step, args.max_steps):
        step_begin = time.time()

        if args.perf_mode == "fpgaconvnet_sample_predict":
            # schedule fpgaconvnet proxy mode
            if i <= 2:
                agent.perf_mode = "fpgaconvnet_run"            
            else:
                agent.perf_mode = "fpgaconvnet_predict"
                agent.build_predictor()

        if args.resume_checkpoint is not None:
            dummy_status = agent.report_status()
            for k in dummy_status.keys():
                setattr(agent, k, checkpoint["agent_status"][k])
            parents = []
            for j in range(parents_size):
                sample = {}
                module_count = 0
                for module, (candidate_wrappers, candidates) in agent.cand_mappings.items():
                    for index in range(len(candidate_wrappers)):
                        cfg = agent.encode_cfg(candidate_wrappers[index])
                        if cfg == checkpoint["parent_status"][j][module_count]:
                            sample[module] = (cfg, candidates[index])
                            break
                    module_count += 1
                assert len(sample) == len(agent.cand_mappings)
                parents.append([sample, checkpoint["parent_status"][j][-2], checkpoint["parent_status"][j][-1]])

        if i == 0:
            population = []
            for j in range(children_size):
                print("First Generation: ", j, "/", children_size)
                sample, accuracy, efficiency = agent.random_sample(weights=init_prob)
                population.append([sample, accuracy, efficiency])
        else:
            for j in range(parents_size):
                agent.apply_sample(parents[j][0])
                parents[j][1] = agent.get_model_accuracy_proxy()
                agent.revert_sample(parents[j][0])

            children = []
            for j in range(mutation_size):
                parent = random.choice(parents)
                sample, accuracy, efficiency = agent.mutate_sample(parent[0], mutate_prob)
                children.append([sample, accuracy, efficiency])

            for j in range(crossover_size):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                sample, accuracy, efficiency = agent.crossover_sample(parent1[0], parent2[0])
                children.append([sample, accuracy, efficiency])
            population = parents + children

        population = sorted(population, key=lambda x: x[1])[::-1]

        if agent.perf_mode == "fpgaconvnet_predict":
            reject_list = []
            for j in range(len(population)):
                agent.apply_sample(sample=population[j][0])
                actual_efficiency = agent.get_model_performance(mode="fpgaconvnet_run", cfg=[x[0] for x in population[j][0].values()])
                agent.revert_sample(sample=population[j][0])
                if actual_efficiency <= agent.target_perf:
                    break
                else:
                    reject_list.append(j)
            print ("False prediction reject list: ", reject_list)
            population = [population[j] for j in range(len(population)) if j not in reject_list]
            if len(population) > parents_size:
                parents = population[:parents_size]
            else:
                assert len(population) > 0
                parents = population
        else:
            parents = population[:parents_size]

        # evaluate
        val_top1, val_top5 = agent.get_model_valid_accuracy(parents[0][0])
        decompositon_config = [config_decoder(v[0]) for v in parents[0][0].values()]
        with open(args.output_path + "/search_decomposition_config.json", 'w') as f:
            json.dump(decompositon_config, f, indent=4)

        info = "Step: {} Search Top1: {} Val Top1: {}".format(i, parents[0][1], val_top1)
        if agent.perf_mode not in ["fpgaconvnet_predict", "fpgaconvnet_macs"]:
            info += " Efficiency: {}".format(parents[0][2])
        else:
            agent.apply_sample(sample=parents[0][0])
            actual_efficiency = agent.get_model_performance(mode="fpgaconvnet_run", cfg=[x[0] for x in parents[0][0].values()])
            info += " Efficiency: {}".format(actual_efficiency)
            agent.revert_sample(sample=parents[0][0])

        agent.apply_sample(sample=parents[0][0])
        info += " MACs: {}".format(agent.get_model_performance(mode="fpgaconvnet_macs", cfg=[x[0] for x in parents[0][0].values()]))
        agent.revert_sample(sample=parents[0][0])
        print(info)

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_info = info
            with open(args.output_path + "/best_decomposition_config.json", 'w') as f:
                json.dump(decompositon_config, f, indent=4)
        print("Best ", best_info)

        step_end = time.time()
        total_step_time += step_end - step_begin
        print("Step: {} Time: {} / {}".format(i, step_end - step_begin, total_step_time))
        print("Acc Queries: ", agent.acc_query_count, "Perf Queries: ", agent.perf_query_count)
        
        # save checkpoint
        agent_status = agent.report_status()
        parent_status = []
        for p in parents:
            parent_status.append([x[0] for x in p[0].values()] + [p[1], p[2]])
        with open(args.output_path + "/checkpoint.pickle", 'wb') as f:
            pickle.dump({"step": i,  "agent_status": agent_status, "parent_status": parent_status}, f)
        
        if agent.perf_mode == "fpgaconvnet_run":
            np.save(args.output_path + "/X.npy", np.array(agent.pred_X))
            np.save(args.output_path + "/Y.npy", np.array(agent.pred_Y))

        csv_entry = [i]
        csv_entry += decompositon_config
        csv_entry += [val_top1, val_top5, parents[0][2]]
        with open(args.output_path + "/nas_log.csv", mode='a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_entry)
            
    search_end = time.time()
    print("Total time: ", search_end - data_begin)

if __name__ == "__main__":
    search_main()
