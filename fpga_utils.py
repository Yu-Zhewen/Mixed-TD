import json
import numpy as np
import onnx
import os
import sys
import torch

from fpgaconvnet.optimiser.cli import main

def set_nodeattr(node, attr_name, attr_value):
    print("annotate ", node.name, attr_name, attr_value)
    new_attr = onnx.helper.make_attribute(attr_name, attr_value)
    node.attribute.append(new_attr)

def annotate_quantisation(model, weight_width, data_width, acc_width, block_floating_point):
    for node in model.graph.node:
        if node.op_type in ["Conv", "Gemm"]:
            set_nodeattr(node, "weight_width", weight_width)
            set_nodeattr(node, "data_width", data_width)
            set_nodeattr(node, "acc_width", acc_width)
            set_nodeattr(node, "block_floating_point", block_floating_point)
        else:
            set_nodeattr(node, "data_width", data_width)

def launch_optimiser(model, model_name, random_input, output_path=None, silence=False):
    if output_path is None:
        output_path = "/tmp/fpgaconvnet/" + model_name

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if silence:
        out = sys.stdout
        #err = sys.stderr
        null = open(os.devnull, 'w')
        sys.stdout = null
        #sys.stderr = null

    onnx_path = os.path.join("/tmp/fpgaconvnet/", "{model_name}.onnx".format(model_name=model_name))
    torch.onnx.export(model, random_input, onnx_path, verbose=False, keep_initializers_as_inputs=True)
    onnx_model = onnx.load(onnx_path)
    annotate_quantisation(onnx_model, 8, 8, 16, True)
    onnx.save(onnx_model, onnx_path)
    
    platform_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], 'examples/platforms/u250.toml')
    optimiser_config_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], 'examples/single_partition_throughput.toml')

    saved_argv = sys.argv
    sys.argv  = ['cli.py']
    sys.argv += ['--name', model_name]
    sys.argv += ['--model_path', onnx_path]
    sys.argv += ['--platform_path', platform_path]
    sys.argv += ['--output_path', output_path]
    sys.argv += ['-b', '1']
    sys.argv += ['--objective', 'throughput']
    sys.argv += ['--optimiser', 'greedy_partition']
    sys.argv += ['--optimiser_config_path', optimiser_config_path]

    main()
    sys.argv = saved_argv

    if silence:
        sys.stdout = out
        #sys.stderr = err

    with open(os.path.join(output_path, 'report.json'), 'r') as f:
        report = json.load(f)
        return report