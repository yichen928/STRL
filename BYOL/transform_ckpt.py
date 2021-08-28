import os
import sys

import torch
import argparse
from collections import OrderedDict
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="input ckpt to transform")
parser.add_argument("--output_path", default=None, type=str, help="output path of transformed ckpt")
args = parser.parse_args()

input_path = args.input_path
if args.output_path is None:
    output_path = input_path[:-5] + "_transformed.ckpt"
else:
    output_path = args.output_path

ckpt = torch.load(input_path)
transformed_ckpt = {}

state_dict = ckpt["state_dict"]
new_state_dict = OrderedDict()

for key in state_dict:
    if ("online_network." in key or "online_netwrok." in key) and ".projector" not in key and ".predictor" not in key:
        print(key)
        new_state_dict[key[15:]] = state_dict[key]

transformed_ckpt["state_dict"] = new_state_dict
print(len(new_state_dict))
print(output_path)
torch.save(transformed_ckpt, output_path)

