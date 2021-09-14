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
    output_path = input_path[:-5] + "_transformed.tar"
else:
    output_path = args.output_path

ckpt = torch.load(input_path, map_location="cpu")
state_dict = ckpt["state_dict"]

in_key = []
not_in_key = []
target_state_dict = {}
for key in state_dict:
    if "online_network" in key:
        if (".sa" in key or ".fp" in key) and "sa_aggre" not in key:
            new_key = key.replace("online_network", "backbone_net")
            print(new_key)
            target_state_dict[new_key] = state_dict[key]

output_ckpt = {"model_state_dict":target_state_dict}
print(len(target_state_dict))

torch.save(output_ckpt, output_path)

