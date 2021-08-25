import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import time
from collections import OrderedDict
import json
import vgtk
import ZPConvNets.utils as M
import vgtk.zpconv.functional as L

class ClsSO3ConvModel(nn.Module):
    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
        self.outblock = M.SO3ClsOutBlockR(params['outblock'])

        # (TODO) remove hard coded anchor dim
        self.na_in = 60
        self.invariance = True

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = M.preprocess_input(x, self.na_in, False)
        # x = M.preprocess_input(x, 1, False)

        for block_i, block in enumerate(self.backbone):
            x = block(x)

        x = self.outblock(x)
        return x

    def get_anchor(self):
        return self.backbone[-1].get_anchor()


# Full Version
# def build_model(device,
#                 mlps=[[16,32], [32,64], [64,128]],
#                 out_mlps=[128],
#                 intra_mlp = [128,128,128],
#                 strides=[2, 2, 8],
#                 initial_radius_ratio = 0.2,
#                 sampling_ratio = 0.5,
#                 sampleing_density = 0.5,
#                 kernel_density = 1,
#                 kernel_multiplier = 2,
#                 input_radius=1.0,
#                 sigma_ratio= 1e-3,
#                 input_num=2048,
#                 dropout_rate=0.,
#                 xyz_pooling = "no-stride", 
#                 so3_pooling = "max",
#                 to_file=None):

# Light Version
def build_model(device,
                mlps=[[16,16], [32,32], [64,64]],
                out_mlps=[64],
                intra_mlp = [64, 64, 64, 64, 64, 64],
                strides=[2, 2, 8],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.5,
                sampleing_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                input_radius=1.0,
                sigma_ratio= 1e-3,
                input_num=2048,
                dropout_rate=0.,
                xyz_pooling = "no-stride", 
                so3_pooling = "max",
                to_file=None):

    params = {'name': 'Invariant ZPConv Model',
              'backbone': [],
              }
    dim_in = 1

    # process args
    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= strides[i]
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

    radius_ratio = [initial_radius_ratio * multiplier**sampleing_density for multiplier in stride_multipliers]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio]

    # n_neighbor = [int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampleing_density)) for i in range(n_layer + 1)]
    # n_neighbor = [32, 64]

    # kernel_size = [1 + int(kernel_density * math.sqrt(n_neighbor[i] / kernel_multiplier)) for i in range(n_layer)]
    # kernel_size = [2,2,2,2]

    # Compute sigma
    weighted_sigma = [sigma_ratio * radii[i]**2 * stride_multipliers[i] for i in range(n_layer + 1)]

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            stride_conv = i == 0 or xyz_pooling != 'stride'

            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampleing_density))          
            kernel_size = 1
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i]
                nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                if stride_conv:
                    neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i+1]**(1/sampleing_density))
                    kernel_size = 2 # if inter_stride < 4 else 3
            else:
                inter_stride = 1
                nidx = i+1

            # print(f"At block {i}, layer {j}!")
            # print(f'neighbor: {neighbor}')
            # print(f'stride: {inter_stride}')
            # print(f'sigma: {weighted_sigma[nidx]}')
            # print(f'radius {radii[nidx]}')
            # import ipdb; ipdb.set_trace()

            # one-inter one-intra policy
            inter_param = {
                'type': 'inter_block',
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': inter_stride,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'n_neighbor': neighbor,
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                }
            }
            block_param.append(inter_param)

            dim_in = dim_out

        params['backbone'].append(block_param)

    out_dim_in = dim_in
    dim_in = out_mlps[-1]
    intra_block_param = []

    for dim_out in intra_mlp:
        intra_param = {
            'type': 'intra_block',
            'args': {
                'dim_in': dim_in,
                'dim_out': dim_out,
                'dropout_rate': dropout_rate,
                'activation': 'leaky_relu',
            }
        }
        intra_block_param.append(intra_param)
        dim_in = dim_out

    params['outblock'] = {
        'dim_in': out_dim_in,
        'mlp': out_mlps,
        'intra': intra_block_param,
        'tempature': 10,
        'fc': [128, 64],
        'k': 40,
        'pooling': so3_pooling,
    }

    # print(params)
    # import ipdb; ipdb.set_trace()

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model

# TODO
def build_model_from(opt, outfile_path=None):
    if opt.model.flag == 'attention':
        return build_model(opt.device, input_num=opt.model.input_num, dropout_rate=opt.model.dropout_rate, to_file=outfile_path, so3_pooling = "attention")
    else:
        return build_model(opt.device, input_num=opt.model.input_num, dropout_rate=opt.model.dropout_rate, to_file=outfile_path)
