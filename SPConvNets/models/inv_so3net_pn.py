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
import SPConvNets.utils as M
import vgtk.spconv.functional as L

class InvSO3ConvModel(nn.Module):
    def __init__(self, params):
        super(InvSO3ConvModel, self).__init__()

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))

        self.outblock = M.InvOutBlockMVD(params['outblock'])
        # self.outblock = M.InvOutBlockPointnet(params['outblock'])
        self.na_in = params['na']
        self.invariance = True

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = M.preprocess_input(x, self.na_in, False)
        # x = M.preprocess_input(x, 1)

        for block_i, block in enumerate(self.backbone):
            x = block(x)

        x = self.outblock(x)
        return x

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

# Full Version
def build_model(opt,
                mlps=[[32,32], [64,64], [128,128], [128,128]],
                out_mlps=[128, 64],
                strides=[2, 2, 2, 2],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.8, #0.4, 0.36
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                sigma_ratio= 0.5, # 1e-3, 0.68
                xyz_pooling = None, # None, 'no-stride'
                to_file=None):

    device = opt.device
    input_num= opt.model.input_num
    dropout_rate= opt.model.dropout_rate
    temperature= opt.train_loss.temperature
    so3_pooling =  opt.model.flag
    input_radius = opt.model.search_radius
    kpconv = opt.model.kpconv

    na = 1 if opt.model.kpconv else opt.model.kanchor

    # to accomodate different input_num
    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    print("[MODEL] USING RADIUS AT %f"%input_radius)
    params = {'name': 'Invariant ZPConv Model',
              'backbone': [],
              'na': na
              }
    dim_in = 1

    # process args
    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio]

    # Compute sigma
    # weighted_sigma = [sigma_ratio * radii[i]**2 * stride_multipliers[i] for i in range(n_layer + 1)]

    weighted_sigma = [sigma_ratio * radii[0]**2]
    for idx, s in enumerate(strides):
        weighted_sigma.append(weighted_sigma[idx] * s)

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            stride_conv = i == 0 or xyz_pooling != 'stride'

            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))

            if i == 0 and j == 0:
                neighbor *= int(input_num / 1024)

            kernel_size = 1
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i]
                nidx = i if i == 0 else i+1
                # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                if stride_conv:
                    neighbor *= 2 # * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                    # neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i+1]**(1/sampling_density))
                    kernel_size = 1 # if inter_stride < 4 else 3
            else:
                inter_stride = 1
                nidx = i+1

            print(f"At block {i}, layer {j}!")
            print(f'neighbor: {neighbor}')
            print(f'stride: {inter_stride}')

            sigma_to_print = weighted_sigma[nidx]**2 / 3
            print(f'sigma: {sigma_to_print}')
            print(f'radius ratio: {radius_ratio[nidx]}')
            # import ipdb; ipdb.set_trace()

            # one-inter one-intra policy
            block_type = 'inter_block' if na != 60  else 'separable_block'

            conv_param = {
                'type': block_type,
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
                    'kanchor': na,
                }
            }
            block_param.append(conv_param)

            dim_in = dim_out

        params['backbone'].append(block_param)

    params['outblock'] = {
        'dim_in': dim_in,
        'mlp': out_mlps,
        'pooling': so3_pooling,
        'temperature': temperature,
        'kanchor': na,
    }


    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = InvSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
