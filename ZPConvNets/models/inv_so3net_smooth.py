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

class InvSO3ConvSmoothModel(nn.Module):
    def __init__(self, params):
        super(InvSO3ConvSmoothModel, self).__init__()

        self.prop = M.PropagationBlock(params['propagation'], dropout_rate=params['dropout_rate'])

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
        # self.outblock = M.InvOutBlockR(params['outblock'])
        self.outblock = M.InvOutBlockPointnet(params['outblock'])
        self.na_in = params['na']
        self.invariance = True

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        # x = M.preprocess_input(x, self.na_in, False)
        # x = M.preprocess_input(x, 1)
        clouds, frag = x
        clouds = clouds.permute(0,2,1).contiguous()

        x = self.prop(frag, clouds)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        x = self.outblock(x)
        return x

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

# Full Version
# def build_model(opt,
#                 mlps=[[32,32], [64,64], [128,128]],
#                 out_mlps=[128, 32],
#                 strides=[2, 2, 2],
#                 initial_radius_ratio = 0.2,
#                 sampling_ratio = 0.8, #0.4, 0.36
#                 sampling_density = 0.5,
#                 kernel_density = 1,
#                 kernel_multiplier = 2,
#                 sigma_ratio= 1e-3, # 0.1
#                 xyz_pooling = None,
#                 to_file=None):
def build_model(opt,
                mlps=[[32,32], [64,64], [128,128], [128,128]],
                out_mlps=[32],
                strides=[2, 2, 2],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.8, #0.4, 0.36
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                sigma_ratio = 0.5, #0.8
                initial_sigma_ratio= 0.5, # 1e-3
                ncenter = 512,
                xyz_pooling = None, # None, 'no-stride'
                to_file=None):

    device = opt.device
    input_num = ncenter # opt.model.input_num
    dropout_rate = opt.model.dropout_rate
    temperature = opt.train_loss.temperature
    so3_pooling = opt.model.flag
    input_radius = opt.model.search_radius
    kpconv = opt.model.kpconv
    na = 1 if opt.model.kpconv else 60
    mlp_0 = mlps[0][0]

    # # to accomodate different input_num
    # if input_num > 1024:
    #     sampling_ratio /= (input_num / 1024)
    #     strides[0] = int(2 * (input_num / 1024))
    #     print("Using sampling_ratio:", sampling_ratio)
    #     print("Using strides:", strides)

    dim_in = 1

    print("[MODEL] USING RADIUS AT %f"%input_radius)

    print("[SMOOTH MODEL] INITIAL SIGMA AT %f"%initial_sigma_ratio)

    r0 = initial_radius_ratio * input_radius

    params = {'name': 'Invariant ZPConv Model',
              'backbone': [],
              'propagation': {
                'dim_in': dim_in,
                'dim_out': mlp_0,
                'n_center': ncenter,
                'kernel_size': 1,
                'radius': initial_radius_ratio * input_radius,
                'sigma' : initial_sigma_ratio * r0**2,
                'kpconv': kpconv,
              },
              'dropout_rate':dropout_rate,
              'na': na,
              }

    dim_in = mlp_0

    # process args
    n_layer = len(mlps)

    # [1,2,4,8,...]
    stride_current = 2
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]


    # 512, 256, 128, ...
    num_centers = [ncenter] + [int(input_num / multiplier) for multiplier in stride_multipliers]

    # r(i+1) = sqrt(2) r(i)
    radius_ratio = [initial_radius_ratio * multiplier**0.5 for multiplier in stride_multipliers]

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
            if i == 0 and j == 0:
                continue
            lazy_sample = True # i != 0 or j != 0
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**2)
            kernel_size = 1
            nidx = i

            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i-1]
                # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                neighbor *= 2 # * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                # neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i+1]**(1/sampling_density))
                kernel_size = 1 # if inter_stride < 4 else 3
            else:
                inter_stride = 1

            print(f"At block {i}, layer {j}!")
            print(f'neighbor: {neighbor}')
            print(f'stride: {inter_stride}')
            sigma_to_print = weighted_sigma[nidx]**2 / 3
            print(f'sigma: {sigma_to_print}')
            print(f'radius ratio: {radius_ratio[nidx]}')

            # import ipdb; ipdb.set_trace()

            # one-inter one-intra policy
            block_type = 'inter_block' if opt.model.kpconv else 'separable_block'

            inter_param = {
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
                    'kpconv': kpconv,
                }
            }
            block_param.append(inter_param)

            # intra_param = {
            #     'type': 'intra_block',
            #     'args': {
            #         'dim_in': dim_out,
            #         'dim_out': dim_out,
            #         'dropout_rate': dropout_rate,
            #         'activation': 'leaky_relu',
            #     }
            # }
            # block_param.append(intra_param)

            dim_in = dim_out

        params['backbone'].append(block_param)

    params['outblock'] = {
        'dim_in': dim_in,
        'mlp': out_mlps,
        'pooling': so3_pooling,
        'temperature': temperature,
    }


    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = InvSO3ConvSmoothModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
