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

ADD_INTRA = False

class ClsSO3ConvModel(nn.Module):
    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
        self.outblock = M.ClsOutBlockR(params['outblock'])

        # (TODO) remove hard coded anchor dim
        self.na_in = 60
        self.invariance = True

    def forward(self, x, rlabel=None):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = M.preprocess_input(x, self.na_in, False)
        # x = M.preprocess_input(x, 1, False)

        for block_i, block in enumerate(self.backbone):
            # print('block_i: %d'%block_i)
            x = block(x)
            # import ipdb; ipdb.set_trace()

        x = self.outblock(x.feats, rlabel)
        return x

    def get_anchor(self):
        return self.backbone[-1].get_anchor()


# Light Version
# def build_model(device,
#                 mlps=[[16,16], [32,32], [64,64]],
#                 out_mlps=[64],
#                 strides=[2, 2, 8],
#                 initial_radius_ratio = 0.2,
#                 sampling_ratio = 0.5,
#                 sampling_density = 0.5,
#                 kernel_density = 1,
#                 kernel_multiplier = 2,
#                 input_radius=1.0,
#                 sigma_ratio= 1e-3,
#                 input_num=2048,
#                 dropout_rate=0.,
#                 xyz_pooling = "no-stride", 
#                 so3_pooling = "max",
#                 to_file=None):


# Full Version
def build_model(device,
                mlps=[[32,32], [64,64], [128,128], [256]],
                out_mlps=[256],
                strides=[2, 2, 2, 4],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.5,
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                input_radius=1.0,
                sigma_ratio= 1e-3, # 0.1
                input_num=2048,
                dropout_rate=0.,
                temperature=1,
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

    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio]

    # n_neighbor = [int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density)) for i in range(n_layer + 1)]
    # n_neighbor = [32, 64]

    # kernel_size = [1 + int(kernel_density * math.sqrt(n_neighbor[i] / kernel_multiplier)) for i in range(n_layer)]
    # kernel_size = [2,2,2,2]

    # Compute sigma
    weighted_sigma = [sigma_ratio * radii[i]**2 * stride_multipliers[i] for i in range(n_layer + 1)]
    # weighted_sigma = [ (sigma_ratio * radii[i])**2 for i in range(n_layer + 1)]

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            stride_conv = i == 0 or xyz_pooling != 'stride'

            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))          
            kernel_size = 2
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i]
                nidx = i if i==0 else i+1
                if stride_conv:
                    neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i+1]**(1/sampling_density))
                    kernel_size = 2 # if inter_stride < 4 else 3
            else:
                inter_stride = 1
                nidx = i+1

            print(f"At block {i}, layer {j}!")
            print(f'neighbor: {neighbor}')
            # print(f'stride: {inter_stride}')
            # print(f'sigma: {weighted_sigma[nidx]}')
            print(f'radius {radius_ratio[nidx]}')
            # import ipdb; ipdb.set_trace()

            # one-inter one-intra policy
            conv_param = {
                'type': 'interintra_block',
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
            block_param.append(conv_param)

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

    ################ IF ADD INTRA IN LAST LAYER #########################
    if ADD_INTRA:
        intra_mlp = [256]
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
            'temperature': temperature,
            'fc': [128],
            'k': 40,
            'pooling': so3_pooling,
        }
    ####################################################################
    else:
        params['outblock'] = {
            'dim_in': dim_in,
            'mlp': out_mlps,
            'fc': [64],
            'k': 40,
            'pooling': so3_pooling,
            'temperature': temperature,
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
    return build_model(opt.device, input_num=opt.model.input_num, \
                       dropout_rate=opt.model.dropout_rate, temperature=opt.train_loss.temperature,\
                       to_file=outfile_path, so3_pooling = opt.model.flag)
