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

ANC = 49

def vis_scale(tensor, k):
    return tensor.sum(0).mean().item() + k * tensor.sum(0).std().item()

class EquiZPConvModel(nn.Module):
    def __init__(self, params):
        super(EquiZPConvModel, self).__init__()

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicZPConvBlock(block_param))
        self.outblock = M.EquiOutBlock(params['outblock'])

        self.na_in = L.get_anchors(params['anchors'][0]).shape[0]
        self.invariance = False

    def forward(self, x):
        x = M.preprocess_input(x, self.na_in)

        for block_i, block in enumerate(self.backbone):
            x = block(x)

            # # --------------- debug only ----------------------------
            # feat = x[1][0,...,0]
            # print(feat.shape)
            # feat = F.normalize(feat, p=2, dim=0)
            # print(feat.std(1).mean().item())
            # import ipdb; ipdb.set_trace()

            # # --------------- debug only ----------------------------
            # vmin = min(x.feat[0].sum(1).min().item(), x.feat[8].sum(1).min().item())
            # vmax = max(x.feat[0].sum(1).max().item(), x.feat[8].sum(1).max().item())
            # vmin = min(vis_scale(x.feat[0], -1.5), vis_scale(x.feat[8], -1.5))
            # vmax = min(vis_scale(x.feat[0], 1.5), vis_scale(x.feat[8], 1.5))
            # import ipdb; ipdb.set_trace()
            # radius = 0.005*(block_i+1) # 0.016 * 0.15 * 3**0.5 # 
            # vgtk.pc.visualize_all_np("data/visualization/debug/test_idx_0_block%d"%block_i, 0, x0, x, vmin=vmin, vmax=vmax, radius=radius)
            # vgtk.pc.visualize_all_np("data/visualization/debug/test_idx_8_block%d"%block_i, 8, x0, x, vmin=vmin, vmax=vmax, radius=radius)

        x = self.outblock(x[1])
        return x

    # fetch output anchor
    def get_anchor(self):
        return self.backbone[-1].get_anchor()

# 5-layer version 
# def build_model(device,
#                 mlps=[[16], [32], [64], [128], [128]], 
#                 out_mlps=[64, 64, 32],
#                 anchors=[8, 8, 8, 8, 8, 8],# [ANC]*6, [62, 49, 30, 24, 12, 12]
#                 # kernel_size=[3, 4, 5, 4, 3],#[4, 5, 6, 5, 4],[5]*5
#                 kernel_density = 1.0,
#                 kernel_multiplier = 3,
#                 radius_ratio=[0.3, 0.46, 0.6, 0.72, 0.88], #[0.16, 0.32, 0.48, 0.64, 0.8]
#                 # n_neighbor=[32, 48, 48, 32, 16],  # [32, 48, 48, 32, 16]
#                 sampling_ratio=0.65,
#                 input_radius=0.15 * 3**0.5,
#                 strides=[2,2,2,2,2], # [2]*5
#                 aperture_ratio=[0.25, 0.4, 0.55, 0.7, 1.0], # [1.0]*5
#                 sigma_ratio=1e-3,
#                 intra_sigma_ratio = 0.01,
#                 input_num=2048,
#                 dropout_rate=0.,
#                 to_file=None):

#     # (temp) some extra params
#     intra_aperature = [np.pi/2, np.pi/2, 0.75 * np.pi, 0.75 * np.pi, np.pi]
    
# debug version
def build_model(device,
                mlps=[[16],[32],[64],[64,64]], 
                out_mlps=[64, 32],
                strides=[2, 2, 2, 32],
                initial_radius_ratio = 0.25, 
                anchors=[42, 42, 12, 12],# [ANC]*4, [92, 42, 42, 12]
                aperture_ratio = [0.25, 0.25, 0.5, 1.0],
                sampling_ratio = 2,
                kernel_density = 2,
                kernel_multiplier = 2,
                input_radius=0.15 * 3**0.5,
                sigma_ratio= 7e-4,
                intra_sigma_ratio = 0.1,
                input_num=2048,
                dropout_rate=0.,
                to_file=None):

#     # (temp) some extra params
    intra_aperature = [np.pi/2]*5
    intra_ann = [7,7,6,6]

# 4-layer version
# def build_model(device,
#                 mlps=[[32,32], [32,64], [64,64], [128]], 
#                 out_mlps=[64, 32],
#                 strides=[2,2,2,4],
#                 initial_radius_ratio = 0.24, 
#                 anchors=[42, 42, 42, 12],# [ANC]*4, [92, 42, 42, 12]
#                 aperture_ratio = [0.25, 0.35, 0.5, 1.0],
#                 sampling_ratio = 2,
#                 kernel_density = 1.5,
#                 kernel_multiplier = 2,
#                 input_radius=0.15 * 3**0.5,
#                 sigma_ratio= 6e-4,
#                 intra_sigma_ratio = 0.1,
#                 input_num=2048,
#                 dropout_rate=0.,
#                 to_file=None):

    # (temp) some extra params
    intra_aperature = [0.15, 0.15, 0.15, 0.4]
    intra_ann = [7,7,7,6]

    params = {'name': 'Invariant ZPConv Model',
              'backbone': [],
              'anchors': anchors,
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

    radius_ratio = [initial_radius_ratio * multiplier**0.25 for multiplier in stride_multipliers][1:]
    radius = [r * input_radius for r in radius_ratio] 
    aperture = [2 * ar * np.pi for ar in aperture_ratio]
    n_neighbor = [int(sampling_ratio * num_centers[i] * radius_ratio[i]**3) for i in range(n_layer)]

    # n_neighbor[-1] = 128

    inter_ann = [int(aperture_ratio[i] * n_neighbor[i]) for i in range(n_layer)]

    intra_aperature = [2 * ar * np.pi for ar in intra_aperature]
    # intra_ann=[1 + int(anchors[i] * intra_aperature[i]/(2*np.pi)) for i in range(len(intra_aperature))]

    kernel_size = [1 + int(kernel_density * math.sqrt(inter_ann[i] / kernel_multiplier)) for i in range(n_layer)]
    intra_kernel_size = [2]*5

    # Compute sigma
    weighted_sigma = [sigma_ratio * radius[i] / radius[0] for i in range(n_layer)]
    weighted_intra_sigma = [intra_sigma_ratio * intra_aperature[i] / intra_aperature[0] for i in range(n_layer)]

    anchors = anchors + [anchors[-1]]

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            # stride at first
            inter_stride = 1
            if j == 0:
                inter_stride = strides[i]
            
            # increase anchor at last
            intra_anchor_out = anchors[i]
            if j == len(block) - 1:
                intra_anchor_out = anchors[i+1]

            # weighted_sigma = sigma_ratio
            
            # one-inter one-intra policy
            inter_param = {
                'type': 'inter_block',
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size[i],
                    'stride': inter_stride,
                    'radius': radius[i],
                    'aperture': aperture[i],
                    'sigma': weighted_sigma[i],
                    'anchors': anchors[i],
                    'n_neighbor': n_neighbor[i],
                    'anchor_nn': inter_ann[i],
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier
                }
            }
            block_param.append(inter_param)

            # ----------- INTRA CONV -------------------------
            intra_param = {
                'type': 'intra_block',
                'args': {
                    'dim_in': dim_out,
                    'dim_out': dim_out,
                    'kernel_size': intra_kernel_size[i],
                    'aperture': intra_aperature[i],
                    'sigma': weighted_intra_sigma[i],
                    'anchor_nn': intra_ann[i],
                    'anchor_in': anchors[i],
                    'anchor_out': intra_anchor_out,
                    'dropout_rate': dropout_rate,
                }
            }
            block_param.append(intra_param)

            # ------------  ANCHOR PROP  -----------------------
            # if intra_anchor_out != anchors[i]:
            #     intra_param = {
            #         'type': 'anchor_prop',
            #         'args': {
            #             'anchor_in': anchors[i],
            #             'anchor_out': intra_anchor_out,
            #             'sigma': 1.0,
            #             'k': 6
            #         }
            #     }
            #     block_param.append(intra_param)

            dim_in = dim_out

        params['backbone'].append(block_param)

    params['outblock'] = {
        'dim_in': dim_in,
        'mlp': out_mlps,
    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = EquiZPConvModel(params).to(device)
    return model

# TODO
def build_model_from(opt):
    return build_model(opt.device, input_num=opt.model.input_num, dropout_rate=opt.model.dropout_rate)

