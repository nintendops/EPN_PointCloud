import math
import os
import numpy as np
import time
from collections import namedtuple
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import vgtk.zpconv as zptk
import vgtk.so3conv as sptk

# [nb, np, 3] -> [nb, 3, np] x [nb, 1, np, na]
def preprocess_input(x, na, add_center=True):
    has_normals = x.shape[2] == 6
    # add a dummy center point at index zero
    if add_center and not has_normals:
        center = x.mean(1, keepdim=True)
        x = torch.cat((center,x),dim=1)[:,:-1]
    xyz = x[:,:,:3]
    return zptk.SphericalPointCloud(xyz.permute(0,2,1).contiguous(), sptk.get_occupancy_features(x, na, add_center), None)

def get_inter_kernel_size(band):
    return np.arange(band + 1).sum() + 1

def get_intra_kernel_size(band):
    return np.arange(band + 1).sum() + 1


# [b, c1, p, a] -> [b, c1, k, p, a] -> [b, c2, p, a]
class IntraZPConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size,
                 aperture, sigma,
                 anchor_nn, anchor_in, anchor_out=None,
                 norm=None, activation='relu', dropout_rate=0):
        super(IntraZPConvBlock, self).__init__()

        if norm is None:
            norm = nn.BatchNorm2d
        self.conv = zptk.IntraZPConv(dim_in, dim_out, kernel_size,
                                     aperture, sigma,
                                     anchor_nn, anchor_in, anchor_out)
        self.norm = norm(dim_out)

        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        # [b, 3, p] x [b, c1, p]
        x = self.conv(x)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        # [b, 3, p] x [b, c2, p]
        return zptk.SphericalPointCloud(x.xyz, feat, x.anchors)


# [b, c1, p1, a] -> [b, c1, k, p2, a] -> [b, c2, p2, a]
class InterZPConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, aperture, sigma,
                 anchors, n_neighbor, anchor_nn, multiplier,
                 lazy_sample=None, norm=None, activation='relu', dropout_rate=0):
        super(InterZPConvBlock, self).__init__()

        if lazy_sample is None:
            lazy_sample = True
        if norm is None:
            norm = nn.BatchNorm2d

        self.conv = zptk.InterZPConv(dim_in, dim_out, kernel_size, stride,
                                     radius, aperture, sigma,
                                     anchors, n_neighbor, anchor_nn, multiplier=multiplier, lazy_sample=lazy_sample)
        self.norm = norm(dim_out)
        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x, inter_idx=None, inter_w=None):
        inter_idx, inter_w, x = self.conv(x, inter_idx, inter_w)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        ## TODO no need to add self.training
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return inter_idx, inter_w, zptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class BasicZPConvBlock(nn.Module):
    def __init__(self, params):
        super(BasicZPConvBlock, self).__init__()

        self.blocks = nn.ModuleList()
        self.layer_types = []
        for param in params:
            # if param['type'] == 'intra':
            #     conv = zptk.IntraZPConv(**param['args'])
            # elif param['type'] == 'inter':
            #     conv = zptk.InterZPConv(**param['args'])
            if param['type'] == 'intra_block':
                conv = IntraZPConvBlock(**param['args'])
            elif param['type'] == 'inter_block':
                conv = InterZPConvBlock(**param['args'])
            elif param['type'] == 'anchor_prop':
                conv = zptk.AnchorProp(**param['args'])
            else:
                raise ValueError(f'No such type of ZPConv {param["type"]}')
            self.layer_types.append(param['type'])
            self.blocks.append(conv)
        self.params = params

    def forward(self, x):
        inter_idx, inter_w = None, None
        for conv, param in zip(self.blocks, self.params):

            end = time.time()

            if param['type'] in ['inter', 'inter_block']:
                inter_idx, inter_w, x = conv(x, inter_idx, inter_w)
                if param['args']['stride'] > 1:
                    inter_idx, inter_w = None, None
            else:
                x = conv(x)

            # print(f'time {time.time()-end}, type {param["type"]}')
            end = time.time()

        return x

    # fetch output anchor
    # TODO too hard coded
    def get_anchor(self):
        if self.layer_types[-1] == 'intra_block':
            return self.blocks[-1].conv.anchor_out
        elif self.layer_types[-1] == 'anchor_prop':
            return self.blocks[-1].anchor_out
        else:
            return self.blocks[-1].conv.anchors

########################## OUTBLOCK ###############################################
'''
equivariant pooling: [nb, c_out, nc, a] -> [nb, c_out, a] (equivariance)
invariant pooling: [nb, c_in, nc, a] -> mlp -> [nb, c_out, nc, a] -> [nb, c_out] (invariance)
'''

class EquiOutBlock(nn.Module):
    def __init__(self, params, norm=None):
        super(EquiOutBlock, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']

        if norm is None:
            norm = nn.BatchNorm1d

        self.linear = nn.ModuleList()
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            # self.linear.append(nn.Linear(c_in, c))
            c_in = c

        self.norm = norm(c_in)

    def forward(self, feats):
        x_out = feats

        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if lid != end - 1:
                x_out = F.relu(x_out)


        # mean pooling at point dim
        x_out = x_out.mean(dim=2)

        # batch norm?
        # x_out = self.norm(x_out)

        # normalize
        nb, c_out, na = x_out.shape
        points_norm = F.normalize(x_out.view(nb,-1),p=2,dim=1).view(nb, c_out, na).contiguous()
        inv_feat = points_norm.mean(dim=2)

        return points_norm, inv_feat


class InvOutBlock(nn.Module):
    def __init__(self, params, norm=None):
        super(InvOutBlock, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']

        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        self.norm = nn.ModuleList()

        self.linear = nn.ModuleList()
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            self.norm.append(nn.BatchNorm2d(c))
            # self.linear.append(nn.Linear(c_in, c))
            c_in = c

        self.out_norm = nn.BatchNorm1d(c_in)


    def forward(self, feats):
        sphere = feats.mean(dim=2)
        x_out = feats

        end = len(self.linear)

        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if lid != end - 1:
                norm = self.norm[lid]
                x_out = F.relu(norm(x_out))

        # mean pooling
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=3).mean(dim=2)
        elif self.pooling_method == 'debug':
            # for debug only
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.mean(2).max(-1)[0]
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        # batch norm in the last layer?
        x_out = self.out_norm(x_out)

        return F.normalize(x_out, p=2, dim=1), sphere
        # return x_out, sphere
