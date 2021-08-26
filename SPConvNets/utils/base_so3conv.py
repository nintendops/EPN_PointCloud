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

import vgtk.spconv as zptk
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
class IntraSO3ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out,
                 norm=None, activation='relu', dropout_rate=0):

        super(IntraSO3ConvBlock, self).__init__()

        if norm is not None:
            norm = getattr(nn,norm)
            # norm = nn.InstanceNorm2d

        self.conv = sptk.IntraSO3Conv(dim_in, dim_out)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim__out)

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


class PropagationBlock(nn.Module):
    def __init__(self, params, norm=None, activation='relu', dropout_rate=0):
        super(PropagationBlock, self).__init__()
        self.prop = sptk.KernelPropagation(**params)
        if norm is None:
            norm = nn.InstanceNorm2d #nn.BatchNorm2d
        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)
        self.norm = norm(params['dim_out'], affine=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, frag, clouds):
        x = self.prop(frag, clouds)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return zptk.SphericalPointCloud(x.xyz, feat, x.anchors)

# [b, c1, p1, a] -> [b, c1, k, p2, a] -> [b, c2, p2, a]
class InterSO3ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, sigma, n_neighbor, multiplier, kanchor=60,
                 lazy_sample=None, norm=None, activation='relu', pooling='none', dropout_rate=0):
        super(InterSO3ConvBlock, self).__init__()

        if lazy_sample is None:
            lazy_sample = True

        if norm is not None:
            norm = getattr(nn,norm)
            
        # if norm is None:
        #     norm = nn.InstanceNorm2d #nn.BatchNorm2d

        pooling_method = None if pooling == 'none' else pooling
        self.conv = sptk.InterSO3Conv(dim_in, dim_out, kernel_size, stride,
                                      radius, sigma, n_neighbor, kanchor=kanchor,
                                      lazy_sample=lazy_sample, pooling=pooling_method)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim_out)

        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x, inter_idx=None, inter_w=None):
        input_x = x
        inter_idx, inter_w, sample_idx, x = self.conv(x, inter_idx, inter_w)
        feat = self.norm(x.feats)
        # feat = x.feats

        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return inter_idx, inter_w, sample_idx, zptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class BasicSO3ConvBlock(nn.Module):
    def __init__(self, params):
        super(BasicSO3ConvBlock, self).__init__()

        self.blocks = nn.ModuleList()
        self.layer_types = []
        for param in params:
            if param['type'] == 'intra_block':
                conv = IntraSO3ConvBlock(**param['args'])
            elif param['type'] == 'inter_block':
                conv = InterSO3ConvBlock(**param['args'])
            elif param['type'] == 'separable_block':
                conv = SeparableSO3ConvBlock(param['args'])
            else:
                raise ValueError(f'No such type of SO3Conv {param["type"]}')
            self.layer_types.append(param['type'])
            self.blocks.append(conv)
        self.params = params

    def forward(self, x):
        inter_idx, inter_w = None, None
        for conv, param in zip(self.blocks, self.params):
            if param['type'] in ['inter', 'inter_block', 'separable_block']:
                inter_idx, inter_w, _, x = conv(x, inter_idx, inter_w)
                # import ipdb; ipdb.set_trace()

                if param['args']['stride'] > 1:
                    inter_idx, inter_w = None, None
            elif param['type'] in ['intra_block']:
                # Intra Convolution
                x = conv(x)
            else:
                raise ValueError(f'No such type of SO3Conv {param["type"]}')

        return x

    def get_anchor(self):
        return torch.from_numpy(sptk.get_anchors())

class SeparableSO3ConvBlock(nn.Module):
    def __init__(self, params):
        super(SeparableSO3ConvBlock, self).__init__()

        dim_in = params['dim_in']
        dim_out = params['dim_out']
        norm = getattr(nn,params['norm']) if 'norm' in params.keys() else None
        
        self.use_intra = params['kanchor'] > 1

        self.inter_conv = InterSO3ConvBlock(**params)

        intra_args = {
            'dim_in': dim_out,
            'dim_out': dim_out,
            'dropout_rate': params['dropout_rate'],
            'activation': params['activation'],
        }

        if self.use_intra:
            self.intra_conv = IntraSO3ConvBlock(**intra_args)
        self.stride = params['stride']

        # 1x1 conv for skip connection
        self.skip_conv = nn.Conv2d(dim_in, dim_out, 1)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim_out)
        self.relu = getattr(F, params['activation'])


    def forward(self, x, inter_idx, inter_w):
        '''
            inter, intra conv with skip connection
        '''
        skip_feature = x.feats
        inter_idx, inter_w, sample_idx, x = self.inter_conv(x, inter_idx, inter_w)

        if self.use_intra:
            x = self.intra_conv(x)
        if self.stride > 1:
            skip_feature = zptk.functional.batched_index_select(skip_feature, 2, sample_idx.long())
        skip_feature = self.skip_conv(skip_feature)
        skip_feature = self.relu(self.norm(skip_feature))
        # skip_feature = self.relu(skip_feature)
        x_out = zptk.SphericalPointCloud(x.xyz, x.feats + skip_feature, x.anchors)
        return inter_idx, inter_w, sample_idx, x_out

    def get_anchor(self):
        return torch.from_numpy(sptk.get_anchors())

class ClsOutBlockR(nn.Module):
    def __init__(self, params, norm=None):
        super(ClsOutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        fc = params['fc']
        k = params['k']

        self.outDim = k

        self.linear = nn.ModuleList()
        self.norm = nn.ModuleList()

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            self.norm.append(nn.BatchNorm2d(c))
            c_in = c
        # -----------------------------------------------

        # ------------------ intra conv -----------------
        if 'intra' in params.keys():
            self.intra = nn.ModuleList()
            self.skipconv = nn.ModuleList()
            for intraparams in params['intra']:
                conv = IntraSO3ConvBlock(**intraparams['args'])
                self.intra.append(conv)
                c_out = intraparams['args']['dim_out']

                # for skip convs
                self.skipconv.append(nn.Conv2d(c_in, c_out, 1))
                self.norm.append(nn.BatchNorm2d(c_out))
                c_in = c_out
        # -----------------------------------------------

        # ----------------- pooling ---------------------
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        # BxCxA -> Bx1xA or BxCxA attention weights
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, 1, 1)
        elif self.pooling_method == 'attention2':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, c_in, 1)
        # ------------------------------------------------

        self.fc1 = nn.ModuleList()
        for c in fc:
            self.fc1.append(nn.Linear(c_in, c))
            # self.norm.append(nn.BatchNorm1d(c))
            c_in = c

        self.fc2 = nn.Linear(c_in, self.outDim)

    def forward(self, feats, label=None):
        x_out = feats
        norm_cnt = 0
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            x_out = F.relu(norm(x_out))
            norm_cnt += 1

        # mean pool at xyz
        out_feat = x_out
        x_out = x_out.mean(2, keepdim=True)

        # group convolution after mean pool
        if hasattr(self, 'intra'):
            x_in = zptk.SphericalPointCloud(None, x_out, None)
            for lid, conv in enumerate(self.intra):
                skip_feat = x_in.feats
                x_in = conv(x_in)

                # skip connection
                norm = self.norm[norm_cnt]
                skip_feat = self.skipconv[lid](skip_feat)
                skip_feat = F.relu(norm(skip_feat))
                x_in = zptk.SphericalPointCloud(None, skip_feat + x_in.feats, None)
                norm_cnt += 1
            x_out = x_in.feats


        # mean pooling
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=3).mean(dim=2)
        elif self.pooling_method == 'debug':
            # for debug only
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.mean(2).max(-1)[0]
        ############## DEBUG ONLY ######################
        elif label is not None:
            def to_one_hot(label, num_class):
                '''
                label: [B,...]
                return [B,...,num_class]
                '''
                comp = torch.arange(num_class).long().to(label.device)
                for i in range(label.dim()):
                    comp = comp.unsqueeze(0)
                onehot = label.unsqueeze(-1) == comp
                return onehot.float()
            x_out = x_out.mean(2)
            label = label.squeeze()
            if label.dim() == 2:
                cdim = x_out.shape[1]
                label = label.repeat(1,5)[:,:cdim]
            confidence = to_one_hot(label, x_out.shape[2])
            if confidence.dim() < 3:
                confidence = confidence.unsqueeze(1)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        elif self.pooling_method.startswith('attention'):
            x_out = x_out.mean(2)
            out_feat = self.attention_layer(x_out)  # Bx1XA or BxCxA
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        # fc layers
        for linear in self.fc1:
            # norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            # x_out = F.relu(norm(x_out))
            x_out = F.relu(x_out)
            # norm_cnt += 1

        x_out = self.fc2(x_out)

        return x_out, out_feat.squeeze()

class ClsOutBlockPointnet(nn.Module):
    def __init__(self, params, norm=None, debug=False):
        super(ClsOutBlockPointnet, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        fc = params['fc']
        k = params['k']
        na = params['kanchor']

        self.outDim = k

        self.linear = nn.ModuleList()
        self.norm = nn.ModuleList()

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            self.norm.append(nn.BatchNorm2d(c))
            c_in = c
        # -----------------------------------------------

        # ----------------- pooling ---------------------
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        # BxCxA -> Bx1xA or BxCxA attention weights
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, 1, 1)

        # ------------------------------------------------

        # self.fc1 = nn.ModuleList()
        # for c in fc:
        #     self.fc1.append(nn.Linear(c_in, c))
        #     # self.norm.append(nn.BatchNorm1d(c))
        #     c_in = c
        self.pointnet = sptk.PointnetSO3Conv(c_in, c_in, na)
        self.norm.append(nn.BatchNorm1d(c_in))
        self.fc2 = nn.Linear(c_in, self.outDim)

        self.debug = debug
        
    def forward(self, x, label=None):
        x_out = x.feats

        if self.debug:
            return x_out[:,:40].mean(-1).mean(-1),None
        
        norm_cnt = 0
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            x_out = F.relu(norm(x_out))
            # x_out = F.relu(x_out)
            norm_cnt += 1

        out_feat = x_out
        x_in = zptk.SphericalPointCloud(x.xyz, out_feat, x.anchors)

        x_out = self.pointnet(x_in)

        norm = self.norm[norm_cnt]
        norm_cnt += 1
        x_out = F.relu(norm(x_out))
        # x_out = F.relu(x_out)
        
        # mean pooling
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=2)
        elif self.pooling_method == 'debug':
            # for debug only
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.max(2)[0]
        elif self.pooling_method.startswith('attention'):
            out_feat = self.attention_layer(x_out)  # Bx1XA or BxCxA
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        x_out = self.fc2(x_out)

        return x_out, out_feat.squeeze()

class InvOutBlockR(nn.Module):
    def __init__(self, params, norm=None):
        super(InvOutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']

        # TODO
        if 'intra' in params.keys():
            pass

        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        self.norm = nn.ModuleList()

        # Attention layer
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(mlp[-1], 1, 1)
            # self.attention_layer = nn.Conv1d(c_in, 1, 1)

        # 1x1 Conv layer
        self.linear = nn.ModuleList()
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            # self.linear.append(nn.Linear(c_in, c))
            self.norm.append(nn.InstanceNorm2d(c, affine=False))
            c_in = c

        # self.out_norm = nn.BatchNorm1d(c_in)


    def forward(self, feats):
        x_out = feats
        end = len(self.linear)

        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if lid != end - 1:
                norm = self.norm[lid]
                x_out = F.relu(norm(x_out))

        out_feat = x_out.mean(2)

        # mean pooling
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=3).mean(dim=2)
        elif self.pooling_method == 'debug':
            # for debug only
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.mean(2).max(-1)[0]
        elif self.pooling_method == 'attention':
            x_out = x_out.mean(2)
            out_feat = self.attention_layer(x_out)
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
            out_feat = confidence.squeeze()
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        # batch norm in the last layer?
        # x_out = self.out_norm(x_out)

        return F.normalize(x_out, p=2, dim=1), out_feat


class InvOutBlockPointnet(nn.Module):
    def __init__(self, params, norm=None):
        super(InvOutBlockPointnet, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        c_out = mlp[-1]

        na = params['kanchor']

        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        self.pointnet = sptk.PointnetSO3Conv(c_in,c_out,na)

        # Attention layer
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_out, 1, 1)


        # self.out_norm = nn.BatchNorm1d(c_out, affine=True)


    def forward(self, x):
        # nb, nc, np, na -> nb, nc, na
        x_out = self.pointnet(x)
        out_feat = x_out

        # mean pooling
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.max(2)[0]
        elif self.pooling_method == 'attention':
            attw = self.attention_layer(x_out)
            confidence = F.softmax(attw * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
            confidence = confidence.squeeze()
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        # batch norm in the last layer?
        # x_out = self.out_norm(x_out)
        return F.normalize(x_out, p=2, dim=1), F.normalize(out_feat, p=2, dim=1)

class InvOutBlockMVD(nn.Module):
    def __init__(self, params, norm=None):
        super(InvOutBlockMVD, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        c_out = mlp[-1]
        na = params['kanchor']

        # Attention layer
        self.temperature = params['temperature']
        self.attention_layer = nn.Sequential(nn.Conv2d(c_in, c_in, 1), \
                                                 nn.ReLU(inplace=True), \
                                                 nn.Conv2d(c_in,c_in,1))

        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        self.pointnet = sptk.PointnetSO3Conv(c_in,c_out,na)

        # self.out_norm = nn.BatchNorm1d(c_out, affine=True)


    def forward(self, x):
        # nb, nc, np, na -> nb, nc, na

        # attention first
        nb, nc, np, na = x.feats.shape

        attn = self.attention_layer(x.feats)
        attn = F.softmax(attn, dim=3)

        # nb, nc, np, 1
        x_out = (x.feats * attn).sum(-1, keepdim=True)
        x_in = zptk.SphericalPointCloud(x.xyz, x_out, None)

        # nb, nc
        x_out = self.pointnet(x_in).view(nb, -1)

        return F.normalize(x_out, p=2, dim=1), attn


# outblock for rotation regression model
class SO3OutBlockR(nn.Module):
    def __init__(self, params, norm=None):
        super(SO3OutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        # fc = params['fc']
        # k = params['k']
        # self.outDim = k

        self.linear = nn.ModuleList()
        # self.norm = nn.ModuleList()
        self.temperature = params['temperature']
        self.representation = params['representation']
        self.attention_layer = nn.Conv2d(mlp[-1], 1, (1,1))

        # out channel equals 4 for quaternion representation, 6 for ortho representation
        self.regressor_layer = nn.Conv2d(mlp[-1],4,(1,1))

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            # self.norm.append(nn.BatchNorm2d(c))
            c_in = c

    def forward(self, feats):
        x_out = feats
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            # norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            x_out = F.relu(x_out)

        # mean pool at xyz ->  BxCxA
        x_out = x_out.mean(2)

        # attention weight
        attention_wts = self.attention_layer(x_out)  # Bx1XA
        confidence = F.softmax(attention_wts * self.temperature, dim=2).view(x_out.shape[0], x_out.shape[2])
        # regressor
        y = self.regressor_layer(x_out) # Bx6xA
        return confidence, y

# outblock for relative rotation regression
class RelSO3OutBlockR(nn.Module):
    def __init__(self, params, norm=None):
        super(RelSO3OutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        na = params['kanchor']

        self.pointnet = sptk.PointnetSO3Conv(c_in, c_in, na)
        c_in = c_in * 2

        self.linear = nn.ModuleList()

        self.temperature = params['temperature']
        rp = params['representation']

        if rp == 'quat':
            self.out_channel = 4
        elif rp == 'ortho6d':
            self.out_channel = 6
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%rp)

        self.attention_layer = nn.Conv2d(mlp[-1], 1, (1,1))

        # out channel equals 4 for quaternion representation, 6 for ortho representation
        self.regressor_layer = nn.Conv2d(mlp[-1],self.out_channel,(1,1))

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, (1,1)))
            c_in = c


    def forward(self, f1, f2, x1, x2):
        # nb, nc, np, na -> nb, nc, na
        sp1 = zptk.SphericalPointCloud(x1, f1, None)
        sp2 = zptk.SphericalPointCloud(x2, f2, None)

        f1 = self._pooling(sp1)
        f2 = self._pooling(sp2)

        nb = f1.shape[0]
        na = f1.shape[2]

        # expand and concat into metric space (nb, nc*2, na_tgt, na_src)
        f2_expand = f2.unsqueeze(-1).expand(-1,-1,-1,na).contiguous()
        f1_expand = f1.unsqueeze(-2).expand(-1,-1,na,-1).contiguous()

        x_out = torch.cat((f1_expand,f2_expand),1)

        # fc layers with relu
        for linear in self.linear:
            x_out = linear(x_out)
            x_out = F.relu(x_out)

        attention_wts = self.attention_layer(x_out).view(nb, na, na)
        confidence = F.softmax(attention_wts * self.temperature, dim=1)
        y = self.regressor_layer(x_out)

        # return: [nb, na, na], [nb, n_out, na, na]
        return confidence, y


    def _pooling(self, x):
        # [nb, nc, na]
        x_out = self.pointnet(x)
        x_out = F.relu(x_out)

        return x_out
        # return feats.mean(2)
