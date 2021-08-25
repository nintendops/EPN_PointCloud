
import numpy as np
from utils.evaluation import ply_io
from sklearn.decomposition import PCA

def create_anchors_from_ply(ply_path):
    # Mx3
    pts = read_point_cloud(ply_path)[:, :3]
    norms = np.sqrt(np.sum(pts**2, axis=1))
    pts_selected = pts[np.where(norms>0.5)]
    pts_selected /= np.expand_dims(norms[np.where(norms>0.5)],1)
    return pts_selected

def find_anchor_lrf(feature, anchor):
    main_idx = np.argmax(np.sum(feature**2, 1))
    zaxis = anchor[main_idx]
    yidx = []
    for idx,ax in enumerate(anchor):
        if abs(np.dot(zaxis,ax)) < 1e-4:
            yidx.append(idx)
    #ycands = anchor[yidx]
    
    xaxis = anchor[yidx[np.argmax(np.sum(feature[yidx]**2,1))]]
    yaxis = np.cross(xaxis, zaxis)
    rotation = np.array([xaxis, yaxis, zaxis],dtype=np.float32)
    return rotation
    

def pca_transform(feats, anchor):
    N = feats.shape[0]
    newf = []
    for i in range(N):
        feature = feats[i]
        r = find_anchor_lrf(feature, anchor)
        newf.append(transform_anchor_feat(feature[None], anchor, r, 0.1)[0])
    return np.array(newf, dtype=np.float32)

def hom_transform(points, T, translation=True):
    if translation:
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = points @ T.T
        points = points[:, :3]
    else:
        points = points[:, :3] @ T[:3, :3].T
    return points


def read_point_cloud(path):
    ply = ply_io.read_ply(path)
    return np.array(ply['points'])


def read_key_point(path):
    with open(path, 'r') as fin:
        point_ids_str = fin.readlines()
    point_ids = [int(i) for i in point_ids_str if i.strip()]
    return np.array(point_ids)


def read_feature(path, descriptor_name, 
                 rotation_equivariance, anchor_ply):
    def read_npz(path):
        return np.load(path)['data']
    def read_np(path):
        return np.load(path)
    if descriptor_name == 'ours':
        if rotation_equivariance == 'mean':
            return np.mean(read_np(path), 1)
        # elif rotation_equivariance == 'pca':
        #     return None
        else:
            return read_np(path)
    elif descriptor_name == '3DSmooth':
        return read_npz(path)
    else:
        raise ValueError('No such descriptor')


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def transform_anchor_feat(feats, anchors, T, sigma):
    anchors_T = hom_transform(anchors, T, translation=False)
    # Spherical interpolation
    dists = np.sum((anchors[:, None] * anchors_T[None]), 2) - 1.0
    indices = np.argsort(-dists, 1, )[:, :3]
    offsets_k = np.arange(anchors.shape[0])[:, None]
    offsets_n = np.arange(feats.shape[0])[:, None, None, None]
    offsets_c = np.arange(feats.shape[2])

    index_weights = offsets_k*anchors.shape[0]+indices
    weights = np.take(dists, index_weights)
    influences = softmax(weights/sigma, 1)

    index_feats = (offsets_n*anchors.shape[0]+indices[..., None])*feats.shape[2]+offsets_c

    new_feats = np.take(feats, index_feats)
    feats_T = (new_feats * influences[..., None]).sum(2)
    return feats_T

def get_feat_format(descriptor_name):
    if descriptor_name == 'ours':
        return "feature%d.npy"
    elif descriptor_name == '3DSmooth':
        return "_cloud_bin_%d.ply_0.150000_16_1.750000_3DSmoothNet.npz"
    else:
        raise ValueError('No such descriptor')


def read_gt_log(path):
    fragment_pairs = []
    gt_transforms = []

    with open(path, 'r') as fin:
        lines = fin.readlines()
    for i in range(len(lines)//5):
        line = lines[i*5]
        data = line.split()
        fragment_pairs.append([int(data[0]), int(data[1])])
        gt_transform = []
        for j in range(4):
            line = lines[i*5+j+1]
            gt_transform.append(list(map(float, line.split())))
        gt_transforms.append(gt_transform)

    return np.array(fragment_pairs), np.array(gt_transforms)