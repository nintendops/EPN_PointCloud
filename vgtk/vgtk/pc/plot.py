import os
import numpy as np
from colour import Color
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from vgtk.pc import save_ply


red = Color('red')
blue = Color('blue')
white = Color('black')
crange = [np.array(c.rgb) for c in list(red.range_to(blue,1000))]
crange = np.array(crange)
wrrange = [np.array(c.rgb) for c in list(white.range_to(red,1000))]
wrrange = np.array(wrrange)

def clip_to_crange(x, spectrum, xmin, xmax):
    x = x.squeeze()
    cscale = len(spectrum)

    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax

    x = (x - xmin) * cscale / (xmax - xmin)
    x = x.astype(np.int32)
    x = np.clip(x, 0, cscale-1)
    return spectrum[x]


def visualize_one_spheres_np(points, anchors, sphere_path, output_path, vmin=None, vmax=None, radius=0.5):
    '''
    Visualize features defined on a single sphere
    points: np array na x c
    anchors:  np array na x 3
    '''
  
    if not output_path.endswith('.ply'):
        output_path += '.ply'
    anchorsDense = create_anchors_from_ply(sphere_path)
    # small var: red. large var: blue
    densef = propagation(points, anchors, anchorsDense)
    densef = densef.mean(axis=1)
    densec = 255 * clip_to_crange(densef, crange, vmin, vmax) # na x 3 rgb
    anchor_xyz = radius * anchorsDense
    spheres = np.concatenate((anchor_xyz,densec),axis=1)
    save_ply(output_path, spheres, use_color=True)


def visualize_point_feature_np(points, feat, output_path, vmin=None, vmax=None):
    '''
    Visualize point feature given 1D statistics 
    points:  Nx3
    feat: N
    '''
    if not output_path.endswith('.ply'):
        output_path += '.ply'
    color = 255 * clip_to_crange(feat, wrrange, vmin, vmax)
    c_xyz = np.concatenate((xyz,color), axis=1)
    save_ply(output_path, c_xyz, use_color=True)
