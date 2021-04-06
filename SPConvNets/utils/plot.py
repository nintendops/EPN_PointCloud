import os
import numpy as np
from colour import Color
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.pc.io import write_color_pointcloud_as_ply as wpp

# ------------ let's first generate some color spectrums --------
red = Color('red')
blue = Color('blue')
crange = [np.array(c.rgb) for c in list(red.range_to(blue,1000))]
crange = np.array(crange)
white = Color('black')
wrrange = [np.array(c.rgb) for c in list(white.range_to(red,1000))]
wrrange = np.array(wrrange)

def clip_to_crange(x, spectrum, xmin, xmax):
    x = x.squeeze()
    cscale = len(spectrum)
    x = (x - xmin) * cscale / (xmax - xmin)
    x = x.astype(np.int32)
    x = np.clip(x, 0, cscale-1)
    return spectrum[x]

# ---------------------------------------------------------------

def visualize_point_efov(xyz, freqs, path):
    '''
    Visualize the effective field of view of the convolution
    xyz: torch tensor BxNx3
    freqs: torch tensor ncxAxN
    '''

    freqs = freqs[0].sum(0).cpu().numpy()
    #freqs = freqs.sum(0).sum(0).cpu().numpy()
    xyz = xyz[0].cpu().numpy()

    color = 255 * clip_to_crange(freqs, wrrange, xmax=int(1 + freqs[0]))
    c_xyz = np.concatenate((xyz,color), axis=1)
    wpp(c_xyz, path, use_color=True)


def visualize_one_spheres_np(points, anchors, vmin=None, vmax=None, radius=0.5, name='debug', torch_in=True, output_folder=None):
    '''
    points: np array na x c
    anchors:  np array na x 3
    '''
    if torch_in:
        bsample = 0
        points = points[bsample].cpu().numpy()
        anchors = anchors.cpu().numpy()
    
    anchorsDense = create_anchors_from_ply("anchors/sphere1962.ply")
    # small var: red. large var: blue
    densef = DensePropagation(points, anchors, anchorsDense)
    if vmin is None:
        vmin = densef.min()
    if vmax is None:
        vmax = densef.max()
    densef = densef.mean(axis=1)

    densec = 255 * clip_to_crange(densef, crange, vmin, vmax) # na x 3 rgb
    anchor_xyz = radius * anchorsDense
    spheres = np.concatenate((anchor_xyz,densec),axis=1)

    if output_folder is None:
        output_folder = 'data/visualization/sphere_features'
    create_dir(output_folder)
    wpp(spheres, os.path.join(output_folder, name + '_sphere.ply'), use_color=True)


def visualize_all_spheres_np(support, samples, group_idx, points, anchors, radius, step=64, name='debug', torch_in=True, output_folder=None):
    '''
    support: ns x 3
    samples: nc x 3
    group_idx: nc x nn
    points: nc x na x c
    anchors: na x 3
    '''

    '''
    convert to numpy
    '''

    if torch_in:
        bsample = 0
        support = support[bsample].cpu().numpy()
        if samples is not None:
            samples = samples[bsample].cpu().numpy()
        group_idx = group_idx[bsample].cpu().numpy()
        points = points[bsample].detach().cpu().numpy()
        anchors = anchors.cpu().numpy()
    
    anchorsDense = create_anchors_from_ply("anchors/sphere1962.ply")
    spheres = []
    nnpts = []
        
    nc, nn = group_idx.shape
    nbrs = np.take(support, group_idx, axis=0) # nc x nn x 3
    colorstep = np.linspace(0,1000,nc, dtype=np.int32)

    # small var: red. large var: blue
    varpoints = points.var(axis=2)
    varmin = varpoints.min()
    varmax = varpoints.max()

    for ni in range(0,nc,step):
        densef = DensePropagation(points[ni], anchors, anchorsDense)
        densef = densef.var(axis=1)
        # print(name + 'variance:', varpoints.mean())
        densec = 255 * clip_to_crange(densef, crange, densef.min(), densef.max()) # na x 3 rgb
        if samples is not None:
            anchor_xyz = radius * anchorsDense + samples[ni][None]  # na x 3 coords
        else:
            anchor_xyz = radius * anchorsDense
        spheres.append(np.concatenate((anchor_xyz,densec),axis=1))

    for ni in range(nc):
        #nnc = 255 * crange[colorstep[ni]]
        nnc = np.array([255,255,255], dtype=np.int32)
        nnpts.append(np.concatenate((nbrs[ni], np.tile(nnc[None],[nn,1])), axis=1)) # nn x 6

    spheres = np.array(spheres).reshape(-1,6)
    nnpts = np.array(nnpts).reshape(-1,6)

    if output_folder is None:
        output_folder = 'data/visualization/sphere_features'

    create_dir(output_folder)
    
    wpp(spheres, os.path.join(output_folder, name + '_spheres.ply'), use_color=True)
    wpp(nnpts, os.path.join(output_folder, name + '_nnpts.ply'), use_color=True)    s

def visualize_conic_receptive_field(xyz, anchor_idx, layer_id, cstep=5, astep=1):
    '''
    xyz: nbxnpx3
    anchor_idx: nbxncxnaxann or nbxncxnaxksxann
    '''

    if anchor_idx.ndim == 5:
        anchor_idx = anchor_idx[:,:,:,0,:]

    bdim = xyz.size(0)
    nsupport = xyz.size(1)
    adim = anchor_idx.size(2)
    ann = anchor_idx.size(3)
    xyz = xyz.cpu().numpy()
    anchor_idx = anchor_idx.cpu().numpy()

    create_dir("data/visualization/conefield")
    output_dir = "data/visualization/conefield"

    color_step = np.linspace(0, len(crange)-1, bdim, dtype=np.int32)
    pc_color = np.array([255,255,255], dtype=np.float32)
    pc_color = np.tile(np.expand_dims(pc_color,0),[xyz.shape[1],1])

    for idx in range(bdim):
        xyzi = xyz[idx]    
        anchori = anchor_idx[idx]
        cxyzi = np.concatenate((xyzi,pc_color), axis=1)
        for ci in range(0,anchori.shape[0],cstep):
            colored_pc = []
            for ai in range(0, adim, astep):
                pidx = anchori[ci,ai]
                pidx = pidx[pidx < nsupport]
                if len(pidx) > 0:                   
                    cone = xyzi[pidx]
                    ccone = 255 * np.tile(np.expand_dims(crange[color_step[ai]],0),[len(pidx),1])
                    colored_pc.append(np.concatenate((cone, ccone), axis=1))
            colored_pc.append(cxyzi)
            colored_pc = np.vstack(colored_pc)
            wpp(colored_pc, os.path.join(output_dir, "%s_pc%d_center%d_cone.ply"%(layer_id, idx, ci)), use_color=True)
    
def visualize_feature_tsne(x_src, x_tgt, savepath):
    '''
    src, tgt: torch tensor BxAxc_out
    '''
    bdim = x_src.size(0)
    x_src = x_src.view(bdim, -1).detach().cpu().numpy()
    x_tgt = x_tgt.view(bdim, -1).detach().cpu().numpy()

    features = np.concatenate((x_src,x_tgt),axis=0) 
    embeddings = TSNE(n_components=2).fit_transform(features)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    color_step = np.linspace(0, len(crange)-1, bdim, dtype=np.int32)

    for bi in range(bdim):
        xs = np.array([embeddings[bi,0], embeddings[bi+bdim,0]])
        ys = np.array([embeddings[bi,1], embeddings[bi+bdim,1]])
        ax.scatter(xs,ys,c=crange[np.array([color_step[bi]]).T])

    plt.savefig(savepath)
    plt.close()
