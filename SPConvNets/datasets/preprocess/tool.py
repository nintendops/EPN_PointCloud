
import os
import numpy as np
import glob
import imageio
import scipy.io as sio
from plyfile import PlyElement, PlyData
from parse import parse
from sklearn.neighbors import NearestNeighbors as nnbrs
from scipy.spatial import KDTree
from collections import Counter
import open3d as o3d


# Configuration

class Config:
    def __init__(self):
        self.verbose = False

        # setting for rgbd fusion
        self.depth_scale = 1000
        self.depth_trunc = 6
        self.tsdf_cubic_size = 3.0
        self.frames_per_frag = 50
        self.width = 640
        self.height = 480

        # setting for keypoint selection 
        self.subsample_ratio = 20
        self.subsample_maxpoints = 100000
        self.overlap_ratio = 0.3
        self.dist_margin = 0.075

        # setting for fpfh filtering
        self.voxel_size = 0.05
        self.fpfh_radius = 0.15
        self.fpfh_thresh = 0.03

        # setting for ballsearch
        self.search_radius = 0.4
        self.patch_maxpoints = 2048

        # setting for serialization
        self.blocksize = 512
        self.max_patches_per_scene = 1000
        self.num_threads = 8


# ---------------- Helper Functions -------------------


def to_hom_np(pc, batch=False, rotate_only=False):
    pc_shape = pc.shape
    padding = 0 if rotate_only else 1
    if batch:
        ones = np.zeros((pc.shape[0], 1, pc.shape[2])) + padding
        return np.concatenate((pc, ones), axis=1)
    else:
        ones = np.zeros((1, pc.shape[1])) + padding
        return np.concatenate((pc, ones), axis=0)

def from_hom_np(pc, batch=False):
    if batch:
        return pc[..., :-1, :]
    else:
        return pc[:-1, :]

def save_kpts_list(path, arr):
    with open(path, 'w') as f:
        for idx in arr:
            f.write('%d\n'%idx)

def transform_np(pc, T, batch=False):
    pc = pc.T
    
    if pc.shape[-2] == 3:
        pc = to_hom_np(pc)
    if batch:
        if T.ndim == 2:
            pcT =  np.einsum('pk,bkl->bpl', T, pc)
        elif T.ndim == 3:
            pcT = np.einsum('bpk,bkl->bpl', T, pc)
    else:
        pcT = np.matmul(T, pc)
    return from_hom_np(pcT).T

def ball_search_np(pc, kpt, knn, search_radius, subsample_ratio=1):
    if subsample_ratio > 1:
        pc_sub = subsample_pc(pc, pc.shape[0]//subsample_ratio)
    else:
        pc_sub = pc
    # knn-ball search
    nn = min(10000, pc_sub.shape[0])
    nbrs = nnbrs(n_neighbors=nn, algorithm='ball_tree').fit(pc_sub)
    dists, indices = nbrs.kneighbors(pc[kpt])
    true_indices = []
    maxcount = 0
    for i in range(len(dists)):
        if dists[i].max() > search_radius:
            lidx = np.where(dists[i] > search_radius)[0][0]
            # print(f'{lidx} vs {knn}')
            if lidx >= knn:
                true_indices.append(np.random.choice(indices[i][:lidx],knn))
            else:
                choice = np.random.choice(range(lidx - 1), knn - lidx)
                true_indices.append(np.append(indices[i][:lidx], indices[i][choice]))
        else:
            true_indices.append(np.random.choice(indices[i],knn))
            maxcount += 1

    print("inclusion ratio: ", 1 - float(maxcount)/float(len(dists)))
    return np.array(true_indices, dtype=np.int32), pc_sub

def parse_registration(path):
    T = {}
    with open(path, 'r') as f:
        line = f.readline()
        cnt = 0
        while line:
            # first line: scene ids
            id1, id2, _ = parse("{:d}\t{:d}\t{:d}\t",line)
            matchid = str(id1) + 'n' + str(id2)

            mat = np.zeros((4,4))
            # 2-5 line: Transformation
            rowcnt = 0
            for i in range(4):
                line = f.readline()
                row = parse("{:f}\t{:f}\t{:f}\t{:f}",line)
                row = [row[0],row[1],row[2],row[3]]
                mat[rowcnt] = row
                rowcnt +=1

            T[matchid] = mat
            cnt+=1
            line = f.readline()
    return T

def parse_scene_id(file):
    if '\\' in file:
        file = file.split('\\')[-1]
    assert file.endswith('.txt') or file.endswith('.ply')
    if file.endswith('.txt'):
        result = parse("{}_{:d}Keypoints.txt", file)
    elif file.endswith('.ply'):
        result = parse("{}_{:d}.ply", file)
    elif file.endswith('.npz'):
        result = parse("{}_{:d}.npz", file)
    if result is not None:
        return result[1]
    else:
        return None

def subsample_pc(pc, k):
    choice = np.random.choice(np.arange(pc.shape[0]),k)
    return pc[choice]

def load_ply(file_name, with_faces=False, with_color=False, with_normal=False):
    ply_data = PlyData.read(file_name)
    vertices = ply_data['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if with_normal:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    pc = ret_val

    return (pc, normals) if with_normal else pc

def save_ply(filepath, color_pc, c=None, use_color=False, use_normal=False, verbose=False):
    # if verbose:
    #     print("Writing color pointcloud to: ", filepath)
    
    f = open(filepath, 'w')
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write('element vertex '+str(int(color_pc.shape[0])) + '\n')
    f.write('property float x\nproperty float y\nproperty float z\n')
    if use_normal:
        f.write('property float nx\nproperty float ny\nproperty float nz\n')
    f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
    f.write("end_header\n")

    if c == 'r':
        color = [255, 0, 0]
    elif c == 'g':
        color = [0, 255, 0]
    elif c == 'b':
        color = [0,0,255]
    elif isinstance(c, list):
        color = c
    else:
        color = [255,255,255]
    
    # default_normal = 0.0
    color_range = range(6,9) if use_normal else range(3,6)

    for i in range(color_pc.shape[0]):
        #f.write("v")
        row = list(color_pc[i,:])
        for j in range(3):
            f.write("%.4f " % float(row[j]))

        # ---------------- normal ---------------
        if use_normal:
            for t in range(3,6):
                f.write("%.4f " % float(row[t]))
        # else:
        #     for t in range(3):
        #         f.write("%.4f " % float(default_normal))


        # ---------------- color ----------------    
        if use_color:
            if color_pc.shape[1] == 4:
                # only intensity provided
                for t in range(3):
                    f.write(str(int(color[t] * row[3])) + ' ')
            else:
                for t in color_range:
                    f.write(str(int(row[t])) + ' ')
        else:
            for t in range(3):
                f.write(str(int(color[t])) + ' ')

        f.write("\n")
    f.write("\n")
    f.close()


def generate_point_cloud(cam_path, img_path, ply_path=None):
    '''
    cam_path: /xxx/xxx/camera-intrinsics.txt
    img_path: /xxx/xxx/frame-000000
    '''
    cam = np.loadtxt(cam_path)
    pos = np.loadtxt(img_path+'.pose.txt')
    # img = imageio.imread(img_path+'.color.png')
    dpt = imageio.imread(img_path+'.depth.png').astype('float32') # in milli-meters
    val = dpt > 0.

    H, W = dpt.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uvd = np.stack((u+0.5, v+0.5, dpt), axis=2)

    uvd = uvd[val] # [N, 4] in camera space
    uvd[:, :2] *= uvd[:, 2:]
    uvd = uvd @ np.linalg.inv(cam).T
    uvd /= 1000.
    uvdw = np.concatenate((uvd, np.ones((uvd.shape[0], 1))), axis=1)

    xyzw = uvdw @ pos.T
    xyz = xyzw[:, :3]

    if ply_path is not None:
        save_ply(ply_path, xyz)
    return xyz
    

def read_intrinsic(filepath, width, height):
    m = np.loadtxt(filepath, dtype=np.float32)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, m[0, 0], m[1, 1], m[0, 2],
                                                  m[1, 2])
    return intrinsic

def test_scenes_overlap(pc1, pc2, overlap_ratio=0.3, margin=1e-2, verbose=False):
    nbrs = nnbrs(n_neighbors=1, algorithm='ball_tree').fit(pc2)
    dists, indices = nbrs.kneighbors(pc1)
    pc1idx = np.argwhere(dists<=margin)[:,0]
    pc2idx = indices[pc1idx].reshape(-1)

    if verbose:
        print("Matched points: ", pc1idx.shape[0])
    n_overlap = pc1idx.shape[0]
    n_pts = max(pc1.shape[0],pc2.shape[0])

    if verbose:
        print("Overlap ratio is %f"%(n_overlap/n_pts))

    return n_overlap >= overlap_ratio * n_pts

