'''
This script is DEPRECATED used to generate training data from 3DMatch
The dataset can be found at: http://3dmatch.cs.princeton.edu/#geometric-registration-benchmark
Which contains the following scenes with (RGB)D image and the associated camera poses:

# 7-scenes-chess
# 7-scenes-fire
# 7-scenes-heads
# 7-scenes-office
# 7-scenes-pumpkin
# 7-scenes-stairs
# bundlefusion-apt0
# bundlefusion-apt1
# bundlefusion-apt2
# bundlefusion-copyroom
# bundlefusion-office0
# bundlefusion-office1
# bundlefusion-office2
# bundlefusion-office3
# rgbd-scenes-v2-scene_01
# rgbd-scenes-v2-scene_02
# rgbd-scenes-v2-scene_03
# rgbd-scenes-v2-scene_04
# rgbd-scenes-v2-scene_05
# rgbd-scenes-v2-scene_06
# rgbd-scenes-v2-scene_07
# rgbd-scenes-v2-scene_08
# rgbd-scenes-v2-scene_09
# rgbd-scenes-v2-scene_10
# rgbd-scenes-v2-scene_11
# rgbd-scenes-v2-scene_12
# rgbd-scenes-v2-scene_13
# rgbd-scenes-v2-scene_14
# sun3d-harvard_c5-hv_c5_1
# sun3d-harvard_c6-hv_c6_1
# sun3d-harvard_c8-hv_c8_3
# sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika
# sun3d-hotel_nips2012-nips_4
# sun3d-hotel_sf-scan1
# sun3d-mit_32_d507-d507_2
# sun3d-mit_46_ted_lab1-ted_lab_2

'''
import os
import numpy as np
import shutil
import glob
import imageio
import scipy.io as sio
from plyfile import PlyElement, PlyData
from parse import parse
from sklearn.neighbors import NearestNeighbors as nnbrs
from multiprocessing import Pool
from scipy.spatial import KDTree
from collections import Counter
from pprint import pprint
import shutil


# ---------------- Helper Functions -------------------

# def npy_file_sort(s):
#     if '/' in s:
#         s = s.split('/')[-1]
#     pid, _ = parse("pair{:d}_{}.npy", s)
#     return pid

# def extract_pair_id(s):
#     if '/' in s:
#         s = s.split('/')[-1]
#     return s.split('_')[1]
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
    
def find_scenes_overlap(pc1, pc2, T, k=5000, margin=1e-2, subsample=False):
    if subsample:
        pc1 = subsample_pc(pc1, pc1.shape[0]//10)
        pc2 = subsample_pc(pc2, pc2.shape[0]//10)

    if T is not None:
        pc2_t = transform_np(pc2, T)
    else:
        pc2_t = pc2

    nbrs = nnbrs(n_neighbors=1, algorithm='ball_tree').fit(pc2_t)
    dists, indices = nbrs.kneighbors(pc1)
    pc1idx = np.argwhere(dists<=margin)[:,0]
    pc2idx = indices[pc1idx].reshape(-1)

    print("Matched points: ", pc1idx.shape[0])

    if pc1idx.shape[0] > 10 * k:
        choice = np.random.choice(pc1idx.shape[0], k, replace=False)
        pc1idx = pc1idx[choice]
        pc2idx = pc2idx[choice]
        return pc1idx, pc2idx, pc2_t
    else:
        # save_ply("example_src.ply",pc1)
        # save_ply("example_tgt.ply", pc2_t)
        raise ValueError("Not enough overlapping points between this pair of scenes.")

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

# ---------------- Main Functions -------------------

# ------------------------------------- DEPRECATED -----------------------------------------------------
# def GenerateDataByRGBD_Training(search_radius, keypoint_sample, search_knn, output_root, data_paths, \
#                         margin=0.01, scale=1.0, localframe=True):
#     '''
#         search_radius: radius in ball search
#         keypoint_sample: how many keypoints to extract patches from
#         search_knn: how many neighbors to include in ball search
#     '''

#     file_id = lambda s: int(os.path.basename(s)[:-4][6:])
#     seq_id = lambda s: os.path.basename(os.path.dirname(s))
    
#     # scene_root = os.path.join(output_root, scene_name)
#     scene_root = output_root
#     os.makedirs(scene_root, exist_ok=True)

#     if isinstance(data_paths, list):
#         for data_path in data_paths:
#             scene_name = os.path.basename(data_path)
#             print("Working on Scene: ", scene_name)

#             # sort scene point clouds
#             all_scanfiles = []
#             for seq in os.listdir(data_path):
#                 if seq.startswith('seq') and os.path.isdir(os.path.join(data_path, seq)):
#                     scanfiles = glob.glob(os.path.join(data_path, seq, "*.ply"))
#                     scanfiles.sort(key=file_id)
#                     # TODO: check whether different seqs have different world coords
#                     all_scanfiles += scanfiles
            
#             # all_scanfiles = all_scanfiles[::5]        
#             for idx, sf in enumerate(all_scanfiles):
#                 src = load_ply(sf)
#                 # 4x4 matrix
#                 src_pose = np.loadtxt(sf[:-4] + '.pose.txt')
#                 sample = min(2, len(all_scanfiles) - idx - 1)
#                 choice = np.random.choice(range(idx + 1, min(idx + 5, len(all_scanfiles))), sample)
#                 for jdx in choice:
#                     tgt = load_ply(all_scanfiles[jdx])
#                     tgt_pose = np.loadtxt(all_scanfiles[jdx][:-4] + '.pose.txt')
#                     new_key = "%s_%s_%dn%s_%d"%(scene_name,seq_id(sf),file_id(sf),\
#                                           seq_id(all_scanfiles[jdx]), file_id(all_scanfiles[jdx]))
#                     print("Working on key %s..."%new_key)      
#                     # KxNx3 patches
#                     nn_anc, nn_pos = nn_patchsearch(src, tgt, keypoint_sample, search_radius, margin)

#                     # save the data
#                     if nn_anc is not None:
#                         if localframe:
#                             nn_anc = nn_anc @ src_pose[:3,:3]
#                             nn_pos = nn_pos @ tgt_pose[:3,:3]
#                         # scaling
#                         nn_anc = nn_anc * scale
#                         nn_pos = nn_pos * scale
#                         save_data = {'src':nn_anc, 'tgt':nn_pos, 'T':np.identity(4)}
#                         sio.savemat(os.path.join(scene_root, new_key + '.mat'), save_data)
#     elif isinstance(data_paths,str):
#         meta = sio.loadmat(data_paths)
#         num_data = len(meta['anc'])

#         for idx, paths in enumerate(zip(meta['anc'],meta['pos'],meta['anc_kpt'],meta['pos_kpt'])):
#             sf,tf,sf_kpt,tf_kpt = paths
#             sf = sf.strip()
#             tf = tf.strip()
#             sf_kpt = sf_kpt.strip()
#             tf_kpt = tf_kpt.strip()

#             src = load_ply(sf)
#             src_pose = np.loadtxt(sf[:-4] + '.pose.txt')
#             tgt = load_ply(tf)
#             tgt_pose = np.loadtxt(tf[:-4] + '.pose.txt')
#             kpts = [np.loadtxt(sf_kpt,dtype=np.int32),np.loadtxt(tf_kpt,dtype=np.int32)]
#             nn_anc, nn_pos = nn_patchsearch(src, tgt, keypoint_sample, search_radius, margin, kpts=kpts)

#             # save the data
#             if nn_anc is not None:
#                 if localframe:
#                     nn_anc = nn_anc @ src_pose[:3,:3]
#                     nn_pos = nn_pos @ tgt_pose[:3,:3]
#                 # scaling
#                 nn_anc = nn_anc * scale
#                 nn_pos = nn_pos * scale
#                 save_data = {'src':nn_anc, 'tgt':nn_pos, 'T':np.identity(4)}

#                 scene_name = os.path.basename(os.path.dirname(os.path.dirname(sf)))
#                 new_key = "%s_%s_%dn%s_%d"%(scene_name,seq_id(sf),file_id(sf),\
#                                             seq_id(tf), file_id(tf))
#                 print(f"Saving...: {new_key}")
#                 sio.savemat(os.path.join(scene_root, new_key + '.mat'), save_data)

#         return meta


if __name__ == '__main__':
    root_path = r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/group1'
    fuse_root = os.path.join(r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/fused_scenes')
    input_root = r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/input'
    more_root = r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/more'

    cfg = Config()

    # fusion
    # run_RGBDFusion(fuse_root, root_path, cfg)

    # pair selection and keypoints
    # run_KeypointSelection(fuse_root, input_root, cfg)

    # patch extraction
    # run_PatchExtraction(fuse_root, input_root, cfg)

    # run_profiling(os.path.join(input_root, 'train_r0.20'), more_root, False)

    print("Done!!!")
