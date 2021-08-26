from tool import *

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
from multiprocessing import Pool
import open3d as o3d



def downsample_and_compute_fpfh(pcd, cfg, T=None):

    pcd_down = pcd.voxel_down_sample(cfg.voxel_size)
    
    if T is not None:
        pcd_down = pcd_down.transform(T)

    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamRadius(cfg.fpfh_radius))

    return pcd_down, pcd_fpfh


'''
    Credit to https://github.com/craigleili/3DLocalMultiViewDesc/blob/master/data/preprocess/compute_kpt_pairs.py
'''
def cross_filtering_via_fpfh(points_i, points_j, fpfh_i, fpfh_j, cfg):

    # fpfh_i = np.asarray(src_fpfh.data).T
    # fpfh_j = np.asarray(tgt_fpfh.data).T

    # rule out those points with no feature

    indices_i = np.arange(len(points_i))[np.any(fpfh_i != 0, axis=1)]
    indices_j = np.arange(len(points_j))[np.any(fpfh_j != 0, axis=1)]

    fpfh_i = fpfh_i[indices_i, :]
    fpfh_j = fpfh_j[indices_j, :]
    points_i = points_i[indices_i, :]
    points_j = points_j[indices_j, :]

    # kd search for pair of points whose fpfh and distance are both closest, aka matching points with distinctive features
    kdtree_j = o3d.geometry.KDTreeFlann(fpfh_j.T)
    nnindices = [
        kdtree_j.search_knn_vector_xd(fpfh_i[k, :], 1)[1][0] for k in range(len(fpfh_i))
    ]
    # points_j are the nn in fpfh
    points_j = points_j[nnindices, :]
    distances = np.sqrt(np.sum(np.square(points_i - points_j), axis=1))


    # now we filter them with a distance threshold
    match_flags = distances <= cfg.fpfh_thresh
    n_kpt = np.sum(match_flags)
    if n_kpt < 16:
        print("Not enough feature points! Only found %d..."%n_kpt)
        return None, None
    else:
        print("Found %d feature points!"%n_kpt)

    points_i = points_i[match_flags, :]
    points_j = points_j[match_flags, :]

    # return: distinctive matching points in the downsampled point clouds
    return points_i, points_j


def test_scenes_overlap(pc1, pc2, overlap_ratio=0.3, margin=1e-2, verbose=True):
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

def generate_kp(src, src_sub, src_down, src_fpfh, j, sdir, save_path, cfg):
    from_o3d = lambda x: np.asarray(x.points)
    tgt_pcd = o3d.io.read_point_cloud(os.path.join(sdir, 'cloud_bin_%d.ply'%j))
    tgt_pose = np.loadtxt(os.path.join(sdir, 'cloud_bin_%d_pose.txt'%j), dtype=np.float32)
    tgt_pcd = tgt_pcd.transform(tgt_pose)
    tgt = from_o3d(tgt_pcd)
    # tgt = load_ply(os.path.join(sdir, 'cloud_bin_%d.ply'%j))
    tgt_sub = subsample_pc(tgt, min(cfg.subsample_maxpoints, tgt.shape[0]))

    # print(f"Working on {os.path.basename(save_path)}...")
    
    if test_scenes_overlap(src_sub, tgt_sub, cfg.overlap_ratio, cfg.dist_margin, cfg.verbose):
        # extract keypoints through fpfh matching
        # print("DSDS")
        tgt_down, tgt_fpfh = downsample_and_compute_fpfh(tgt_pcd, cfg)
        # print("YYYY")
        tgt_fpfh = np.asarray(tgt_fpfh.data).T
        kpti, kptj = cross_filtering_via_fpfh(src_down, from_o3d(tgt_down), src_fpfh, tgt_fpfh, cfg)
        # if enough keypoints
        # print("KKKK")
        if kpti is not None:
            kpts = []
            for pcd, kpt in zip([src, tgt], [kpti, kptj]):
                nbrs = nnbrs(n_neighbors=1, algorithm='ball_tree').fit(pcd)
                dists, indices = nbrs.kneighbors(kpt)
                # sanity check?
                if np.sum(dists.squeeze() > 0.03) > 0:
                    print("WARNING: SOME SEARCHED POINTS MAY BE TOO FAR FROM SELECTED KEYPOINTS!!!")
                kpts.append(indices)
            kpts = np.concatenate(kpts,axis=1)
            print("Keypoint saved to", save_path)
            np.savetxt(save_path, kpts, fmt='%i')
            

def run_KeypointSelection(root_path, output_path, cfg):

    # just FYI
    from_o3d = lambda x: np.asarray(x.points)
    get_folders = lambda s: list(filter(lambda f: os.path.isdir(f), glob.glob(s + '/*')))
        
    for sdir in get_folders(root_path):
        scene_name = os.path.basename(sdir)
        ckpt_path = os.path.join(output_path, 'keypoints', scene_name)

        # Hackish
        if os.path.exists(ckpt_path) and len(glob.glob(os.path.join(ckpt_path,"*.txt"))) > 1:
            print("Skipping scene %s!"%scene_name)
            continue

        os.makedirs(ckpt_path, exist_ok=True)
        n_frag = len(glob.glob(os.path.join(sdir, 'cloud_bin_*.ply')))
        assert n_frag == len(glob.glob(os.path.join(sdir, 'cloud_bin_*_pose.txt')))

        for i in range(n_frag):
            # read pc, pose and subsample the pc
            src_pcd = o3d.io.read_point_cloud(os.path.join(sdir, 'cloud_bin_%d.ply'%i))
            src_pose = np.loadtxt(os.path.join(sdir, 'cloud_bin_%d_pose.txt'%i), dtype=np.float32)
            src_pcd = src_pcd.transform(src_pose)
            src = from_o3d(src_pcd)    
            src_sub = subsample_pc(src, min(cfg.subsample_maxpoints, src.shape[0]))
            src_down, src_fpfh = downsample_and_compute_fpfh(src_pcd, cfg)

            mp_args = []

            print(f"At scene {scene_name}, fragment {i}")
            
            for j in range(i+1, min(i+20,n_frag)):
                save_path = os.path.join(ckpt_path,'cloud_bin_%d-%d.keypoints.txt'%(i,j))
                if os.path.exists(save_path):
                    continue
                mp_args.append([src, src_sub,from_o3d(src_down),np.asarray(src_fpfh.data).T,
                                j, sdir, save_path, cfg])

            # pool = Pool(cfg.num_threads)
            # pool.starmap(generate_kp, mp_args)
            for args in mp_args:
                generate_kp(*args)

if __name__ == '__main__':
    fuse_root = os.path.join(r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/fused_scenes')
    input_root = r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/input'
    
    cfg = Config()
    run_KeypointSelection(fuse_root, input_root, cfg)
    print("Done!!!")
