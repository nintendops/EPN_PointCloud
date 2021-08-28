import os
import argparse
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
from tool import *



SCENE_TO_FILTER = [    
              # 'kitchen',
              # '7-scenes-redkitchen',
              # 'sun3d-home_at-home_at_scan1_2013_jan_1',
              # 'sun3d-home_md-home_md_scan9_2012_sep_30', 
              # 'sun3d-hotel_uc-scan3', 
              # 'sun3d-hotel_umd-maryland_hotel1', 
              # 'sun3d-hotel_umd-maryland_hotel3', 
              # 'sun3d-mit_76_studyroom-76-1studyroom2',
              # 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
            ]

def nnbrsearch(feat1, feat2, knn=1, dist=False):
    nbrs = nnbrs(n_neighbors=knn)
    nbrs.fit(feat2)
    rst = nbrs.kneighbors(feat1,return_distance=dist)
    if dist:
        return (rst[0].squeeze(),rst[1].squeeze())
    else:
        return rst.squeeze()


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
def cross_filtering_via_fpfh(points_i, points_j, fpfh_i, fpfh_j, cfg, nonplanar_param=-1):
    # rule out those points with no feature
    indices_i = np.arange(len(points_i))[np.any(fpfh_i != 0, axis=1)]
    indices_j = np.arange(len(points_j))[np.any(fpfh_j != 0, axis=1)]
    fpfh_i = fpfh_i[indices_i, :]
    fpfh_j = fpfh_j[indices_j, :]
    points_i = points_i[indices_i, :]
    points_j = points_j[indices_j, :]

    # first we found euclidean neirhbors
    dists, indices = nnbrsearch(points_i, points_j, knn=1,dist=True)
    dist_filter = np.argwhere(dists<=cfg.fpfh_thresh).squeeze()
    dists = dists[dist_filter]
    indices = indices[dist_filter]

    # filter out points too far apart
    fi = fpfh_i[dist_filter]
    fj = fpfh_j[indices]
    points_i = points_i[dist_filter]
    points_j = points_j[indices]

    if nonplanar_param > 0:
        # filter out planar patches, aka std() >= nonplanar_param
        match_flags = []
        for idx, fs in enumerate(zip(fi,fj)):
            fii, fjj = fs
            if all([fii.std() < nonplanar_param, fjj.std() < nonplanar_param]):
                match_flags += [idx]
        n_kpt = len(match_flags)
        
        if n_kpt < 128:
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

    
    if test_scenes_overlap(src_sub, tgt_sub, cfg.overlap_ratio, cfg.dist_margin, cfg.verbose):
        # extract keypoints through fpfh matching
        tgt_down, tgt_fpfh = downsample_and_compute_fpfh(tgt_pcd, cfg)
        tgt_fpfh = np.asarray(tgt_fpfh.data).T
        kpti, kptj = cross_filtering_via_fpfh(src_down, from_o3d(tgt_down), src_fpfh, tgt_fpfh, cfg)

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
            np.save(save_path, kpts.astype(np.int32))
            

def run_KeypointSelection(root_path, cfg):

    # just FYI
    from_o3d = lambda x: np.asarray(x.points)
    get_folders = lambda s: list(filter(lambda f: os.path.isdir(f), glob.glob(s + '/*')))
    fragment_path = os.path.join(root_path, 'fused_fragments')
    for sdir in get_folders(fragment_path):
        scene_name = os.path.basename(sdir)
        if scene_name in SCENE_TO_FILTER:
            print("Skipping scene", scene_name)
            continue

        ckpt_path = os.path.join(root_path, 'kpts', scene_name)

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
            
            for j in range(i+1, min(i+20,n_frag),4):
                save_path = os.path.join(ckpt_path,'cloud_bin_%d-cloud_bin_%d.npy'%(i,j))
                if os.path.exists(save_path):
                    continue
                mp_args.append([src, src_sub,from_o3d(src_down),np.asarray(src_fpfh.data).T,
                                j, sdir, save_path, cfg])

            # pool = Pool(cfg.num_threads)
            # pool.starmap(generate_kp, mp_args)
            for args in mp_args:
                generate_kp(*args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', type=str, required=True)
    args = parser.parse_args()

    # fuse_root = os.path.join(r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/fused_scenes')
    # input_root = r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/input'
    
    cfg = Config()
    run_KeypointSelection(args.root_path, cfg)
    print("Done!!!")
