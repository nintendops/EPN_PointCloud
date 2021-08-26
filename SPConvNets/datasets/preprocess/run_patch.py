'''
This script is used to generate training data from 3DMatch
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
import open3d as o3d
from pprint import pprint
from multiprocessing import Pool

def get_tag(mat_path, more=None, modify=True):
    filt = ['scene-to-cleanse']
    mat = sio.loadmat(mat_path)
    tags = [tag[0].split('_')[0] for pc, tag in mat['src']]

    # if modify and tag in filt and np.random.rand() > 0.5:
    #     print(f"Moving data for {tag} at {os.path.basename(mat_path)}...")
    #     shutil.move(mat_path, os.path.join(more, os.path.basename(mat_path)))
    #     tag = 'removed_%s'%tag

    return tags

def run_profiling(dataroot, more=None, modify=False):
    print("Profiling dataset...")

    tracker_path = os.path.join(dataroot, 'tracker.txt')
    if not os.path.exists(tracker_path):
        return None, None

    keyset = set()
    scene_ctn = Counter()
    
    # ----------------- only if you have time to run this ----------------------
    mats = glob.glob(os.path.join(dataroot, 'all_patches', '*.mat'))
    mp_args = []
    for mat_path in mats:
        mp_args.append([mat_path, more, False])
    pool = Pool(8)
    rst = pool.starmap(get_tag, mp_args)
    for tags in rst:
        for tag in tags:
            scene_ctn[tag] += 1

    # with open(os.path.join(dataroot, 'tracker.txt'),'r') as f:
    #     keys = f.readlines()
    #     for k in keys:
    #         try:
    #             scene, id1, id2 = parse('{}_{:d}-{:d}', k)
    #             scene_ctn[scene] += 1
    #             keyset.add(k.strip())
    #         except TypeError as e:
    #             continue

    pprint(scene_ctn)    
    npoints = np.array(list(scene_ctn.values()))
    import ipdb; ipdb.set_trace()

    return scene_ctn, keyset

def new_ball_search_np(pc, kpt, cfg):

    '''
        pc: Nx3
        kpt: Kx3

        return: KxN0x3 patches
    '''

    # knn-ball search
    search = KDTree(pc)
    results = search.query_ball_point(kpt, cfg.search_radius)

    all_pc = []
    for indices in results:
        patch = pc[indices]
        if len(indices) > cfg.patch_maxpoints:
            patch = subsample_pc(patch, cfg.patch_maxpoints)
        all_pc.append(patch)

    return all_pc

def run_PatchExtraction(fuse_root, input_root, cfg):
    get_folders = lambda s: list(filter(lambda f: os.path.isdir(f), glob.glob(s + '/*')))

    # vis_folder = os.path.join(input_root, 'vis')
    # os.makedirs(vis_folder,exist_ok=True)

    train_dir = os.path.join(input_root, 'train_r%.2f'%cfg.search_radius)
    patch_dir = os.path.join(train_dir, 'all_patches')
    os.makedirs(patch_dir,exist_ok=True)
    
    pose_table = dict()
    counter = 0
    bufferA = []
    bufferB = []

    # load tracker
    scene_ctn, tracker = run_profiling(train_dir)
    if tracker is None:
        tracker = set()
        scene_ctn = Counter()        

    exist_mats = glob.glob(os.path.join(patch_dir, "*.mat"))
    if len(exist_mats) > 0:
        mid = [parse('patches{:d}.mat', os.path.basename(m))[0] for m in exist_mats]
        mid.sort()
        last_id = mid[-1] + 1
    else:
        last_id = 0

    for spath in get_folders(os.path.join(input_root,'keypoints')):
        scene_name = os.path.basename(spath)
        dpath = os.path.join(fuse_root, scene_name)        
        kptfiles = glob.glob(os.path.join(spath, "*.keypoints.txt"))
        for kptf in kptfiles:
            # fetch point clouds
            idA, idB = parse('cloud_bin_{:d}-{:d}.keypoints.txt', os.path.basename(kptf))
            unique_key = f'{scene_name}_{idA}-{idB}'
            print(f"Working on key {unique_key}...")
            keyA = f'{scene_name}_{idA}'
            keyB = f'{scene_name}_{idB}'

            if unique_key in tracker:
                print(f"Skipping {unique_key}...")
                continue

            scene_ctn[scene_name] += 1
            if scene_ctn[scene_name] > cfg.max_patches_per_scene:
                print(f"Skipping {unique_key}... (Due to # of patches at this scene exceeding {cfg.max_patches_per_scene}")
                continue

            pcA = load_ply(os.path.join(fuse_root, scene_name,'cloud_bin_%d.ply'%idA))
            pcB = load_ply(os.path.join(fuse_root, scene_name,'cloud_bin_%d.ply'%idB))
            poseA = np.loadtxt(os.path.join(fuse_root, scene_name,'cloud_bin_%d_pose.txt'%idA))
            poseB = np.loadtxt(os.path.join(fuse_root, scene_name,'cloud_bin_%d_pose.txt'%idB))
            kpts = np.loadtxt(kptf,dtype=np.int32)

            # record pose
            if keyA not in pose_table.keys():
                pose_table[keyA] = poseA
            if keyB not in pose_table.keys():
                pose_table[keyB] = poseB

            # gather keypoint locations and (optional) downsampling
            resultA = new_ball_search_np(subsample_pc(pcA,2 * cfg.subsample_maxpoints), pcA[kpts[:,0]],cfg)
            resultB = new_ball_search_np(subsample_pc(pcB,2 * cfg.subsample_maxpoints), pcB[kpts[:,1]],cfg)

            dl = len(resultA)
            buffer_len = len(bufferA)
            if dl < cfg.blocksize - buffer_len:
                bufferA += [(pc,keyA) for pc in resultA] 
                bufferB += [(pc,keyB) for pc in resultB] 
            else:
                diff = cfg.blocksize - buffer_len
                bufferA += [(pc,keyA) for pc in resultA[:diff]]
                bufferB += [(pc,keyB) for pc in resultB[:diff]]

                # save data 
                sio.savemat(os.path.join(train_dir, 'pose_table.mat'), pose_table)
                with open(os.path.join(train_dir, 'tracker.txt'), 'w') as f:
                    for key in tracker:
                        f.write(key + '\n')
                data = {'src':bufferA, 'tgt':bufferB}
                block_id = counter//cfg.blocksize + last_id
                sio.savemat(os.path.join(patch_dir,'patches%d.mat'%block_id), data)
                # reset
                bufferA = [(pc, keyA) for pc in resultA[diff:]][:(cfg.blocksize-1)]
                bufferB = [(pc, keyB) for pc in resultB[diff:]][:(cfg.blocksize-1)]
            
            counter += dl
            print('Counter at block %d'% (counter//cfg.blocksize))
            tracker.add(unique_key)

if __name__ == '__main__':
    fuse_root = os.path.join(r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/fused_scenes')
    input_root = r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/input'
    more_root = r'/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train/more'
    cfg = Config()
    # patch extraction
    run_PatchExtraction(fuse_root, input_root, cfg)
    run_profiling(os.path.join(input_root, 'train_r0.26'), more_root, False)
    print("Done!!!")
