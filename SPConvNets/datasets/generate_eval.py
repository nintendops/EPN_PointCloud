# script to generate test patches
import os
from os.path import join as join
import glob
import numpy as np
from match_3dmatch import *
from collections import namedtuple
import vgtk.pc as pctk


ALLSCENES = [ # 'kitchen',
              '7-scenes-redkitchen',
              'sun3d-home_at-home_at_scan1_2013_jan_1',
              'sun3d-home_md-home_md_scan9_2012_sep_30',
              'sun3d-hotel_uc-scan3',
              'sun3d-hotel_umd-maryland_hotel1',
              'sun3d-hotel_umd-maryland_hotel3',
              'sun3d-mit_76_studyroom-76-1studyroom2',
              'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
             ]


RADIUS = 0.52

class Config():
    def __init__(self):
        model = namedtuple('model', ['input_num','search_radius'])
        self.model = model(2048, RADIUS)
        # self.dataset_path = '/home/ICT2000/chenh/Haiwei/LocalMVD/data/3DMatchBenchmark'
        self.dataset_path = '/home/ICT2000/chenh/Datasets/MScenes/evaluation/3DMatch'
        self.batch_size = 8

def radius_ball_search_np_radii(pc, kpt_indices, radii, search_radius, input_num=None, msg=None):
    if msg is not None:
      print(msg)

    # radius-ball search
    keypoints = pc[kpt_indices]
    if pc.shape[0] > 50000:
        _, pc = pctk.uniform_resample_np(pc, 50000)

    search = KDTree(pc)

    all_pc = []
    for idx, kpt in enumerate(kpt_indices):
        r = search_radius * radii[kpt] / 0.026
        indices = search.query_ball_point(keypoints[idx], r)
        if len(indices) <= 1:
            i = 1024 if input_num is None else input_num
            all_pc.append(np.zeros([i,3],dtype=np.float32))
        else:
            if input_num is not None:
                _, patch = pctk.uniform_resample_np(pc[indices], input_num)
            else:
                patch = pc[indices]
            all_pc.append(patch)

    return all_pc

vis_dir = "/home/ICT2000/chenh/Haiwei/ConicMatch/vis/3dmatch_samples/eval"

# temp generator function for dataset with precomputed point density (radius)
def precompute_patches_radii(data_path, search_radius, num_worker=8):
    # outpkptsfilesut: # fragments x 5000 x nn x 3 npy files
    ball_search = radius_ball_search_o3d_radii #radius_ball_search_np_radii
    kptsfiles = glob.glob(join(data_path, 'seq-01', "*.keypts.npy"))
    save_dir = os.path.join(data_path, 'grouped_data_r%.2f'%search_radius)
    os.makedirs(save_dir, exist_ok=True)
    mp_args = []
    sid_list = []

    for kid, kptf in enumerate(kptsfiles):
        kpts = np.load(kptf).astype(np.int32)
        radii = np.load(kptf.split('.keypts.')[0] + '.radius.npy')
        ply_path = kptf.split('.keypts.')[0] + '.ply'

        # scene_pcd = pctk.load_ply(ply_path)
        scene_pcd = o3d.io.read_point_cloud(ply_path)
        mp_args.append([scene_pcd,kpts,radii,search_radius,2048,'At %s'%os.path.basename(ply_path)])
        sid_list.append(os.path.basename(ply_path)[:-4])
    if num_worker > 1:
        pool = Pool(num_worker)
        rsts = pool.starmap(ball_search, mp_args)
        for rst,sid,arg in zip(rsts,sid_list,mp_args):
            # n_keypoints x knn x3
            grouped_points = np.array(rst)
            save_path = os.path.join(save_dir, 'grouped_%s.npz'%sid)
            np.savez(save_path, grouped_points)
    else:
        # non parallel method
        for sid,arg in zip(sid_list,mp_args):
            rst = ball_search(*arg)
            grouped_points = np.array(rst)
            # # #######################################
            # for i in range(0,5000,250):
            #   pctk.save_ply(os.path.join(vis_dir,'normal_patch_%d.ply'%i), grouped_points[i])
            # # ######################################
            # import ipdb; ipdb.set_trace()
            save_path = os.path.join(save_dir, 'grouped_%s.npz'%sid)
            np.savez(save_path, grouped_points)

for scene in ALLSCENES:
    print(f"Working on scene {scene}!")
    opt = Config()

    dataset = SceneTestLoader(opt)
    dataset.prepare(scene)
    dataset.precompute_patches(scale=1.0, input_num=2048, num_worker=1)

    # USE THIS TO GENERATE RADII
    # precompute_patches_radii(join(opt.dataset_path,scene), RADIUS, 1)

print("Done!")
