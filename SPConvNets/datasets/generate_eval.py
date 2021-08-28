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


class Config():
    def __init__(self, datapath, radius):
        model = namedtuple('model', ['input_num','search_radius'])
        self.model = model(2048, radius)
        self.dataset_path = datapath
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--search-radius', type=float, default=0.4)
    args = parser.parse_args()
    opt = Config(args.data_path, args.search_radius)

    for scene in ALLSCENES:
        print(f"Working on scene {scene}!")
        dataset = SceneTestLoader(opt)
        dataset.prepare(scene)
        dataset.precompute_patches(scale=1.0, input_num=2048, num_worker=1)

    print("Done!")
