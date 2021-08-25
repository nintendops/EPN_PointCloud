# precompute fpfh data

import numpy as np
import open3d as o3d
import vgtk.pc as pctk
import glob
import os

train_dir = '/home/ICT2000/chenh/Haiwei/Datasets/MScenes/train'

def precompute_with_fpfh(pcd):
    # radius-ball search
    fpfh = []

    from_o3d = lambda pcd: np.asarray(pcd.points)
    
    pcd_down = pcd.voxel_down_sample(voxel_size=0.015)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamRadius(0.15))

    pc_down = from_o3d(pcd_down)
    fpfh = np.asarray(pcd_fpfh.data).T
     
    return pc_down, fpfh


if __name__ == "__main__":
    data_path = os.path.join(train_dir, 'fused_scenes')
    parse_string = 'cloud_bin_{:d}.ply'

    out_path = os.path.join(train_dir,'fpfh')

    for scene in os.listdir(data_path):
        os.makedirs(os.path.join(out_path,scene),exist_ok=True)
        print("At scene %s"%scene)
        for pcf in glob.glob(os.path.join(data_path, scene,"*.ply")):
            print("Working on %s..."%os.path.basename(pcf))
            name = os.path.basename(pcf)
            pcd = o3d.io.read_point_cloud(pcf)
            pcd_down, fpfh = precompute_with_fpfh(pcd)
            pctk.save_ply(os.path.join(out_path,scene,name), pcd_down)
            np.save(os.path.join(out_path,scene,name[:-4] + '.fpfh.npy'), fpfh)