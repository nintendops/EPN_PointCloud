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


def run_RGBDFusion(output_root, root_path, cfg):

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def FusionFromRGBD(frame_paths, intrinsics, 
                        cfg):
        '''
        frame_paths: [(color_path, depth_path, cam_path)]
        '''
        read = o3d.io.read_image
        volume = o3d.integration.ScalableTSDFVolume(
            voxel_length= cfg.tsdf_cubic_size / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8)

        # first pose is the canonical pose
        pose_base2world = np.loadtxt(frame_paths[0][2], dtype=np.float32)
        pose_world2base = np.linalg.inv(pose_base2world)

        for tup in frame_paths:
            cp, dp, camp = tup

            # current pose to canonical pose
            pose_cam2world = np.loadtxt(camp, dtype=np.float32)
            pose_cam2base = pose_world2base @ pose_cam2world
            # read rgbd image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(\
                read(cp), read(dp), cfg.depth_scale, cfg.depth_trunc, False)
            # fusion to canonical volume
            volume.integrate(rgbd_image, intrinsic, np.linalg.inv(pose_cam2base))

        pc = volume.extract_point_cloud()
        pc.estimate_normals()
        return pc, pose_base2world

    # scene under root folders
    get_folders = lambda s: list(filter(lambda f: os.path.isdir(f), glob.glob(s + '/*')))
    color_id = lambda x: parse('frame-{:d}.color.png', os.path.basename(x))[0]
    depth_id = lambda x: parse('frame-{:d}.depth.png', os.path.basename(x))[0]
    pose_id = lambda x: parse('frame-{:d}.pose.txt', os.path.basename(x))[0]

    scenes = get_folders(root_path)
    for s in scenes:
        frag_counter = 0
        scene_name = os.path.basename(s)
        output_dir = os.path.join(output_root,scene_name)
        os.makedirs(output_dir, exist_ok=True)

        # scene camera intrinsics
        intrinsic = read_intrinsic(os.path.join(s, 'camera-intrinsics.txt'), cfg.width, cfg.height)

        # seq under scene folders
        for seq in get_folders(s):
            if os.path.basename(seq).startswith('seq'):
                # here contain the color, depth, pose images
                cpaths = glob.glob(os.path.join(seq, "*.color.png"))
                dpaths = glob.glob(os.path.join(seq, "*.depth.png"))
                ppaths = glob.glob(os.path.join(seq, "*.pose.txt"))

                cpaths.sort(key=color_id)
                dpaths.sort(key=depth_id)
                ppaths.sort(key=pose_id)

                # sanity check
                assert len(cpaths) == len(dpaths) and len(cpaths) == len(ppaths)
                frame_paths = list(zip(cpaths, dpaths, ppaths))

                # loop over n frames for fusion
                nframes = cfg.frames_per_frag
                head = 0
                tail = min(nframes, len(cpaths))
                while tail <= len(cpaths):
                    print("Processing %d:%d/%d at scene %s..."%(head,tail,len(cpaths), scene_name))
                    pc, pose = FusionFromRGBD(frame_paths[head:tail], intrinsic, cfg)

                    np.savetxt(os.path.join(output_dir,'cloud_bin_%d_pose.txt'%frag_counter), pose)
                    if o3d.io.write_point_cloud(os.path.join(output_dir, 'cloud_bin_%d.ply'%frag_counter), pc):
                        print("Successfully written fused point cloud #%d for scene %s"%(frag_counter, scene_name))

                    # update counter
                    frag_counter += 1
                    head = tail
                    tail += nframes

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()
    cfg = Config()
    fuse_root = os.path.join(args.output_path, 'fused_fragments')
    # fusion
    run_RGBDFusion(fuse_root, args.data_path, cfg)
    print("Done!!!")
