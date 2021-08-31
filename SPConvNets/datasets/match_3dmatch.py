import os
import sys
import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
import scipy.io as sio
from parse import parse
import glob
from sklearn.neighbors import NearestNeighbors as nnbrs
from multiprocessing import Pool
import vgtk.so3conv.functional as L
import vgtk.pc as pctk
from vgtk.functional import RigidMatrix

# ------------------------ utilities for 3DMatch Data ---------------------------

# SCENE_SEARCH_RADIUS = 0.4

'''
registration file format:
scene_1     scene_2     #_of_scenes
[R t]
[0 1]
'''
def npy_file_sort(s):
    if '/' in s:
        s = s.split('/')[-1]
    pid, _ = parse("pair{:d}_{}.npy", s)
    return pid

def extract_pair_id(s):
    if '/' in s:
        s = s.split('/')[-1]
    return s.split('_')[1]


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
                row = parse("{:e}\t {:e}\t {:e}\t {:e}\t",line)
                row = [row[0],row[1],row[2],row[3]]
                mat[rowcnt] = row
                rowcnt +=1

            T[matchid] = mat
            cnt+=1
            line = f.readline()
    return T

def parse_scene_id(file):
    if '/' in file:
        file = file.split('/')[-1]
    if file.endswith('.txt'):
        result = parse("cloud_bin_{:d}Keypoints.txt", file)
    elif file.endswith('.ply'):
        result = parse("cloud_bin_{:d}.ply", file)
    elif file.endswith('.npz'):
        result = parse("grouped_cloud_bin_{:d}.npz", file)

    if result is not None:
        return result[0]
    else:
        return None


def find_scenes_overlap(pc1, pc2, T, k=5000, margin=5e-3):
    pc2_t = pctk.transform_np(pc2, T)
    nbrs = nnbrs(n_neighbors=1, algorithm='ball_tree').fit(pc2_t)
    dists, indices = nbrs.kneighbors(pc1)
    pc1idx = np.argwhere(dists<=margin)[:,0]
    pc2idx = indices[pc1idx].reshape(-1)

    print("Matched points: ", pc1idx.shape[0])

    if pc1idx.shape[0] > k:
        choice = np.random.choice(pc1idx.shape[0], k, replace=False)
        pc1idx = pc1idx[choice]
        pc2idx = pc2idx[choice]
        return pc1idx, pc2idx, pc2_t
    else:
        pctk.save_ply("example_src.ply",pc1)
        pctk.save_ply("example_tgt.ply", pc2_t)
        import ipdb; ipdb.set_trace()
        raise ValueError("Not enough overlapping points between this pair of scenes.")

# ---------------------------------------------------------------------------------


# This dataloader reads keypoint and fused fragment. Point cloud patches are computed online.
from collections import namedtuple
from scipy.spatial import KDTree
Kptmeta = namedtuple('Kptmeta','indices, id, pathA, pathB, poseA, poseB')

def radius_ball_search_o3d(pcd, kpt, search_radius, voxel_size=0.015, return_normals=False, input_num=None, name=None):
    # radius-ball search

    normals_at_kpt = None
    from_o3d = lambda pcd: np.asarray(pcd.points)
    keypoints = from_o3d(pcd)[kpt]
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size) #0.015
    # pcd_down = pcd
    pc = from_o3d(pcd_down)

    if return_normals:
        if len(pcd.normals) != len(pcd.points):
                raise RuntimeError('[!] The point cloud needs normals.')
        normals_at_kpt = np.asarray(pcd.normals)[kpt]

    search = KDTree(pc)
    results = search.query_ball_point(keypoints, search_radius)
    all_pc = []
    for indices in results:
        # print(len(indices))
        if len(indices) <= 1:
            i = 1024 if input_num is None else input_num
            all_pc.append(np.zeros([i,3],dtype=np.float32))
        else:
            if input_num is not None:
                resample_indices, patch = pctk.uniform_resample_np(pc[indices], input_num)
            else:
                patch = pc[indices]
            all_pc.append(patch)
    if return_normals:
        all_pc = transform_with_normals(all_pc, normals_at_kpt)

    return all_pc, pcd_down, normals_at_kpt

def transform_with_normals(all_pc, normals_at_kpt):
    normalize = lambda x: x / (np.linalg.norm(x) + 1e-5)
    up = np.array([0,-1,0],dtype=np.float32)
    all_pc_T = []
    for patch, normal in zip(all_pc, normals_at_kpt):
        axis_z = normalize(normal) 
        axis_y = up
        axis_x = normalize(np.cross(axis_y, axis_z)) 
        axis_y = normalize(np.cross(axis_z, axis_x)) 
        axis = np.concatenate((axis_x[:,None], axis_y[:,None], axis_z[:,None]), axis=1)
        all_pc_T.append(patch @ axis)
    return all_pc_T

def radius_ball_search_np(pc, kpt, search_radius, input_num=None, log=None):
    if log is not None:
        print(log)

    # radius-ball search
    keypoints = pc[kpt]
    maxpoints = 50000
    if pc.shape[0] > maxpoints:
        _, pc = pctk.uniform_resample_np(pc, maxpoints)

    search = KDTree(pc)
    results = search.query_ball_point(keypoints, search_radius)
    all_pc = []
    for indices in results:
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

def to_o3d(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return pcd

def smoothing(pc, vs=0.015):
    pcd = to_o3d(pc)
    pcd = pcd.voxel_down_sample(voxel_size=vs)
    return np.asarray(pcd.points)

def radius_ball_search_o3d_radii(pc, kpt_indices, radii, search_radius, input_num=None, msg=None):
    # radius-ball search
    keypoints = pc[kpt_indices]
    search = KDTree(pc)
    all_pc = []
    for idx, kpt in enumerate(kpt_indices):
        r = search_radius * 2
        indices = search.query_ball_point(pc[kpt], r)
        if len(indices) == 0:
            i = 1024 if input_num is None else input_num
            all_pc.append(np.zeros([i,3],dtype=np.float32))
        else:
            searched_thresh = search_radius * radii[indices] / 0.025
            candidates = pc[indices]
            dists = np.sqrt(np.sum((candidates - keypoints[idx][None])**2, axis=1))
            select = np.argwhere(dists <= searched_thresh).reshape(-1)
            subsampled = smoothing(candidates[select])
            all_pc.append(subsampled)
    return all_pc

import random
class PointCloudPairSampler(data.Sampler):

    def __init__(self, datasize, batch_size=1):
        self.datasize = datasize
        self.indices = self._generate_iter_indices()
        self.regen_flag = False

    def __iter__(self):
        if self.regen_flag:
            self.indices = self._generate_iter_indices()
        else:
            self.regen_flag = True
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def _generate_iter_indices(self):
        iter_indices = list(range(self.datasize))
        random.shuffle(iter_indices)
        return iter_indices


class FragmentLoader(data.Dataset):
    def __init__(self, opt, search_radius, npt=24, kptname='kpts', use_normals=False):
        super(FragmentLoader, self).__init__()
        self.opt = opt
        self.data_path = os.path.join(opt.dataset_path, 'fused_fragments')

        self.use_normals = use_normals

        scene_selection = None
        # scene_selection = ['sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']

        if kptname != 'kpts':
            print('[Dataloader] Using keypoint folder at %s!!!'%os.path.join(opt.dataset_path, kptname))
        if use_normals:
            print('[Dataloader] Normals are used to transform the input points!')
        if not self.opt.no_augmentation:
            print('[Dataloader] Using rotational augmentation!!')

        keypoint_folder = kptname
        keypoint_method = np.load
        keypoint_prefix = 'npy'
        parse_string = 'cloud_bin_{:d}-cloud_bin_{:d}.npy'
        self.keypoint_path = os.path.join(opt.dataset_path, keypoint_folder)
        self.search_radius = search_radius
        self.input_num = opt.model.input_num
        self.voxel_size = 0.03 if self.input_num < 1024 else 0.015
        # number of keypoints to sample from a fragment p
        self.npt = npt
        # get all the paths
        get_fragment_path = lambda scene,seq,idx: os.path.join(self.data_path, scene,seq, 'cloud_bin_%d.ply'%idx)
        def get_fragment_pose(secene, seq, idx):
            path1 = os.path.join(self.data_path, scene,seq, 'cloud_bin_%d.pose.npy'%idx)
            path2 = os.path.join(self.data_path, scene,seq, 'cloud_bin_%d_pose.txt'%idx)
            if os.path.exists(path1):
                return np.load(path1)
            else:
                return np.loadtxt(path2)
        # get_fragment_pose = lambda scene,seq,idx: np.load(os.path.join(self.data_path, scene,seq, 'cloud_bin_%d.pose.npy'%idx))
        self.kptfiles = []
        for scene in os.listdir(self.keypoint_path):
            if scene_selection is None or scene in scene_selection:
                seq_paths = [sq for sq in glob.glob(os.path.join(self.keypoint_path, scene,"seq*")) \
                             if os.path.isdir(sq)]
                if len(seq_paths) == 0:
                    seq_path = os.path.join(self.keypoint_path, scene)
                    seq = ""
                    for kptf in glob.glob(os.path.join(seq_path,"*.%s"%keypoint_prefix)):
                        idx1, idx2 = parse(parse_string, os.path.basename(kptf))
                        meta = Kptmeta(keypoint_method(kptf), f"{scene}_{seq}_{idx1}_{idx2}", \
                                       get_fragment_path(scene,seq,idx1), get_fragment_path(scene,seq,idx2),\
                                       get_fragment_pose(scene,seq,idx1), get_fragment_pose(scene,seq,idx2))
                        self.kptfiles.append(meta)
                else:
                    for seq_path in seq_paths:
                        seq = os.path.basename(seq_path)
                        for kptf in glob.glob(os.path.join(seq_path,"*.%s"%keypoint_prefix)):
                            idx1, idx2 = parse(parse_string, os.path.basename(kptf))
                            meta = Kptmeta(keypoint_method(kptf), f"{scene}_{seq}_{idx1}_{idx2}", \
                                           get_fragment_path(scene,seq,idx1), get_fragment_path(scene,seq,idx2),\
                                           get_fragment_pose(scene,seq,idx1), get_fragment_pose(scene,seq,idx2))
                            self.kptfiles.append(meta)

        print('[Dataloader] Training set length:', len(self.kptfiles))

    def __len__(self):
        return len(self.kptfiles)

    # loader with no scene order
    def __getitem__(self, index):
        meta = self.kptfiles[index]
        choice = np.random.choice(np.arange(meta.indices.shape[0]), self.npt)
        kpts = meta.indices[choice].astype(np.int32)
        pcdA = o3d.io.read_point_cloud(meta.pathA)
        pcdB = o3d.io.read_point_cloud(meta.pathB)
        rawA, pcdA_down, normalA = radius_ball_search_o3d(pcdA, kpts[:,0], self.search_radius, self.voxel_size, self.use_normals)
        rawB, pcdB_down, normalB = radius_ball_search_o3d(pcdB, kpts[:,1], self.search_radius, self.voxel_size, self.use_normals)

        pcdA = np.asarray(pcdA.points).astype(np.float32)
        pcdB = np.asarray(pcdB.points).astype(np.float32)

        # [nptxNx3]
        inputA = []
        inputB = []

        # preprocessing and augmentation
        T = RigidMatrix(meta.poseA).R.T @ RigidMatrix(meta.poseB).R

        self.R_aug_src = None
        self.R_aug_tgt = None        
        if not self.opt.no_augmentation:
            _, self.R_aug_src = pctk.rotate_point_cloud(None, max_degree=30)
            _, self.R_aug_tgt = pctk.rotate_point_cloud(None, max_degree=30)
            _, pcdA = pctk.rotate_point_cloud(pcdA, self.R_aug_src)
            _, pcdB = pctk.rotate_point_cloud(pcdB, self.R_aug_tgt)

        for pca, pcb in zip(rawA, rawB):
            inputA.append(self._preprocess(pca, self.R_aug_src))
            inputB.append(self._preprocess(pcb, self.R_aug_tgt))

        inputA = np.array(inputA)
        inputB = np.array(inputB)
        
        data = {'src': torch.from_numpy(inputA.astype(np.float32)),\
                'tgt': torch.from_numpy(inputB.astype(np.float32)),\
                'frag_src': pcdA,\
                'frag_tgt': pcdB,\
                'T': torch.from_numpy(T.astype(np.float32)),\
                'fn': meta.id}
        return data

    def _preprocess(self, pc, R_aug=None, n=None):
        idx, pc = pctk.uniform_resample_np(pc, self.input_num)
        if n is not None:
            n = n[idx]
        if R_aug is not None:
            # rotational augmentation
            pc, _ = pctk.rotate_point_cloud(pc, R_aug)
            if n is not None:
                n, _ = pctk.rotate_point_cloud(n, R_aug)
        if n is not None:
            pc = np.concatenate([pc,n],axis=1)
        return pc

class FragmentTestLoader(data.Dataset):
    def __init__(self, opt, test_path, search_radius, use_normals=False, npt=24):
        super(FragmentTestLoader, self).__init__()
        self.opt = opt
        self.data_path = test_path
        self.use_normals = use_normals

        if not self.opt.no_augmentation:
            print('[Dataloader] Using rotational augmentation!!')

        parse_string = 'cloud_bin_{:d}-cloud_bin_{:d}.keypts.npy'

        self.search_radius = search_radius
        self.input_num = opt.model.input_num
        # number of keypoints to sample from a fragment p
        self.voxel_size = 0.03 if self.input_num < 1024 else 0.015
        self.npt = npt
        # get all the paths
        get_fragment_path = lambda scene,idx: os.path.join(self.data_path, scene, 'cloud_bin_%d.ply'%idx)
        self.kptfiles = list()

        N_split_base = 2

        for scene in os.listdir(self.data_path):
            test_path = os.path.join(self.data_path, scene, 'lmvd_test_kpts')
            if os.path.isdir(test_path):
                for kptf in glob.glob(os.path.join(test_path,"*.keypts.npy")):
                    idx1, idx2 = parse(parse_string, os.path.basename(kptf))
                    kpts = np.load(kptf)
                    if kpts.shape[0] > N_split_base * npt:
                        arrs = np.array_split(kpts, N_split_base,0)
                        for arr in arrs:
                            meta = Kptmeta(arr, f"{scene}AT{idx1}_{idx2}", \
                                get_fragment_path(scene,idx1), get_fragment_path(scene,idx2),\
                                None, None)
                            self.kptfiles.append(meta)

        self.kptfiles = self.kptfiles[::10]
        print('[Dataloader] Testset length:', len(self.kptfiles))

    def __len__(self):
        return len(self.kptfiles)

    # loader with no scene order
    def __getitem__(self, index):
        meta = self.kptfiles[index]
        # choice = np.random.choice(np.arange(meta.indices.shape[0]), self.npt)
        kpts = meta.indices[:self.npt].astype(np.int32)
        pcdA = o3d.io.read_point_cloud(meta.pathA)
        pcdB = o3d.io.read_point_cloud(meta.pathB)

        rawA, pcdA_down, normalA = radius_ball_search_o3d(pcdA, kpts[:,0], self.search_radius, self.voxel_size, self.use_normals)
        rawB, pcdB_down, normalB = radius_ball_search_o3d(pcdB, kpts[:,1], self.search_radius, self.voxel_size, self.use_normals)

        # [nptxNx3(6)]
        inputA = []
        inputB = []

        for pca, pcb in zip(rawA, rawB):
            inputA.append(self._preprocess(pca))
            inputB.append(self._preprocess(pcb))

        inputA = np.array(inputA)
        inputB = np.array(inputB)
        data = {'src': torch.from_numpy(inputA.astype(np.float32)),\
                'tgt': torch.from_numpy(inputB.astype(np.float32)),\
                'frag_src': torch.from_numpy(np.asarray(pcdA.points).astype(np.float32)),\
                'frag_tgt': torch.from_numpy(np.asarray(pcdB.points).astype(np.float32)),\
                # 'T': torch.from_numpy(T.R.astype(np.float32)),\
                'id': meta.id}
        return data

    def _preprocess(self, pc, n=None):
        idx, pc = pctk.uniform_resample_np(pc, self.input_num)
        if n is not None:
            n = n[idx]
            pc = np.concatenate([pc,n],axis=1)
        return pc

# TEMP_TEST_FRAGMENT_PATH

class SceneEvalLoader(data.Dataset):
    def __init__(self, opt, scene):
        super(SceneEvalLoader, self).__init__()
        self.opt = opt
        self.data_path = os.path.join(opt.dataset_path, scene)
        self.batch_size = opt.batch_size
        self.search_radius = opt.model.search_radius # search_radius
        self.input_num = opt.model.input_num
        self.voxel_size = 0.03 if self.input_num < 1024 else 0.015
        self.use_normals = self.opt.model.normals

        # paths to data files
        self.kptsfiles = glob.glob(os.path.join(self.data_path,'01_Keypoints','cloud_bin_*Keypoints.txt'))
        # self.grouped = os.path.isdir(os.path.join(self.data_path, 'grouped_data_r%.2f'%self.search_radius))

        self.readkptf = lambda idx: \
                        np.loadtxt(os.path.join(self.data_path,'01_Keypoints',\
                        f'cloud_bin_{idx}Keypoints.txt')).astype(np.int32)
        self.readfrag = lambda idx: \
                        o3d.io.read_point_cloud(os.path.join(self.data_path, f'cloud_bin_{idx}.ply'))
        self.grouped_path = lambda idx: \
                        os.path.join(self.data_path, 'grouped_data_r%.2f'%self.search_radius,\
                        f'grouped_cloud_bin_{idx}.npz')
        self.readgdata = lambda idx: \
                        np.load(self.grouped_path(idx))['arr_0']

    def __len__(self):
        return len(self.kptsfiles)

    def __getitem__(self, index):
        frag = self.readfrag(index)
        if os.path.exists(self.grouped_path(index)):
            raw_clouds = self.readgdata(index).astype(np.float32)
            if self.use_normals:      
                kpts = self.readkptf(index)      
                if len(frag.normals) != len(frag.points):
                    raise RuntimeError('[!] The point cloud needs normals.')
                normals_at_kpt = np.asarray(frag.normals)[kpts]
                raw_clouds = np.array(transform_with_normals(raw_clouds, normals_at_kpt))
            
            if raw_clouds.shape[1] != self.input_num:
                # print(f'Founde grouped pc with {raw_clouds.shape[1]} points. Downsampling them to {self.input_num}...')
                clouds = list()
                for pc in raw_clouds:
                    clouds.append(self._process(pc))
                clouds = np.array(clouds).astype(np.float32)
            else:
                clouds = raw_clouds
        else:
            kpts = self.readkptf(index)
            outpath = self.grouped_path(index)
            print(f"Saving precompted patches to {outpath}...")
            clouds = list()
            raw_clouds, _, _ = radius_ball_search_o3d(frag, kpts, self.search_radius, self.voxel_size, self.use_normals)
            for pc in raw_clouds:
                pc = self._process(pc)
                # idx, pc = pctk.uniform_resample_np(pc, self.input_num)
                clouds.append(pc)
            clouds = np.array(clouds).astype(np.float32)
            np.savez(outpath, clouds)

        frag = np.asarray(frag.points)

        data = {'clouds': torch.from_numpy(clouds).float(),\
        'frag': torch.from_numpy(frag).float(),\
        # 'T': torch.from_numpy(T.R.astype(np.float32)),\
        'sid': index}

        return data

    def _process(self, pc):
        if pc.shape[0] != self.input_num:
            _, pc = pctk.uniform_resample_np(pc, self.input_num)
        return pc

class SceneTestLoader():
    def __init__(self, opt, grouped=False, datafilter=None):
        self.opt = opt
        self.data_path_root = opt.dataset_path
        self.batch_size = opt.batch_size
        self.search_radius = opt.model.search_radius # search_radius
        self.knn = opt.model.input_num
        self.grouped = grouped
        self.datafilter = datafilter
        self.ball_search = radius_ball_search_o3d # radius_ball_search_np # pctk.ball_search_np
        print('[Test Dataset] Using search radius at %f!!!'%self.search_radius)

    def prepare(self,scene):
        self.data_path = os.path.join(self.data_path_root,scene)
        self.current_scene = scene
        if self.grouped:
            # load npz files if patches are precomputed and grouped with self.precompute_patches()
            # self.search_radius = 0.34
            self.datafiles = glob.glob(os.path.join(self.data_path, 'grouped_data_r%.2f/*.npz'%self.search_radius))
            if len(self.datafiles) == 0:
                raise ValueError("Test data patches do not exst: %s"%self.data_path)
            if self.datafilter is not None:
                self.datafiles = list(filter(self.datafilter, self.datafiles))
            self.datafiles.sort(key=parse_scene_id)
            # if startsat is not None:
            #     self.datafiles = self.datafiles[startsat:]
            self.datasize = len(self.datafiles)
        else:
            self.kptsfiles = glob.glob(os.path.join(self.data_path,'01_Keypoints') + '/*.txt')
            if self.datafilter is not None:
                self.kptsfiles = list(filter(self.datafilter, self.kptsfiles))
            self.pcfiles = glob.glob(self.data_path + '/*.ply')
            if len(self.kptsfiles) == 0 or len(self.pcfiles) == 0:
                raise ValueError("Test data does not exst: %s"%self.data_path)
            self.pcfiles.sort(key=parse_scene_id)
            self.kptsfiles.sort(key=parse_scene_id)
            # if startsat is not None:
            #     self.kptsfiles = self.kptsfiles[startsat:]
            self.datasize = len(self.kptsfiles)
        self.batch_pt = 0
        self.scene_pt = -1
        self.reload()
        print(f'[Test Dataset] Loaded scene {scene}')

    def reload(self):
        self.scene_pt += 1
        self.batch_pt = 0

        if self.grouped:
            if self.scene_pt < len(self.datafiles):
                self.current_grouped_points = np.load(self.datafiles[self.scene_pt])['arr_0']
                self.current_sid = parse_scene_id(self.datafiles[self.scene_pt])
        else:
            if self.scene_pt < len(self.kptsfiles):
                self.current_kpts = np.loadtxt(self.kptsfiles[self.scene_pt], dtype=np.int32)
                self.current_sid = parse_scene_id(self.kptsfiles[self.scene_pt])

    def precompute_patches(self, scale=1.0, input_num=1024, num_worker=8):
        # output: # fragments x 5000 x nn x 3 npy files
        save_dir = os.path.join(self.data_path, 'grouped_data_r%.2f'%self.search_radius)
        os.makedirs(save_dir, exist_ok=True)
        mp_args = []
        sid_list = []
        for kid, kptf in enumerate(self.kptsfiles):
            kpts = np.loadtxt(kptf, dtype=np.int32)
            sid = parse_scene_id(kptf)
            assert str(sid) in self.pcfiles[sid]
            # scene_pcd = pctk.load_ply(self.pcfiles[sid])
            # mp_args.append([scene_pcd,kpts,self.search_radius, input_num, 'cloud_bin_%d'%sid])
            scene_pcd = o3d.io.read_point_cloud(self.pcfiles[sid])
            mp_args.append([scene_pcd, kpts, self.search_radius, 0.015, False, input_num])
            sid_list.append(sid)

        if num_worker > 1:
            pool = Pool(num_worker)
            rsts = pool.starmap(self.ball_search, mp_args)
            for rst,sid,arg in zip(rsts,sid_list,mp_args):
                # n_keypoints x knn x3
                grouped_points = np.array(rst[0]) * scale
                save_path = os.path.join(save_dir, 'grouped_cloud_bin_%d.npz'%sid)
                np.savez(save_path, grouped_points)
        else:
            # non parallel method
            for sid,arg in zip(sid_list,mp_args):
                print("Working on cloud bin #%d"%sid)
                rst = self.ball_search(*arg)
                grouped_points = np.array(rst[0]) * scale
                save_path = os.path.join(save_dir, 'grouped_cloud_bin_%d.npz'%sid)
                np.savez(save_path, grouped_points)

    def next_batch(self):
        buf = self.current_grouped_points if self.grouped else self.current_kpts
        if self.scene_pt >= self.datasize:
            return False
        if self.batch_pt + self.batch_size >= buf.shape[0]:
            kpts = buf[self.batch_pt:]
        else:
            kpts = buf[self.batch_pt: self.batch_pt+self.batch_size]
        if self.grouped:
            grouped_points = kpts
            if grouped_points.shape[1] != self.knn:
                resampled_points = []
                for pc in grouped_points:
                    _, pc_down = pctk.uniform_resample_np(pc, self.knn)
                    resampled_points.append(pc_down)
                grouped_points = np.array(resampled_points)
            # print(grouped_points.shape)
        else:
            grouped_indices = self.ball_search(self.current_scene, kpts, self.knn, self.search_radius)
            # # BxNx3
            grouped_points = self.current_scene[grouped_indices]

        self.batch_data = grouped_points
        self.batch_pt += self.batch_size
        if self.batch_pt >= buf.shape[0]:
            self.reload()
        return True

    def get_patches(self, sid, idx, normalize=False):
        if self.grouped:
            assert str(sid) in self.datafilter[sid]
            grouped_points = np.load(self.datafiles[sid], dtype=np.float32)['arr_0'][idx]
        else:
            assert str(sid) in self.kptsfiles[sid]
            assert str(sid) in self.pcfiles[sid]
            kpts = np.loadtxt(self.kptsfiles[sid], dtype=np.int32)[idx]
            cloud = pio.load_ply(self.pcfiles[sid])
            grouped_indices = self.ball_search(cloud, kpts, self.knn, self.search_radius)
            grouped_points = cloud[grouped_indices]

        return grouped_points

    @property
    def is_new_scene(self):
        return self.batch_pt == 0

    @property
    def current_scene_length(self):
        buf = self.current_grouped_points if self.grouped else self.current_kpts
        return buf.shape[0]

