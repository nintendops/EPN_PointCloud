import numpy as np
import trimesh
import os
import glob
import scipy.io as sio
import torch
import torch.utils.data as data
import vgtk.pc as pctk
import vgtk.point3d as p3dtk
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np, label_relative_rotation_np
from scipy.spatial.transform import Rotation as sciR

class Dataloader_ModelNet40(data.Dataset):
    def __init__(self, opt, mode=None):
        super(Dataloader_ModelNet40, self).__init__()
        self.opt = opt

        # 'train' or 'eval'
        self.mode = opt.mode if mode is None else mode

        # attention method: 'attention | rotation'
        self.flag = opt.model.flag

        self.anchors = L.get_anchors()

        if self.flag == 'rotation':
            cats = ['airplane']
            print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")
        else:
            cats = os.listdir(opt.dataset_path)

        self.dataset_path = opt.dataset_path
        self.all_data = []
        for cat in cats:
            for fn in glob.glob(os.path.join(opt.dataset_path, cat, self.mode, "*.mat")):
                self.all_data.append(fn)

        print("[Dataloader] : Training dataset size:", len(self.all_data))

        if self.opt.no_augmentation:
            print("[Dataloader]: USING ALIGNED MODELNET LOADER!")
        else:
            print("[Dataloader]: USING ROTATED MODELNET LOADER!")


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = sio.loadmat(self.all_data[index])
    
        if self.mode == 'train':
            _, pc = pctk.uniform_resample_np(data['pc'], self.opt.model.input_num)
        else:
            pc = data['pc']
    
        pc = p3dtk.normalize_np(pc.T)
        pc = pc.T

        R = np.eye(3)
        R_label = 29

        if not self.opt.no_augmentation:
            if 'R' in data.keys() and self.mode != 'train':
                pc, R = pctk.rotate_point_cloud(pc, data['R'])
            else:
                pc, R = pctk.rotate_point_cloud(pc)

            _, R_label, R0 = rotation_distance_np(R, self.anchors)

            if self.flag == 'rotation':
                R = R0

        return {'pc':torch.from_numpy(pc.astype(np.float32)),
                'label':torch.from_numpy(data['label'].flatten()).long(),
                'fn': data['name'][0],
                'R': R,
                'R_label': torch.Tensor([R_label]).long(),
               }

# for relative rotation alignment
class Dataloader_ModelNet40Alignment(data.Dataset):
    def __init__(self, opt, mode=None):
        super(Dataloader_ModelNet40Alignment, self).__init__()
        self.opt = opt

        # 'train' or 'eval'
        self.mode = opt.mode if mode is None else mode

        # attention method: 'attention | rotation'
        self.flag = opt.model.flag
        self.anchors = L.get_anchors(self.opt.model.kanchor)

        # if self.flag == 'rotation':
        #     cats = ['airplane']
        #     print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")
        # else:
        #     cats = os.listdir(opt.dataset_path)

        cats = ['airplane']
        print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")

        self.dataset_path = opt.dataset_path
        self.all_data = []
        for cat in cats:
            for fn in glob.glob(os.path.join(opt.dataset_path, cat, self.mode, "*.mat")):
                self.all_data.append(fn)
        print("[Dataloader] : Training dataset size:", len(self.all_data))


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = sio.loadmat(self.all_data[index])
        _, pc = pctk.uniform_resample_np(data['pc'], self.opt.model.input_num)

        # normalization
        pc = p3dtk.normalize_np(pc.T)
        pc = pc.T

        # R = np.eye(3)
        # R_label = 29

        # source shape
        # if 'R' in data.keys() and self.mode != 'train':
        #     pc_src, R_src = pctk.rotate_point_cloud(pc, data['R'])
        # else:
        #     pc_src, R_src = pctk.rotate_point_cloud(pc)

        pc_src, R_src = pctk.rotate_point_cloud(pc)
        # target shape

        # pc_tgt, R_tgt = pctk.rotate_point_cloud(pc)
        pc_tgt = pc

        # if self.mode == 'test':
        #     data['R'] = R
        #     output_path = os.path.join(self.dataset_path, data['cat'][0], 'testR')
        #     os.makedirs(output_path,exist_ok=True)
        #     sio.savemat(os.path.join(output_path, data['name'][0] + '.mat'), data)
        # _, R_label, R0 = rotation_distance_np(R, self.anchors)

        # T = R_src @ R_tgt.T
        T = R_src # @ R_tgt.T

        # RR_regress = np.einsum('abc,bj,ijk -> aick', self.anchors, T, self.anchors)
        # R_label = np.argmax(np.einsum('abii->ab', RR_regress),axis=1)
        # idxs = np.vstack([np.arange(R_label.shape[0]), R_label]).T
        # R = RR_regress[idxs[:,0], idxs[:,1]]
        R, R_label = label_relative_rotation_np(self.anchors, T)
        pc_tensor = np.stack([pc_src, pc_tgt])

        return {'pc':torch.from_numpy(pc_tensor.astype(np.float32)),
                'fn': data['name'][0],
                'T' : torch.from_numpy(T.astype(np.float32)),
                'R': torch.from_numpy(R.astype(np.float32)),
                'R_label': torch.Tensor([R_label]).long(),
               }
