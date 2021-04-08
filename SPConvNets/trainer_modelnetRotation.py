from importlib import import_module
from SPConvNets import Dataloader_ModelNet40, Dataloader_ModelNet40Alignment
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import torch.nn.functional as F
from sklearn.neighbors import KDTree

class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.summary.register(['Loss', 'Reg_Loss','Mean_Err', 'R_Acc'])
        self.epoch_counter = 0
        self.iter_counter = 0
        self.test_accs = []

    def _setup_datasets(self):
        self.opt.model.flag = 'rotation'
        dataloader = Dataloader_ModelNet40Alignment # Dataloader_ModelNet40

        if self.opt.mode == 'train':
            dataset = dataloader(self.opt)
            self.dataset = torch.utils.data.DataLoader(dataset, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)

        dataset_test = dataloader(self.opt, 'testR')
        self.dataset_test = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)


    def _setup_model(self):
        if self.opt.resume_path is not None:
            splits = os.path.basename(self.opt.resume_path).split('_net_')
            self.exp_name = splits[0] + splits[1][:-4]
            print("[trainer] setting experiment id to be %s!"%self.exp_name)
        else:
            self.exp_name = None

        if self.opt.mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

    def _setup_metric(self):
        # regressor + classifier
        import vgtk.so3conv.functional as L
        anchors = torch.from_numpy(L.get_anchors(self.opt.model.kanchor)).to(self.opt.device)

        if self.opt.model.representation == 'quat':
            out_channel = 4
        elif self.opt.model.representation == 'ortho6d':
            out_channel = 6
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%self.opt.model.representation)

        self.metric = vgtk.MultiTaskDetectionLoss(anchors, nr=out_channel)

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset)):
            self._optimize(data)

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
            if data['R_label'].shape[0] < self.opt.batch_size:
                raise StopIteration
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)

        self._optimize(data)
        self.iter_counter += 1

    def _optimize(self, data):
        in_tensors = data['pc'].to(self.opt.device)
        nb, _, npoint, _ = in_tensors.shape
        # in_tensors = torch.cat([in_tensors[:,0], in_tensors[:,1]],dim=0)
        in_rot_label = data['R_label'].to(self.opt.device).view(nb,-1)
        in_alignment = data['T'].to(self.opt.device).float()
        in_R = data['R'].to(self.opt.device).float()

        preds, y = self.model(in_tensors)
        self.optimizer.zero_grad()

        # TODO
        self.loss, cls_loss, l2_loss, acc, error = self.metric(preds, in_rot_label, y, in_R, in_alignment)
        self.loss.backward()
        self.optimizer.step()

        # Log training stats
        log_info = {
            'Loss': self.loss.item(),
            'Reg_Loss': l2_loss.item(),
            'Mean_Err': error.mean().item(),
            'R_Acc': 100 * acc.item(),
        }

        self.summary.update(log_info)


    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self):
        self.eval()
        return None

    def eval(self):
        self.logger.log('Testing','Evaluating test set!')
        self.model.eval()
        self.metric.eval()

        all_error = []
        all_acc = []

        with torch.no_grad():
            for it, data in enumerate(self.dataset_test):
                in_tensors = data['pc'].to(self.opt.device)
                nb, _, npoint, _ = in_tensors.shape
                # in_tensors = torch.cat([in_tensors[:,0], in_tensors[:,1]],dim=0)
                in_rot_label = data['R_label'].to(self.opt.device).view(nb,-1)
                in_alignment = data['T'].to(self.opt.device).float()
                in_R = data['R'].to(self.opt.device).float()

                preds, y = self.model(in_tensors)
                # TODO
                self.loss, cls_loss, l2_loss, acc, error = self.metric(preds, in_rot_label, y, in_R, in_alignment)
                # all_labels.append(in_label.cpu().numpy())
                # all_feats.append(feat.cpu().numpy())
                all_acc.append(acc.cpu().numpy())
                all_error.append(error.cpu().numpy())
                self.logger.log("Testing", "Accuracy: %.1f, error: %.2f!"%(100*acc.item(), error.mean().item()))

            # import ipdb; ipdb.set_trace()
            all_error = np.concatenate(all_error, 0)
            all_acc = np.array(all_acc, dtype=np.float32)
            self.logger.log('Testing', 'Average classifier acc is %.2f!!!!'%(100 * all_acc.mean()))
            self.logger.log('Testing', 'Median angular error is %.2f degree!!!!'%(np.median(all_error) * 180 / np.pi))
            
            if self.exp_name is not None:
                save_path = os.path.join('data','alignment_errors', f'{self.exp_name}_error.txt')
                np.savetxt(save_path,all_error)


        self.model.train()
        self.metric.train()
