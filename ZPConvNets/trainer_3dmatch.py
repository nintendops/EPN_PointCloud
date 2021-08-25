from importlib import import_module
from ZPConvNets import FragmentLoader, FragmentTestLoader, PointCloudPairSampler, Dataloader_3dmatch_eval
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import os.path as osp

SEQUENTIAL_LOADER = False

class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.opt.train_loss.equi_alpha > 0:
            self.summary.register(['Loss', 'InvLoss', 'Pos', 'Neg', 'Acc', \
                'EquiLoss', 'EquiPos', 'EquiNeg', 'EquiAcc' ])
        else:
            self.summary.register(['Loss', 'Pos', 'Neg', 'Acc'])

        self.epoch_counter = 0
        self.iter_counter = 0

    def _setup_datasets(self):

        load_in_sequence = SEQUENTIAL_LOADER or (self.opt.model.model == 'equiv_zpnet')
        if load_in_sequence:
            self.logger.log('Dataloader', 'Loading data in scene order!!!')

        if self.opt.mode == 'train':
            # dataset = Dataloader_3dmatch(self.opt)

            dataset = FragmentLoader(self.opt, self.opt.model.search_radius, kptname=self.opt.dataset, \
                                     use_normals=self.opt.model.normals, npt=self.opt.npt)

            sampler = PointCloudPairSampler(len(dataset))

            self.dataset_train = torch.utils.data.DataLoader(dataset, \
                                             batch_size=self.opt.batch_size, \
                                             shuffle=False, \
                                             sampler=sampler,
                                             num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset_train)

        if self.opt.mode in ['train','test']:
            dataset_test = FragmentTestLoader(self.opt, \
                                            '../Datasets/MScenes/evaluation/3DMatch', \
                                            self.opt.model.search_radius, use_normals=self.opt.model.normals, npt=self.opt.npt)
            self.dataset_test = torch.utils.data.DataLoader(dataset_test, \
                                 batch_size=1, \
                                 shuffle=False, \
                                 num_workers=self.opt.num_thread)
            # self.dataset_test_iter = iter(self.dataset_test)

        if self.opt.mode == 'eval':
            self.dataset_train = None
            # self.dataset_test = Dataloader_3dmatch_test(self.opt, grouped=True)


    def _setup_eval_datasets(self, scene):
        dataset_eval = Dataloader_3dmatch_eval(self.opt, scene)
        self.dataset_eval = torch.utils.data.DataLoader(dataset_eval, \
                            batch_size=1, \
                            shuffle=False, \
                            num_workers=1)

    def _setup_model(self):
        param_outfile = osp.join(self.root_dir, "params.json")
        module = import_module('ZPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)
        # flag for whether the model requires a input fragment (besides patches)
        self.smooth_model = type(self.model) == import_module('ZPConvNets.models').inv_so3net_smooth.InvSO3ConvSmoothModel


    def _setup_metric(self):
        self.anchors = self.model.get_anchor().to(self.opt.device)
        self.metric = vgtk.loss.TripletBatchLoss(self.opt,\
                                                 self.anchors,
                                                 alpha = self.opt.train_loss.equi_alpha) \

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset_train)):
            self._optimize(data)

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset_train)
            data = next(self.dataset_iter)
        self._optimize(data)
        # log_info = {
        #         'Loss': 0,
        #         'Pos': 0,
        #         'Neg': 0,
        #         'Acc': 0,
        #     }

        # self.summary.update(log_info)

    def _prepare_input(self, data):
        in_tensor_src = data['src'].to(self.opt.device)
        in_tensor_tgt = data['tgt'].to(self.opt.device)
        nchannel = in_tensor_src.shape[-1]
        in_tensor_src = in_tensor_src.view(-1, self.opt.model.input_num, nchannel)
        in_tensor_tgt = in_tensor_tgt.view(-1, self.opt.model.input_num, nchannel)

        # ----------------- DEBUG only -----------------------
        # gtt = data['T'][0].numpy()

        # data = in_tensor_src.cpu().numpy()
        # data_tgt = in_tensor_tgt.cpu().numpy()
        # data_tgt = data_tgt @ gtt.T

        # print(gtt)
        
        # for i in range(data.shape[0]):
        #     pctk.save_ply(osp.join('vis_gpu07', "train%d_src_patches.ply"%i), pctk.cent(data[i]))
        #     pctk.save_ply(osp.join('vis_gpu07', "train%d_tgt_patches.ply"%i), pctk.cent(data_tgt[i]))

        # import ipdb; ipdb.set_trace()
        # ---------------------------------------------------

        if self.smooth_model:
            fragment_src = data['frag_src'].to(self.opt.device).squeeze()
            in_tensor_src = (in_tensor_src, fragment_src)            
            fragment_tgt = data['frag_tgt'].to(self.opt.device).squeeze()
            in_tensor_tgt = (in_tensor_tgt, fragment_tgt)
            
        return in_tensor_src, in_tensor_tgt


    def _optimize(self, data):

        gt_T = data['T'].to(self.opt.device)

        in_tensor_src, in_tensor_tgt = self._prepare_input(data)

        ####################################################
        # gt_T = self.anchors[3][None]       
        # in_tensor_src = torch.matmul(in_tensor_tgt, gt_T[0].T)
        #######################################################
        
        # in_tensors = torch.cat((in_tensor_src, in_tensor_tgt), dim=0)
        # ys, yw = self.model(in_tensors)
        # y_src, y_tgt = torch.chunk(ys, 2, dim=0)
        # yw_src, yw_tgt = torch.chunk(yw, 2, dim=0)
        y_src, yw_src = self.model(in_tensor_src)
        y_tgt, yw_tgt = self.model(in_tensor_tgt)
        
        self.optimizer.zero_grad()

        if self.opt.train_loss.equi_alpha > 0:
            self.loss, inv_info, equi_info = self.metric(y_src, y_tgt, gt_T, yw_src, yw_tgt)
            invloss, pos_loss, neg_loss, accuracy = inv_info
            equiloss, equi_accuracy, equi_pos_loss, equi_neg_loss = equi_info
        else:
            self.loss, accuracy, pos_loss, neg_loss = self.metric(y_src, y_tgt, gt_T)
            
        self.loss.backward()
        self.optimizer.step()

        # Log training stats
        if self.opt.train_loss.equi_alpha > 0:
            log_info = {
                'Loss': self.loss.item(),
                'InvLoss': invloss.item(),
                'Pos': pos_loss.item(),
                'Neg': neg_loss.item(),
                'Acc': 100 * accuracy.item(),
                'EquiLoss': equiloss.item(),
                'EquiPos': equi_pos_loss.item(),
                'EquiNeg': equi_neg_loss.item(),
                'EquiAcc': 100 * equi_accuracy.item(),
            }
        else:
            log_info = {
                'Loss': self.loss.item(),
                'Pos': pos_loss.item(),
                'Neg': neg_loss.item(),
                'Acc': 100 * accuracy.item(),
            }
        self.summary.update(log_info)
        self.iter_counter += 1


    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self):
        self.logger.log('Testing','Evaluating test set!')
        self.model.eval()
        self.metric.eval()
        with torch.no_grad():
            all_acc = {}
            for it, data in enumerate(self.dataset_test):                
                in_tensor_src, in_tensor_tgt = self._prepare_input(data)
                scene_id = data['id'][0].split('AT')[0]
                if scene_id not in all_acc.keys():
                    all_acc[scene_id] = list()
                # in_tensors = torch.cat((in_tensor_src, in_tensor_tgt), dim=0)
                # ys, yw = self.model(in_tensors)
                # y_src, y_tgt = torch.chunk(ys, 2, dim=0)
                # yw_src, yw_tgt = torch.chunk(yw, 2, dim=0)

                y_src, yw_src = self.model(in_tensor_src)
                y_tgt, yw_tgt = self.model(in_tensor_tgt)

                # if self.opt.model.flag == 'attention':
                #     self.loss, invloss, rloss, accuracy, pos_loss, neg_loss = self.metric(y_src, y_tgt, None, [yw_src, yw_tgt])
                # else:

                self.loss, accuracy, pos_loss, neg_loss = self.metric(y_src, y_tgt, None)

                # Log training stats
                if self.opt.train_loss.equi_alpha > 0:
                    log_info = {
                        'Loss': self.loss.item(),
                        'InvLoss': self.loss.item(),
                        'Pos': pos_loss.item(),
                        'Neg': neg_loss.item(),
                        'Acc': 100 * accuracy.item(),
                        'EquiLoss': 0,
                        'EquiPos': 0,
                        'EquiNeg': 0,
                        'EquiAcc': 0,
                    }
                else:
                    log_info = {
                        'Loss': self.loss.item(),
                        'Pos': pos_loss.item(),
                        'Neg': neg_loss.item(),
                        'Acc': 100 * accuracy.item(),
                    }


                all_acc[scene_id].append(100 * accuracy.item())
                self.logger.log('Testing', '\t'.join(f'{k}:{v:.4f}' for k,v in log_info.items()))


            for scene in all_acc.keys():
                scene_acc = np.array(all_acc[scene])
                self.logger.log('Testing', "The mean accuracy for scene %s is only %.2f..."%(scene,np.mean(scene_acc)))

        self.model.train()
        self.metric.train()

    def eval(self, select):
        '''
            3D Match evaluation. Only works for invariant setting
        '''
        from ZPConvNets.datasets import evaluation_3dmatch as eval3dmatch

        # set up where to store the output feature
        all_results = dict()
        for scene in select:
            assert osp.isdir(osp.join(self.opt.dataset_path, scene))
            print(f"Working on scene {scene}...")
            target_folder = osp.join('data/evaluate/3DMatch/', self.opt.experiment_id, scene, f'{self.opt.model.output_num}_dim')

        ###############################################
            # if osp.isdir(target_folder):
            #     print("Target folder already exists: %s"%target_folder)
            #     print("Overwriting features...")
            # self.dataset_test.prepare(scene)
            ###########################################

            self._setup_eval_datasets(scene)
            self._generate(target_folder)
            # recalls: [tau, ratio]
            results = eval3dmatch.evaluate_scene(self.opt.dataset_path, target_folder, scene)
            all_results[scene] = results
        
        self._write_csv(all_results)
        print("Done!")


    def _generate(self, target_folder):
        with torch.no_grad():
            self.model.eval()
            bs = self.opt.batch_size
            # sid = self.dataset_test.current_sid
            # print("\nWorking on fragment id", sid)
            # feature_buffer = []
            print("\n---------- Evaluating the network! ------------------")
            # ctn = 0

            ################### NEW EVAL LOADER ###############################3
            from tqdm import tqdm
            for it, data in enumerate(self.dataset_eval):
                sid = data['sid'].item()
                # scene = data['scene']

                checknan = lambda tensor: torch.sum(torch.isnan(tensor))
                
                print("\nWorking on fragment id", sid)
                n_keypoints = data['clouds'].shape[0]
                # 5000 x N x 3
                clouds = data['clouds'].to(self.opt.device).squeeze()
                npt = clouds.shape[0]

                if self.smooth_model:
                    frag = data['frag'].to(self.opt.device).squeeze()

                feature_buffer = []
                for bi in tqdm(range(0, npt, bs)):
                    in_tensor_test = clouds[bi : min(npt,bi+bs)]
                    if self.smooth_model:
                        in_tensor_test = (in_tensor_test, frag)
                    feature, _ = self.model(in_tensor_test)
                    feature_np = feature.detach().cpu().numpy()
                    if checknan(feature).item() > 0:
                        feature_np = np.nan_to_num(feature_np)
                    feature_buffer.append(feature_np)
                    # print("Batch counter at %d/%d"%(bi, npt), end='\r')

                # target_folder = osp.join('data/evaluate/3DMatch/', self.opt.experiment_id, scene, f'{self.opt.model.output_num}_dim')
                os.makedirs(target_folder, exist_ok=True)
                feature_out = np.vstack(feature_buffer)
                out_path = osp.join(target_folder, "feature%d.npy"%sid)
                print(f"\nSaving features to {out_path}")
                np.save(out_path, feature_out)
            ######################################################################

            ########################## DEPRECATED EVAL LOADER ###############################
            # while self.dataset_test.next_batch():
            #     in_tensor_test = torch.from_numpy(self.dataset_test.batch_data).float().to(self.opt.device)

            #     ######################## DEBUG ##########################3
            #     # patches_sample_path = 'vis/3dmatch_samples/eval'
            #     # os.makedirs(patches_sample_path, exist_ok=True)
            #     # for idx, pc in enumerate(self.dataset_test.batch_data):
            #     #     pctk.save_ply(osp.join(patches_sample_path, 'test_patches%d.ply'%(ctn*8+idx)), pctk.cent(pc))
            #     # ctn += 1
            #     # import ipdb; ipdb.set_trace()
            #     #########################################################

            #     feature, _ = self.model(in_tensor_test)
            #     feature_np = feature.detach().cpu().numpy()
            #     feature_buffer.append(feature_np)
            #     print("Batch counter at %d/%d"%(self.dataset_test.batch_pt, self.dataset_test.current_scene_length), end='\r')
            #     if self.dataset_test.is_new_scene:
            #         feature_out = np.vstack(feature_buffer)
            #         print("\nSaving features with shape", feature_out.shape)
            #         np.save(osp.join(target_folder, "feature%d.npy"%sid), feature_out)
            #         feature_buffer = []
            #         sid = self.dataset_test.current_sid
            #         print("\nWorking on fragment id", sid)
            # print("\n ------------------------------------------------")
            ################################################################################S

    def _write_csv(self, results):
        import csv
        from ZPConvNets.datasets import evaluation_3dmatch as eval3dmatch

        csvpath = osp.join('data/evaluate/3DMatch/', self.opt.experiment_id, 'recall.csv')
        with open(csvpath, 'w', newline='') as csvfile:
            fieldnames = ['Scene'] + ['tau_%.2f'%tau for tau in eval3dmatch.TAU_RANGE]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for scene in results.keys():
                recalls = results[scene]
                row = dict()
                row['Scene'] = scene
                for tup in recalls:
                    tau, ratio = tup
                    row['tau_%.2f'%tau] = "%.2f"%ratio
                writer.writerow(row)

        ### print out the stats
        all_recall = []
        for scene in results.keys():
            tau, ratio = results[scene][0]
            print("%s recall is %.2f at tau %.2f"%(scene, ratio, tau))
            all_recall.append(ratio)

        avg = np.array(all_recall).mean()
        print("Average recall is %.2f !" % avg)
                
