
import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import vgtk


# TODO add dataparallel
# TODO add the_world = ipdb.set_trace

class Trainer():
    def __init__(self, opt):
        super(Trainer, self).__init__()

        opt_dict = vgtk.dump_args(opt)
        self.check_opt(opt)

        # set random seed
        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed_all(self.opt.seed)
        # np.set_printoptions(precision=3, suppress=True)

        # create model dir
        experiment_id = self.opt.experiment_id if self.opt.mode == 'train' else f"{self.opt.experiment_id}_{self.opt.mode}"
        model_id = f'model_{time.strftime("%Y%m%d_%H:%M:%S")}'
        self.root_dir = os.path.join(self.opt.model_dir, experiment_id, model_id)
        os.makedirs(self.root_dir, exist_ok=True)

        # saving opt
        opt_path = os.path.join(self.root_dir, 'opt.txt')
        # TODO: hierarchical args are not compatible wit json dump
        with open(opt_path, 'w') as fout:
            json.dump(opt_dict, fout, indent=2)

        # create logger
        log_path = os.path.join(self.root_dir, 'log.txt')
        self.logger = vgtk.Logger(log_file=log_path)
        self.logger.log('Setup', f'Logger created! Hello World!')
        self.logger.log('Setup', f'Random seed has been set to {self.opt.seed}')
        self.logger.log('Setup', f'Experiment id: {experiment_id}')
        self.logger.log('Setup', f'Model id: {model_id}')

        # ckpt dir
        self.ckpt_dir = os.path.join(self.root_dir, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger.log('Setup', f'Checkpoint dir created!')

        # build dataset
        self._setup_datasets()

        # create network
        self._setup_model()
        self._setup_optim()
        self._setup_metric()

        # init
        self.start_epoch = 0
        self.start_iter = 0

        # check resuming
        self._resume_from_ckpt(opt.resume_path)
        self._setup_model_multi_gpu()

        # setup summary
        self.summary = vgtk.Summary()

        # setup timer
        self.timer = vgtk.Timer()
        self.summary.register(['Time'])

        # done
        self.logger.log('Setup', 'Setup finished!')

    def train(self):
        self.opt.mode = 'train'
        self.model.train()
        if self.opt.num_epochs is not None:
            self.train_epoch()
        else:
            self.train_iter()

    def test(self):
        self.opt.mode = 'test'
        self.model.eval()

    def train_iter(self):
        for i in range(self.opt.num_iterations):
            self.timer.set_point('train_iter')
            self.lr_schedule.step()
            self.step()
            # print({'Time': self.timer.reset_point('train_iter')})
            self.summary.update({'Time': self.timer.reset_point('train_iter')})

            if i % self.opt.log_freq == 0:
                if hasattr(self, 'epoch_counter'):
                    step = f'Epoch {self.epoch_counter}, Iter {i}'
                else:
                    step = f'Iter {i}'
                self._print_running_stats(step)

            if i > 0 and i % self.opt.save_freq == 0:
                self._save_network(f'Iter{i}')
                self.test()

    def train_epoch(self):
        for i in range(self.opt.num_epochs):
            self.lr_schedule.step()
            self.epoch_step()

            if i % self.opt.log_freq == 0:
                self._print_running_stats(f'Epoch {i}')

            if i > 0 and i % self.opt.save_freq == 0:
                self._save_network(f'Epoch{i}')


    # TODO: check that the options have the required key collection
    def check_opt(self, opt, print_opt=True):
        self.opt = opt
        self.opt.device = torch.device('cuda')

    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')

    def step(self):
        raise NotImplementedError('Not implemented')

    def epoch_step(self):
        raise NotImplementedError('Not implemented')

    def _setup_datasets(self):
        self.logger.log('Setup', 'Setup datasets!')
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        raise NotImplementedError('Not implemented')

    def _setup_model(self):
        self.logger.log('Setup', 'Setup model!')
        self.model = None
        raise NotImplementedError('Not implemented')

    def _setup_model_multi_gpu(self):
        if torch.cuda.device_count() > 1:
            self.logger.log('Setup', 'Using Multi-gpu and DataParallel!')
            self._use_multi_gpu = True
            self.model = nn.DataParallel(self.model)
        else:
            self.logger.log('Setup', 'Using Single-gpu!')
            self._use_multi_gpu = False

    def _setup_optim(self):
        self.logger.log('Setup', 'Setup optimizer!')
        # torch.autograd.set_detect_anomaly(True)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.opt.train_lr.init_lr)
        self.lr_schedule = vgtk.LearningRateScheduler(self.optimizer,
                                                      **vars(self.opt.train_lr))
        self.logger.log('Setup', 'Optimizer all-set!')

    def _setup_metric(self):
        self.logger.log('Setup', 'Setup metric!')
        self.metric = None
        raise NotImplementedError('Not implemented')

    # def _resume_from_ckpt(self, resume_path):
    #     if resume_path is None:
    #         self.logger.log('Setup', f'Seems like we train from scratch!')
    #         return
    #     self.logger.log('Setup', f'Resume from checkpoint: {resume_path}')
    #     state_dicts = torch.load(resume_path)
    #     self.model.load_state_dict(state_dicts['model'])
    #     self.optimizer.load_state_dict(state_dicts['optimizer'])
    #     self.start_epoch = state_dicts['epoch']
    #     self.start_iter = state_dicts['iter']
    #     self.logger.log('Setup', f'Resume finished! Great!')

    def _resume_from_ckpt(self, resume_path):
        if resume_path is None:
            self.logger.log('Setup', f'Seems like we train from scratch!')
            return
        self.logger.log('Setup', f'Resume from checkpoint: {resume_path}')

        state_dicts = torch.load(resume_path)

        # self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(state_dicts)
        # self.model = self.model.module
        # self.optimizer.load_state_dict(state_dicts['optimizer'])
        # self.start_epoch = state_dicts['epoch']
        # self.start_iter = state_dicts['iter']
        self.logger.log('Setup', f'Resume finished! Great!')



    # TODO
    def _save_network(self, step, label=None,path=None):
        label = self.opt.experiment_id if label is None else label
        if path is None:
            save_filename = '%s_net_%s.pth' % (label, step)
            save_path = os.path.join(self.root_dir, 'ckpt', save_filename)
        else:
            save_path = f'{path}.pth'
            
        if self._use_multi_gpu:
            params = self.model.module.cpu().state_dict()
        else:
            params = self.model.cpu().state_dict()
        torch.save(params, save_path)

        if torch.cuda.is_available():
            # torch.cuda.device(gpu_id)
            self.model.to(self.opt.device)
        self.logger.log('Training', f'Checkpoint saved to: {save_path}!')
