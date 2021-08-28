from SPConvNets.trainer_3dmatch import Trainer
from SPConvNets.options import opt as opt_3dmatch

SCENE_TO_TEST = [    
              '7-scenes-redkitchen',
              'sun3d-home_at-home_at_scan1_2013_jan_1',
              'sun3d-home_md-home_md_scan9_2012_sep_30', 
              'sun3d-hotel_uc-scan3', 
              'sun3d-hotel_umd-maryland_hotel1', 
              'sun3d-hotel_umd-maryland_hotel3', 
              'sun3d-mit_76_studyroom-76-1studyroom2',
              'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
            ]

def config_opt_3dmatch(opt):
    opt.model.search_radius = 0.4
    opt.model.flag = 'attention'
    opt.model.model = "inv_so3net_pn"
    opt.no_augmentation = True

    if opt.mode == 'train':
        opt.npt = 2
        opt.batch_size = 1
        opt.num_iterations = 150000
        opt.save_freq = 4000
        opt.decay_step = 20000
    elif opt.mode == 'eval':
        opt.npt = 24
        opt.batch_size = 8

    return opt

if __name__ == '__main__':
    opt_3dmatch = config_opt_3dmatch(opt_3dmatch)
    trainer = Trainer(opt_3dmatch)
    
    if opt_3dmatch.mode == 'train':    
        trainer.train()
    elif opt_3dmatch.mode == 'eval':
        assert opt_3dmatch.resume_path is not None
        opt_3dmatch.experiment_id = opt_3dmatch.resume_path.split('/')[2]
        trainer = Trainer(opt_3dmatch)
        trainer.eval(SCENE_TO_TEST)
