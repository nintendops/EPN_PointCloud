from ZPConvNets.trainer_3dmatch import Trainer
import ZPConvNets.datasets.evaluation_3dmatch as eval3dmatch
import os.path as osp

SCENE_TO_TEST = [    
              # 'kitchen',
              '7-scenes-redkitchen',
              'sun3d-home_at-home_at_scan1_2013_jan_1',
              'sun3d-home_md-home_md_scan9_2012_sep_30', 
              'sun3d-hotel_uc-scan3', 
              'sun3d-hotel_umd-maryland_hotel1', 
              'sun3d-hotel_umd-maryland_hotel3', 
              'sun3d-mit_76_studyroom-76-1studyroom2',
              'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
            ]

def our():
  from ZPConvNets.options import opt
  opt.mode = 'eval'
  assert opt.resume_path is not None
  opt.experiment_id = opt.resume_path.split('/')[2]
  trainer = Trainer(opt)
  trainer.eval(SCENE_TO_TEST)
  
def our_test():
  from ZPConvNets.options import opt
  opt.mode = 'test'
  assert opt.resume_path is not None
  opt.experiment_id = opt.resume_path.split('/')[2]
  trainer = Trainer(opt)
  trainer.test()

def others(dataset_path, target_folder):
    '''
        3D Match evaluation. Only works for invariant setting
    '''
    # set up where to store the output feature
    scene_name = osp.basename(dataset_path)
    eval3dmatch.evaluate_scene(dataset_path, target_folder, scene_name, suffix="lmvd")


if __name__ == '__main__':
  import fire
  # fire.Fire()
  our()
  # our_test()
