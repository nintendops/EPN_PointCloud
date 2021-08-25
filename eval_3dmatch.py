from SPConvNets.trainer_3dmatch import Trainer
import SPConvNets.datasets.evaluation_3dmatch as eval3dmatch
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
  from SPConvNets.options import opt
  opt.mode = 'eval'
  assert opt.resume_path is not None
  opt.experiment_id = opt.resume_path.split('/')[2]
  trainer = Trainer(opt)
  trainer.eval(SCENE_TO_TEST)
  

if __name__ == '__main__':
  our()
