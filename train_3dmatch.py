from SPConvNets.trainer_3dmatch import Trainer
from SPConvNets.options import opt

opt.mode = 'train'
trainer = Trainer(opt)
trainer.train()
