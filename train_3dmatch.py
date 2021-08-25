from ZPConvNets.trainer_3dmatch import Trainer
from ZPConvNets.options import opt

opt.mode = 'train'
trainer = Trainer(opt)
trainer.train()
