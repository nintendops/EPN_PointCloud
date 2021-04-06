from ZPConvNets.trainer_modelnetRotation import Trainer as TrainerR
from ZPConvNets.trainer_modelnet import Trainer 
from ZPConvNets.options import opt

opt.mode = 'eval'
trainer = TrainerR(opt)
trainer.eval()
