from SPConvNets.trainer_modelnetRotation import Trainer as TrainerR
from SPConvNets.trainer_modelnet import Trainer 
from SPConvNets.options import opt

opt.mode = 'eval'
trainer = TrainerR(opt)
trainer.eval()
