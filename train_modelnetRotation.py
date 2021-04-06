from SPConvNets.trainer_modelnetRotation import Trainer as TrainerR
from SPConvNets.trainer_modelnet import Trainer
from SPConvNets.options import opt


if __name__ == '__main__':
	opt.mode = 'train'
	opt.model.flag = 'rotation'
	print("Performing a regression task...")
	opt.model.model = 'reg_so3net'
	trainer = TrainerR(opt)
	trainer.train()
