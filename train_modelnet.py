from SPConvNets.trainer_modelnet import Trainer
from SPConvNets.options import opt

def process_opt_for_modelnet_training(opt):
	# TODO
	return opt


if __name__ == '__main__':
	opt.mode = 'train'
	opt = process_opt_for_modelnet_training(opt)
	trainer = Trainer(opt)
	trainer.train()
