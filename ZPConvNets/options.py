
import vgtk


parser = vgtk.HierarchyArgmentParser()

# Experiment arguments
exp_args = parser.add_parser("experiment")
exp_args.add_argument('--experiment-id', type=str, default='playground',
                      help='experiment id')
exp_args.add_argument('-d', '--dataset-path', type=str, default='data/train/7-scenes-redkitchen',
                      help='path to datasets')
exp_args.add_argument('--dataset', type=str, default='kpts',
                      help='name of the datasets')
exp_args.add_argument('--model-dir', type=str, default='data/models',
                      help='path to models')
exp_args.add_argument('-s', '--seed', type=int, default=2913,
                      help='random seed')
exp_args.add_argument('--run-mode', type=str, default='train',
                      help='train | test | eval')

# Network arguments
net_args = parser.add_parser("model")
net_args.add_argument('-m', '--model', type=str, default='inv_so3net',
                      help='type of model to use')
net_args.add_argument('--input-num', type=int, default=1024,
                      help='the number of the input points')
net_args.add_argument('--output-num', type=int, default=32,
                      help='the number of the input points')
net_args.add_argument('--search-radius', type=float, default=0.4)
net_args.add_argument('--normalize-input', action='store_true',
                      help='normalize the input points')
net_args.add_argument('--dropout-rate', type=float, default=0.,
                      help='dropout rate, no dropout if set to 0')
net_args.add_argument('--init-method', type=str, default="xavier",
                      help='method for weight initialization')
net_args.add_argument('-k','--kpconv', action='store_true', help='If set, use a kpconv structure instead')
net_args.add_argument('--kanchor', type=int, default=60, help='# of anchors used: {1,20,40,60}')
net_args.add_argument('--normals', action='store_true', help='If set, add normals to the input')
net_args.add_argument('-u', '--flag', type=str, default='max',
                      help='pooling method: max | mean | attention | rotation')
net_args.add_argument('--representation', type=str, default='quat',
                      help='how to represent rotation: quaternion | ortho6d ')



# Training arguments
train_args = parser.add_parser("train")
train_args.add_argument('-e', '--num-epochs', type=int, default=None,
                        help='maximum number of training epochs')
train_args.add_argument('-i', '--num-iterations', type=int, default=1000000,
                        help='maximum number of training iterations')
train_args.add_argument('-b', '--batch-size', type=int, default=2,
                        help='batch size to train')
train_args.add_argument('--npt', type=int, default=24,
                        help='number of point per fragment')
train_args.add_argument('-t', '--num-thread', default=8, type=int,
                        help='number of threads for loading data')
train_args.add_argument('--no-augmentation', action="store_true",
                        help='no data augmentation if set true')
train_args.add_argument('-r','--resume-path', type=str, default=None,
                        help='Training using the pre-trained model')
train_args.add_argument('--save-freq', type=int, default=5000,
                        help='the frequency of saving the checkpoint (iters)')
train_args.add_argument('-lf','--log-freq', type=int, default=100,
                        help='the frequency of logging training info (iters)')
train_args.add_argument('--eval-freq', type=int, default=5000,
                        help='frequency of evaluation (iters)')
train_args.add_argument('--debug-mode', type=str, default=None,
                        help='if specified, train with a certain debug procedure')


# Learning rate arguments
lr_args = parser.add_parser("train_lr")
lr_args.add_argument('-lr', '--init-lr', type=float, default=1e-3,
                     help='the initial learning rate')
lr_args.add_argument('-lrt', '--lr-type', type=str, default='exp_decay',
                     help='learning rate schedule type: exp_decay | constant')
lr_args.add_argument('--decay-rate', type=float, default=0.5,
                     help='the rate of exponential learning rate decaying')
lr_args.add_argument('--decay-step', type=int, default=10000,
                     help='the frequency of exponential learning rate decaying')

# Loss funtion arguments
loss_args = parser.add_parser("train_loss")
loss_args.add_argument('--loss-type', type=str, default='soft',
                       help='type of loss function')
loss_args.add_argument('--attention-loss-type', type=str, default='no_reg',
                       help='type of attention loss function: default | schedule | no_reg ')
loss_args.add_argument('--margin', type=float, default=1.0,
                       help='margin of hard batch loss')
loss_args.add_argument('--temperature', type=float, default=3,
                       help='attention temperature')
loss_args.add_argument('--attention-margin', type=float, default=1.0,
                       help='margin of attention loss')
loss_args.add_argument('--attention-pretrain-step', type=int, default=2000,
                       help='step for scheduled pretrain (only used when attention type is schedule')
loss_args.add_argument('--equi-alpha', type=float, default=0.0,
                       help='weight for equivariance loss')
# Eval arguments
eval_args = parser.add_parser("eval")

# Test arguments
test_args = parser.add_parser("test")


opt = parser.parse_args()
opt.mode = opt.run_mode