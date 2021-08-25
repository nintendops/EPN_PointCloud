set -x

MODEL=${2:-cls_so3net_pn}
TRAIN_LOSS=${3:-default}

python eval_modelnet.py experiment --experiment-id eval_modelnet -d ../Datasets/EvenAlignedModelNet40PC -m $MODEL --kanchor 1 --no-augmentation \
       -b 16 -s 2913 -r $1 train_loss --attention-loss-type $TRAIN_LOSS
