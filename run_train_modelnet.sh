set -x

EID=${1:-ModelnetCls_Debug}
MODEL=${2:-cls_so3net_pn}
DRATE=${3:-0.0}
LF=${4:-100}

python train_modelnet.py experiment --experiment-id $EID -d ../Datasets/EvenAlignedModelNet40PC -m $MODEL --kanchor 60 -u attention  \
       --dropout-rate $DRATE --input-num 1024 -b 12 --save-freq 5000 -lr 1e-3 -i 200000 --decay-rate 0.5 --decay-step 20000 -lf $LF \
       train_loss --temperature 3.0 --attention-loss-type default --attention-margin 1.0 \


# python train_modelnet.py experiment --experiment-id $EID -d ../Datasets/EvenAlignedModelNet40PC -m reg_so3net --kanchor 20 \
#        --dropout-rate $DRATE --input-num 1024 -b 8 --save-freq 5000 -lr 1e-3 --decay-rate 0.97 --decay-step 3000 -lf $LF -u rotation --representation quat \
#        train_loss --temperature 1 --attention-loss-type default --attention-margin 1.0 --attention-pretrain-step 2000 \
