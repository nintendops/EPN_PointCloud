set -x

EID=${1:-3dmatch_debug}
MODEL=${2:-inv_so3net_pn}
DATASET=${3:- ../Datasets/MScenes/train}
# DATASET=${3:- /gldata/ForHaiwei/MScenes/train}

# IF USING KPCONV
# python train_3dmatch.py experiment --experiment-id $EID -d $DATASET \
#        model -m $MODEL --search-radius 0.4 -k \
#        --dropout-rate 0.0 --input-num 1024 -lf 100 -b 1 --npt 24 -lr 1e-3 -i 68000 --save-freq 4000 --decay-step 8000 \
#        --no-augmentation \
#        train_loss --attention-loss-type no_reg --margin 1.0 --loss-type soft \
       # -r data/models/kponcv_pnbaseline_r26s5_normals/model_20201015_20:06:54/ckpt/kponcv_pnbaseline_r26s5_normals_net_Iter40000.pth

# IF USING SO3CONV
python train_3dmatch.py experiment --experiment-id $EID -d $DATASET model -m $MODEL --search-radius 0.4 -u attention \
       --dropout-rate 0.0 --input-num 1024 -lf 100 -b 1 --npt 16 -lr 1e-3 -i 150000 --save-freq 4000 --decay-step 20000 \
       --no-augmentation \
       train_loss --equi-alpha 0.0 --margin 1.0 --loss-type soft \
       # -r data/models/so3conv_3dmatch_mvdpnr4/model_20201106_13:10:11/ckpt/so3conv_3dmatch_mvdpnr4_net_Iter96000.pth

