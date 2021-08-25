set -x

MODEL=${2:-inv_so3net_pn}
DATASET=${3:-../Datasets/MScenes/evaluation/3DMatch}

# kp conv
# python eval_3dmatch.py experiment --experiment-id eval_3dmatch -d $DATASET -m $MODEL -k --search-radius 0.26 \
# --input-num 1024 -b 64 --npt 16 --no-augmentation -r $1

# so3 conv 
python eval_3dmatch.py experiment --experiment-id eval_3dmatch -d $DATASET -m $MODEL -u attention --search-radius 0.4 \
-b 8 --input-num 1024 --npt 24 --no-augmentation -r $1
