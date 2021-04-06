# python evaluation.py \
#     --data-dir /Users/lustralisk/Downloads/3DMatch \
#     --output-dir /Users/lustralisk/Desktop/playground/SCN_test \
#     --scene-name 7-scenes-redkitchen \
#     --descriptor-name 3DSmooth \
#     --descriptor-dim 32_dim \
#     --num-thread 12 \
#     --verbose


# python evaluation.py \
#     --data-dir /home/ICT2000/chenh/Haiwei/ConicMatch/data/evaluate/3DMatch_eval \
#     --output-dir /home/ICT2000/chenh/Haiwei/3DSmoothNet/data/evaluate \
#     --scene-name kitchen \
#     --descriptor-name 3DSmooth \
#     --descriptor-dim 16_dim \
#     --rotation-equivariance pca \
#     --num-thread 12 \
#     --verbose

python evaluation.py \
    --data-dir ../data/evaluate/3DMatch_eval \
    --output-dir ../data/evaluate/conic_gamma_dr7 \
    --anchor-path ../anchors/sphere62.ply\
    --scene-name kitchen \
    --descriptor-name ours \
    --descriptor-dim 16_dim \
    --rotation-equivariance gt \
    --nn-sigma 0.1 \
    --num-thread 12 \
    --verbose


