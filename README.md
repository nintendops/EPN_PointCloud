
# Equivariant Point Network (EPN)

This repository contains the code (in PyTorch) for [Equivariant Point Network for 3D Point Cloud Analysis](https://arxiv.org/abs/2103.14147)  (CVPR'2021) by Haiwei Chen, [Shichen Liu](https://shichenliu.github.io/), [Weikai Chen](http://chenweikai.github.io/) and [Hao Li](http://www.hao-li.com/Hao_Li/Hao_Li_-_about_me.html).


## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Experiments](#experiments)
4. [Contact](#contact)

## Introduction

EPN is a SE(3)-equivariant network model that is designed for deep point cloud analysis. The core of the architecture is the **SE(3) Separable Convolution** that combines two sequential, equivariant convolution layers to approximate convolution in the SE(3) space. With the incorporation of an attention mechanism, the EPN network model can be used to extract both SE(3) equivariant features and selectively pooled invariant features for various feature learning tasks.

![](https://github.com/nintendops/EPN_PointCloud/blob/main/media/spconv.png)



## Usage

The code has been tested on Python3.7, PyTorch 1.7.1 and CUDA (10.1). The module and additional dependencies can be installed with 
```
cd vgtk
python setup.py install
```

## Experiments

**Datasets**

The rotated Modelnet40 point cloud dataset is generated from the [Aligned Modelnet40 subset](https://github.com/lmb-freiburg/orion) and can be downloaded using this [link](https://drive.google.com/file/d/1xRoYjz2KCwkyIPf21E-WKIZkjLYabPgJ/view?usp=sharing).

The original 3DMatch training and evaluation dataset can be found [here](https://3dmatch.cs.princeton.edu/#keypoint-matching-benchmark). We followed [this repo](https://github.com/craigleili/3DLocalMultiViewDesc) to preprocess rgb frames into fused fragments and extract matching keypoints for training. The preprocessed data ready for training can be downloaded [here](https://drive.google.com/file/d/1ME42RjtrNJNz1zSTBrO2NtG89fsOkQLv/view?usp=sharing) (146GB). We also prepared to preprocessed 3DMatch evaluation dataset [here](https://drive.google.com/file/d/14ZGJZHuQLhg87En4C5po6bgTFn4tns4R/view?usp=sharing) (40GB), where local patches around testing keypoints have been precomputed.

**Pretrained Model**

Pretrained model can be downloaded using this [link](https://drive.google.com/file/d/1vy9FRGWQsuVi4nf--YIqg_8yHFiWWJhh/view?usp=sharing)

**Training**

The following lines can be used for the training of each experiment

```
# modelnet classification
CUDA_VISIBLE_DEVICES=0 python run_modelnet.py experiment -d PATH_TO_MODELNET40
# modelnet shape alignment
CUDA_VISIBLE_DEVICES=0 python run_modelnet_rotation.py experiment -d PATH_TO_MODELNET40
# 3DMatch shape registration
CUDA_VISIBLE_DEVICES=0 python run_3dmatch.py experiment -d PATH_TO_3DMATCH
```

**Evaluation**

The following lines can be used for the evaluation of each experiment

```
# modelnet classification
CUDA_VISIBLE_DEVICES=0 python run_modelnet.py experiment -d PATH_TO_MODELNET40 -r PATH_TO_CKPT --run-mode eval
# modelnet shape alignment
CUDA_VISIBLE_DEVICES=0 python run_modelnet_rotation.py experiment -d PATH_TO_MODELNET40 -r PATH_TO_CKPT --run-mode eval
# 3DMatch shape registration
CUDA_VISIBLE_DEVICES=0 python run_3dmatch.py experiment -d PATH_TO_3DMATCH -r PATH_TO_CKPT --run-mode eval
```


## Contact

Haiwei Chen: chw9308@hotmail.com
Any discussions or concerns are welcomed!

**Citation**
If you find our project useful in your research, please consider citing:

```
@article{chen2021equivariant,
  title={Equivariant Point Network for 3D Point Cloud Analysis},
  author={Chen, Haiwei and Liu, Shichen and Chen, Weikai and Li, Hao and Hill, Randall},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14514--14523},
  year={2021}
}
```
