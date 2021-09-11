
# Equivariant Point Network (EPN)

This repository contains the code (in PyTorch) for [Equivariant Point Network for 3D Point Cloud Analysis](https://arxiv.org/abs/2103.14147)  (CVPR'2021) by Haiwei Chen, [Shichen Liu](https://shichenliu.github.io/), [Weikai Chen](http://chenweikai.github.io/) and [Hao Li](http://www.hao-li.com/Hao_Li/Hao_Li_-_about_me.html).


## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Experiments](#experiments)
4. [Contacts](#contacts)

## Introduction

EPN is a SE(3)-equivariant network model that is designed for deep point cloud analysis. The core of the architecture is the **SE(3) Separable Convolution** that combines two sequential, equivariant convolution layers to approximate convolution in the SE(3) space. With the incorporation of an attention mechanism, the EPN network model can bes used to extract both SE(3) equivariant features and selectively pooled invariant features for various feature learning tasks.

## Usage

The code has been tested on Python3.7, PyTorch ? and CUDA (10.1). The module and additional dependencies can be installed with 
```
cd vgtk
python setup.py install
```

## Experiments

To be updated...


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
