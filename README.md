## ZSTTT - Zero-Shot Test Time Training

This project introduces several approaches and analytical tools for test time training for zero-shot classification. It builds upon the codebase of the CLSWGAN paper:

### Feature Generating Networks for Zero-shot Learning. Yongqin Xian, Tobias Lorenz, Bernt Schiele, Zeynep Akata. IEEE CVPR 2018.

This is the code for our CVPR 2018 paper on CNN visual features generation using conditional GANs. You can use it to reproduce the results reported in the paper.

#### Setup Instructions

``conda install pytorch torchvision cudatoolkit -c pytorch``
``conda install -c conda-forge scikit-learn scipy tqdm timm tensorboard wandb matplotlib``

#### How to reproduce the results:

1.Download data of CUB, AWA and SUN from www.mpi-inf.mpg.de/zsl-benchmark

2.Modify default "dataroot" in clswgan.py or in shell scripts to point to your data path

3.Run one of the scripts, e.g. ./scripts/reproduce_zsl_flowers.sh

#### Citation

If you find this useful, please cite our work as follows:
@inproceedings {xianCVPR18,     
 title = {Feature Generating Networks for Zero-Shot Learning},  
 booktitle = {IEEE Computer Vision and Pattern Recognition (CVPR)},     
 year = {2018},     
 author = {Yongqin Xian and Tobias Lorenz and Bernt Schiele and Zeynep Akata} 
} 
