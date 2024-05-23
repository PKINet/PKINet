# Poly Kernel Inception Network for Remote Sensing Detection




## Introduction

This repository is the official implementation of "Poly Kernel Inception Network for Remote Sensing Detection".


## Results and models

### Pretrained models

Imagenet 300-epoch pretrained PKINet-T backbone: [Download](https://1drv.ms/u/c/9ce9a57f1a400a74/EXQKQBp_pekggJxvAAAAAAABWyCuNnKnuiA47qX6Wr7TMQ?e=pWhU1h)

Imagenet 300-epoch pretrained PKINet-S backbone: [Download](https://1drv.ms/u/c/9ce9a57f1a400a74/EXQKQBp_pekggJxrAAAAAAAB46whGHAZkAw-Pnkwgc_OWQ?e=n0NrZl)

### Experiments results

DOTAv1.0

|          Model           |  mAP  | Angle | Aug |                                                Configs                                                 |                                   Download                                 |
|:------------------------:|:-----:|:-----:| :-: |:------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| PKINet-T (1024,1024,200) | 77.87 | le90  |  -  | [pkinet-t_fpn_o-rcnn_dotav1-ss_le90](./configs/pkinet/pkinet-t_fpn_o-rcnn-dotav1-ss_le90.py) |      [model](https://1drv.ms/u/c/9ce9a57f1a400a74/EXQKQBp_pekggJxuAAAAAAABKAmGDsIXgkjY5_WjNzQorQ?e=Lcibnd)     |
| PKINet-S (1024,1024,200) | 78.39 | le90  |  -  | [pkinet-s_fpn_o-rcnn_dotav1-ss_le90](./configs/pkinet/pkinet-s_fpn_o-rcnn-dotav1-ss_le90.py) | [model](https://1drv.ms/u/c/9ce9a57f1a400a74/EXQKQBp_pekggJxsAAAAAAABWxHIx4vrnkZsRy1JW3BRaw?e=e07o7V)|

DOTAv1.5

|          Model           |  mAP  | Angle | Aug |                                                 Configs                                                 |                             Download                             |
|:------------------------:|:-----:|:-----:| :-: |:-------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| PKINet-S (1024,1024,200) | 71.47 | le90  |  -  |[pkinet-s_fpn_o-rcnn_dotav15-ss_le90](./configs/pkinet/pkinet-s_fpn_o-rcnn-dotav15-ss_le90.py) |[model](https://1drv.ms/u/c/9ce9a57f1a400a74/EXQKQBp_pekggJxtAAAAAAABYD69GUAHCtBp4RRSoKLuYQ?e=xh6iwO) |


## Installation

MMRotate-PKINet depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install yapf==0.40.1
pip install -U openmim
mim install mmcv-full
mim install mmdet
mim install mmengine
git clone 
cd PKINet
mim install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Citation
```
@inproceedings{cai2024pkinet,
  title={Poly Kernel Inception Network for Remote Sensing Detection},
  author={Cai, Xinhao and Lai, Qiuxia and Wang, Yuwei and Wang, Wenguan and Sun, Zeren and Yao, Yazhou},
  booktitle={CVPR},
  year={2024}
}
```
