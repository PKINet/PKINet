# Poly Kernel Inception Network for Remote Sensing Detection




## Introduction

This repository is the official implementation of "Poly Kernel Inception Network for Remote Sensing Detection".


## Results and models

### Pretrained models

Imagenet 300-epoch pretrained PKINet-T backbone: [Download](https://1drv.ms/u/s!AnQKQBp_pemcb_IYp_mNO2JcC-4?e=xTibzw)

Imagenet 300-epoch pretrained PKINet-S backbone: [Download](https://1drv.ms/u/s!AnQKQBp_pemca_Yxu6GPlXCV8hQ?e=aRLhO8)

### Experiments results

DOTAv1.0

|          Model           |  mAP  | Angle | Aug |                                                Configs                                                 |                                   Download                                 |
|:------------------------:|:-----:|:-----:| :-: |:------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| PKINet-T (1024,1024,200) | 77.87 | le90  |  -  | [pkinet-t_fpn_o-rcnn_dotav1-ss_le90](./configs/pkinet/pkinet-t_fpn_o-rcnn-dotav1-ss_le90.py) |      [model](https://1drv.ms/u/s!AnQKQBp_pemcbjfHQ1RXPKWWNQs?e=DnaaxP)     |
| PKINet-S (1024,1024,200) | 78.39 | le90  |  -  | [pkinet-s_fpn_o-rcnn_dotav1-ss_le90](./configs/pkinet/pkinet-s_fpn_o-rcnn-dotav1-ss_le90.py) | [model](https://1drv.ms/u/s!AnQKQBp_pemcbLu0tQfHY8872KM?e=yrdL4Z)|

DOTAv1.5

|          Model           |  mAP  | Angle | Aug |                                                 Configs                                                 |                             Download                             |
|:------------------------:|:-----:|:-----:| :-: |:-------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| PKINet-S (1024,1024,200) | 71.47 | le90  |  -  |[pkinet-s_fpn_o-rcnn_dotav15-ss_le90](./configs/pkinet/pkinet-s_fpn_o-rcnn-dotav15-ss_le90.py) |[model](https://1drv.ms/u/s!AnQKQBp_pemcbTwZ8mUPSkosGD4?e=Y3QZHa) |


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
