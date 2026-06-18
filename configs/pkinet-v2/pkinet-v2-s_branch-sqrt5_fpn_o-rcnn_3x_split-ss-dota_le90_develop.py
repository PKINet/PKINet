_base_ = './pkinet-v2-s_fpn_o-rcnn_3x_dotav1-ss_le90_develop.py'

data_root = 'data/split_ss_dota/'

data = dict(
    train=dict(
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    val=dict(
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    test=dict(
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/'))
