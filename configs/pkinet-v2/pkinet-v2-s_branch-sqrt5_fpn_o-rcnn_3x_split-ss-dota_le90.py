_base_ = './pkinet-v2-s_fpn_o-rcnn_3x_dotav1-ss_le90.py'

data_root = 'data/split_ss_dota/'

model = dict(
    backbone=dict(
        type='PKINetV2_S_BranchSqrt5',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/pkinet-v2-s_pretrain.pth.tar')))

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
