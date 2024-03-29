max_epochs = 60
base_lr = 0.0025
val_interval = 3

# evaluation
evaluation = dict(interval=val_interval, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[33, 42, 51, 57])
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=1)
