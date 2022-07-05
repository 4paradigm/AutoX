# optimizer
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)

# runtime settings
find_unused_parameters = False
evaluation = dict(
    interval=2, metrics=['top_k_accuracy'])

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

gpu_ids = range(0,1)