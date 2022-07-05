def get_pipeline_cfg(
        clip_len=8, frame_interval=6, num_clips=1,
        is_test=False):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
    if not is_test:
        res = [
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    else:
        res = [
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='ThreeCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    return res

def get_dataset_cfg(
        data_root,
        ann_file_train,
        ann_file_val,
        ann_file_test,
        videos_per_gpu=8,
        video_length=2):
    train_pipeline = get_pipeline_cfg(video_length*4, 6, 1, False)
    test_pipeline = get_pipeline_cfg(video_length*4, 6, 1, True)
    data = dict(
        videos_per_gpu=videos_per_gpu,
        workers_per_gpu=2,
        val_dataloader=dict(
            videos_per_gpu=1,
            workers_per_gpu=2
        ),
        test_dataloader=dict(
            videos_per_gpu=1,
            workers_per_gpu=2
        ),
        train=dict(
            type='VideoDataset',
            ann_file=ann_file_train,
            data_prefix=data_root,
            pipeline=train_pipeline),
        val=dict(
            type='VideoDataset',
            ann_file=ann_file_val,
            data_prefix=data_root,
            pipeline=test_pipeline),
        test=dict(
            type='VideoDataset',
            ann_file=ann_file_test,
            data_prefix=data_root,
            pipeline=test_pipeline))
    return data