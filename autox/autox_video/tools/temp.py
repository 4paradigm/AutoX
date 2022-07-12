from autox.autox_video import AutoXVideo

autox_video = AutoXVideo()

# Load dataset from cfg
autox_video.read_cfg('config.yaml')
autox_video.fit()
autox_video.transform()

# Manually specify datasets
autox_video.fit(
    data_root='data/demo/videos',
    ann_file_train='data/demo/annotations/train.txt',
    ann_file_val='data/demo/annotations/val.txt',
    video_length=2,
    num_class=24,
    videos_per_gpu=8
)
autox_video.transform(
    data_root='data/demo/videos',
    ann_file_test='data/demo/annotations/test.txt',
)

# transform only
autox_video.transform(
    data_root='data/demo/videos',
    ann_file_train='data/demo/annotations/test.txt',
    checkpoints='work_dirs/demo/latest.pth'
)

