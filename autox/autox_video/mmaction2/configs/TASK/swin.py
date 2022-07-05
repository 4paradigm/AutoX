_base_ = [
    '../_base_/models/swin/swin_base.py', '../_base_/default_runtime.py', '../_base_/default_train_set.py'
]

load_from = 'checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
