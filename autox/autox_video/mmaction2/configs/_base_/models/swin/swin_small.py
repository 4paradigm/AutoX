# model settings
_base_ = "swin_tiny.py"
model = dict(backbone=dict(depths=[2, 2, 18, 2]))
