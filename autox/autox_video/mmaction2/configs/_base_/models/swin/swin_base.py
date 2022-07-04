# model settings
_base_ = "swin_tiny.py"
model = dict(backbone=dict(depths=[2, 2, 18, 2],
                           embed_dim=128,
                           num_heads=[4, 8, 16, 32]),
             cls_head=dict(in_channels=1024))
