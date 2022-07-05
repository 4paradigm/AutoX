# Copyright (c) OpenMMLab. All rights reserved.
import copy
import argparse
from tools.GetPipeline import get_dataset_cfg
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import yaml
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

os.chdir('mmaction2')
from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger,
                            register_module_hooks, setup_multi_processes)
os.chdir('..')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    default_cfg = 'mmaction2/configs/TASK/swin.py'
    cfg = Config.fromfile(default_cfg)
    if not os.path.exists(cfg.load_from):
        cfg.load_from = None
    with open('config.yaml','r') as cus_cfg:
        cus_cfg = yaml.load(cus_cfg, Loader=yaml.FullLoader)
    cfg.total_epochs = cus_cfg['epoch']
    cfg.model.cls_head.num_classes = cus_cfg['num_class']
    cfg.work_dir = cus_cfg['work_dir']
    cfg.data = get_dataset_cfg(
        data_root=cus_cfg['data_root'],
        ann_file_train=cus_cfg['ann_file_train'],
        ann_file_val=cus_cfg['ann_file_val'],
        ann_file_test=cus_cfg['ann_file_test'],
        videos_per_gpu=cus_cfg['videos_per_gpu'],
        video_length=cus_cfg['video_length'])

    resume_path = os.path.join(cus_cfg['work_dir'], 'latest.pth')
    if os.path.exists(os.path.join(cus_cfg['work_dir'], 'latest.pth')):
        cfg.resume_from = resume_path

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(42, distributed=distributed)
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed, deterministic=False)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(default_cfg)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    if cfg.omnisource:
        # If omnisource flag is set, cfg.data.train should be a list
        assert isinstance(cfg.data.train, list)
        datasets = [build_dataset(dataset) for dataset in cfg.data.train]
    else:
        datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        # For simplicity, omnisource is not compatible with val workflow,
        # we recommend you to use `--validate`
        assert not cfg.omnisource
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    test_option = dict(test_last=False, test_best=False)
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        test=test_option,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
