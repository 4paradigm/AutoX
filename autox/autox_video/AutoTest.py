# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from tools.GetPipeline import get_dataset_cfg
import mmcv
import torch
from mmcv import Config
from mmcv.fileio.io import file_handlers
from mmcv.runner import get_dist_info, init_dist
from tools.Inference import turn_off_pretrained, inference_pytorch
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.utils import (build_ddp, build_dp, default_device,
                            register_module_hooks, setup_multi_processes)

try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
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
    with open('config.yaml','r') as cus_cfg:
        cus_cfg = yaml.load(cus_cfg, Loader=yaml.FullLoader)
    cfg.model.cls_head.num_classes = cus_cfg['num_class']
    cfg.work_dir = cus_cfg['work_dir']
    cfg.data = get_dataset_cfg(
        data_root=cus_cfg['data_root'],
        ann_file_train=cus_cfg['ann_file_train'],
        ann_file_val=cus_cfg['ann_file_val'],
        ann_file_test=cus_cfg['ann_file_test'],
        videos_per_gpu=8,
        video_length=cus_cfg['video_length'])
    work_dir = cus_cfg['work_dir']
    checkpoints = None
    for file in os.listdir(work_dir):
        if file.startswith('best') and file.endswith('pth'):
            checkpoints = os.path.join(work_dir, file)
    if checkpoints is None:
        checkpoints = os.path.join(work_dir, 'latest.pth')
    assert checkpoints is not None and os.path.exists(checkpoints), \
        'checkpoints not found in work dir %s, please check the work dir in config.yaml'%work_dir

    # set multi-process settings
    setup_multi_processes(cfg)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    output_config = Config._merge_a_into_b(
        dict(out=cus_cfg['test_results_path']), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    eval_config = Config._merge_a_into_b(
        dict(metrics='top_k_accuracy'), eval_config)

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    outputs = inference_pytorch(checkpoints, cfg, distributed, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        if output_config.get('out', None):
            out = output_config['out']
            print(f'\nwriting results to {out}')
            dataset.dump_results(outputs, **output_config)
        if eval_config:
            eval_res = dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
