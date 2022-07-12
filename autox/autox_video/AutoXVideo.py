import copy
import argparse
from .tools.GetPipeline import get_dataset_cfg
from .tools.Inference import inference_pytorch
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import yaml
import pickle
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmcv.fileio.io import file_handlers

from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset, build_dataloader
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger,
                            register_module_hooks, setup_multi_processes)


class AutoXVideo():
    default_cfg = os.path.join(os.path.dirname(__file__), 'mmaction2/configs/TASK/swin.py')
    def __init__(self):
        self.cfg = Config.fromfile(self.default_cfg)
        if not os.path.exists(self.cfg.load_from):
            self.cfg.load_from = None

        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # The flag is used to determine whether it is omnisource training
        self.cfg.setdefault('omnisource', False)

        # The flag is used to register module's hooks
        self.cfg.setdefault('module_hooks', [])

        # default setting
        self.cfg.total_epochs = 50
        self.cfg.work_dir = 'work_dirs/default_work_dir'

        self.cfg.num_class = -1
        self.cfg.video_length = 2

    def read_cfg(self, cfg_file):
        with open(cfg_file, 'r') as cus_cfg:
            cus_cfg = yaml.load(cus_cfg, Loader=yaml.FullLoader)
        if 'epoch' in cus_cfg.keys():
            self.cfg.total_epochs = cus_cfg['epoch']
        if 'work_dir' in cus_cfg.keys():
            self.cfg.work_dir = cus_cfg['work_dir']
            resume_path = os.path.join(cus_cfg['work_dir'], 'latest.pth')
            if os.path.exists(os.path.join(cus_cfg['work_dir'], 'latest.pth')):
                self.cfg.resume_from = resume_path
        if 'data_root' in cus_cfg.keys():
            self.cfg.model.cls_head.num_classes = cus_cfg['num_class']
            self.cfg.num_class = cus_cfg['num_class']
            self.cfg.video_length=cus_cfg['video_length']
            self.cfg.data = get_dataset_cfg(
                data_root=cus_cfg['data_root'],
                ann_file_train=cus_cfg.get('ann_file_train', None),
                ann_file_val=cus_cfg.get('ann_file_val', None),
                ann_file_test=cus_cfg.get('ann_file_test', None),
                videos_per_gpu=cus_cfg.get('videos_per_gpu', 8),
                video_length=cus_cfg.get('video_length', 2))

    def fit(self,
            data_root=None,
            ann_file_train=None,
            ann_file_val=None,
            video_length=-1,
            num_class=-1,
            epoch=-1,
            videos_per_gpu=-1,
            evaluation=5,
            work_dir=None,
            distributed=False,
            gpus=1):
        if data_root is not None:
            # specify custom dataset
            assert ann_file_train is not None and video_length != -1 and num_class != -1,\
                'Yon need to specify all dataset config by params or read_cfg().'
            self.cfg.model.cls_head.num_classes = num_class
            if ann_file_val != None:
                flag_val = True
            else:
                flag_val = False
                ann_file_val = ann_file_train
            if self.cfg.num_class == -1 and videos_per_gpu == -1:
                videos_per_gpu = 1
            elif self.cfg.num_class != -1 and videos_per_gpu == -1:
                videos_per_gpu = self.cfg.data.videos_per_gpu
            self.cfg.data = get_dataset_cfg(
                data_root=data_root,
                ann_file_train=ann_file_train,
                ann_file_val=ann_file_val,
                ann_file_test=(self.cfg.get('data', {})).get('ann_file_test', None),
                videos_per_gpu=videos_per_gpu,
                video_length=video_length)

        if epoch != -1:
            self.cfg.total_epochs = epoch

        if work_dir is not None:
            self.cfg.work_dir = work_dir

        assert self.cfg.work_dir is not None, 'work dir not specified.'

        resume_path = os.path.join(self.cfg.work_dir, 'latest.pth')
        if os.path.exists(os.path.join(self.cfg.work_dir, 'latest.pth')):
            self.cfg.resume_from = resume_path

        if evaluation == 0:
            flag_val = False
        else:
            self.cfg.evaluation.interval = evaluation
            flag_val = True

        # set multi-process settings
        setup_multi_processes(self.cfg)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
        # init logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=self.cfg.log_level)

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
        logger.info(f'Config: {self.cfg.pretty_text}')

        # set random seeds
        seed = init_random_seed(42, distributed=distributed)
        logger.info(f'Set random seed to {seed}')
        set_random_seed(seed, deterministic=False)

        self.cfg.seed = seed
        meta['seed'] = seed
        meta['work_dir'] = osp.basename(self.cfg.work_dir.rstrip('/\\'))

        model = build_model(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg'))

        if len(self.cfg.module_hooks) > 0:
            register_module_hooks(model, self.cfg.module_hooks)

        if self.cfg.omnisource:
            # If omnisource flag is set, cfg.data.train should be a list
            assert isinstance(self.cfg.data.train, list)
            datasets = [build_dataset(dataset) for dataset in self.cfg.data.train]
        else:
            datasets = [build_dataset(self.cfg.data.train)]

        if len(self.cfg.workflow) == 2:
            # For simplicity, omnisource is not compatible with val workflow,
            # we recommend you to use `--validate`
            assert not self.cfg.omnisource
            val_dataset = copy.deepcopy(self.cfg.data.val)
            datasets.append(build_dataset(val_dataset))
        if self.cfg.checkpoint_config is not None:
            # save mmaction version, config file content and class names in
            # checkpoints as meta data
            self.cfg.checkpoint_config.meta = dict(
                mmaction_version=__version__ + get_git_hash(digits=7),
                config=self.cfg.pretty_text)

        with open(os.path.join(self.cfg.work_dir, 'cfg.pkl'),'wb') as file_pkl:
            pickle.dump(self.cfg, file_pkl)

        test_option = dict(test_last=False, test_best=False)
        train_model(
            model,
            datasets,
            self.cfg,
            distributed=distributed,
            validate=flag_val,
            test=test_option,
            timestamp=timestamp,
            meta=meta)

    def transform(self,
                  data_root=None, ann_file_test=None, video_length=-1,
                  checkpoints=None,
                  output_path='results.json',
                  distributed=False, gpus=1):
        if checkpoints is not None:
            assert os.path.exists(checkpoints), '%s not found.' % checkpoints
            with open(os.path.join(os.path.dirname(checkpoints), 'cfg.pkl'), 'rb') as file_pkl:
                cfg = pickle.load(file_pkl)
        else:
            cfg = self.cfg
            assert os.path.exists(cfg.work_dir), 'checkpoints not specified.'
            for file in os.listdir(cfg.work_dir):
                if file.startswith('best') and file.endswith('pth'):
                    checkpoints = os.path.join(cfg.work_dir, file)
            if checkpoints is None:
                checkpoints = os.path.join(cfg.work_dir, 'latest.pth')
            assert checkpoints is not None and os.path.exists(checkpoints), \
                'checkpoints not found in work dir %s. ' % cfg.work_dir

        if data_root is not None:
            if ann_file_test is None:
                warnings.warn('No annotations provided, top_k_accuracy is invalid.')
                with open(os.path.join(data_root, 'temp_list.txt'), 'w') as file_temp:
                    for video in os.listdir(data_root):
                        if not os.path.isfile(os.path.join(data_root, video)) or video.endswith('txt'):
                            continue
                        file_temp.write('%s 0\n'%video)
                ann_file_test = os.path.join(data_root, 'temp_list.txt')
            if video_length == -1:
                video_length = self.cfg.video_length
            cfg.data = get_dataset_cfg(
                data_root=data_root,
                ann_file_test=ann_file_test,
                videos_per_gpu=1,
                video_length=video_length)
        else:
            cfg.data = self.cfg.data

        # set multi-process settings
        setup_multi_processes(cfg)

        # Load output_config from cfg
        output_config = cfg.get('output_config', {})
        output_config = Config._merge_a_into_b(
            dict(out=output_path), output_config)

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
        if os.path.exists(os.path.join(data_root, 'temp_list.txt')):
            os.remove(os.path.join(data_root, 'temp_list.txt'))

        return outputs

