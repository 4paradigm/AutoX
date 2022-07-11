English | [简体中文](./README.md)

## What is AutoX_video

autox_video provides an automatic machine learning framework for video understanding based on the mmaction2 codebase, and you can conveniently train video understanding tasks with simple commands.

[![framework](https://github.com/4paradigm/AutoX/raw/master/autox/autox_video/resources/framework.png)](https://github-com.translate.goog/4paradigm/AutoX/blob/master/autox/autox_video/resources/framework.png?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp)

## Table of contents

[Install](https://github-com.translate.goog/4paradigm/AutoX/tree/master/autox/autox_video?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#安装)

[quick start](https://github-com.translate.goog/4paradigm/AutoX/tree/master/autox/autox_video?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#快速开始)

[Pretrained weights](https://github-com.translate.goog/4paradigm/AutoX/tree/master/autox/autox_video?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#预训练权重)

[Show results](https://github-com.translate.goog/4paradigm/AutoX/tree/master/autox/autox_video?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#效果展示)

[Follow-up](https://github-com.translate.goog/4paradigm/AutoX/tree/master/autox/autox_video?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#后续工作)

## 

## Installation

#### Dependencies

1. Python 3.6+
2. PyTorch 1.3+
3. CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
4. GCC 5+
5. mmcv 1.1.1+
6. Numpy
7. ffmpeg (4.2 is preferred)
8. decorated (optional, 0.4.1+): Install CPU version by pip install decorated==0.4.1 and install GPU version from source

### Pytorch

Install PyTorch and torchvision according to the [official documentation , such as:](https://translate.google.com/website?sl=zh-CN&tl=en&hl=zh-CN&client=webapp&u=https://pytorch.org)

```
conda install pytorch torchvision -c pytorch
```

Make sure the build version of CUDA matches the run version of CUDA. Users can refer to the [PyTorch official website](https://translate.google.com/website?sl=zh-CN&tl=en&hl=zh-CN&client=webapp&u=https://pytorch.org) to check the CUDA version supported by the precompiled package.

### MMCV

To install mmcv-full, we recommend that you install the following prebuilt packages:

```
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

Alternatively, users can compile from source by using the following command:

```
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

Or run the script directly:

```
pip install mmcv-full
```

### other

```
pip install -r build.txt
python mmaction2/setup.py develop
```

## quick start

We provide a demo dataset as a test, you can run your own dataset in the same way.

### train

```
python AutoTrain.py
```

This starts the training of the model, we automatically save the training results of the model, and at the end of every two epochs, evaluate the model on the validation set and store the optimal weights.

In case of an unexpected interruption, re-executing this command, we will restore the previous training results instead of starting over (unless you have changed the location of the working directory).

You can train and test the model on your own dataset by modifying the dataset settings in [config.yaml .](https://github-com.translate.goog/4paradigm/AutoX/blob/master/autox/autox_video/config.yaml?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp)

### test

```
python AutoTest.py
```

This will automatically read the optimal weights stored in the working directory, use it to test the model on the test set, and output the inference results to the location specified in [config.yaml](https://github-com.translate.goog/4paradigm/AutoX/blob/master/autox/autox_video/config.yaml?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp) (results.json by default).

## Pretrained weights

The pre-trained weights used by the model can be downloaded from the link below. After downloading, store the weight files in the checkpoints directory, and the pre-trained weights will be used automatically to start training during training
(the pre-trained weight files are provided by [Video-Swin-Transformer](https://github-com.translate.goog/SwinTransformer/Video-Swin-Transformer?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp) )

| Backbone | Pretrain                  | Lr Schd | spatial crops | acc@1 | acc@5 | #params | FLOPs  | model                                                        |
| -------- | ------------------------- | ------- | ------------- | ----- | ----- | ------- | ------ | ------------------------------------------------------------ |
| Swin-B   | ImageNet22k & Kinetics600 | 30ep    | 224           | 84.0  | 96.5  | 88M     | 281.6G | [github](https://github-com.translate.goog/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp) / [baidu](https://translate.google.com/website?sl=zh-CN&tl=en&hl=zh-CN&client=webapp&u=https://pan.baidu.com/s/1ZMeW6ylELTje-o3MiaZ-MQ) |

## Show results

We took first place in the video classification track of the ACM MM 22 PRE-TRAINING FOR VIDEO UNDERSTANDING CHALLENGE competition[![leaderboard](https://github.com/4paradigm/AutoX/raw/master/autox/autox_video/resources/leaderboard.jpeg)](https://github-com.translate.goog/4paradigm/AutoX/blob/master/autox/autox_video/resources/leaderboard.jpeg?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp)

Test on public datasets:

| Dataset                                                      | Top 1 Accuracy |
| ------------------------------------------------------------ | -------------- |
| [HMDB51](https://translate.google.com/website?sl=zh-CN&tl=en&hl=zh-CN&client=webapp&u=https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) | 0.5902         |
| [UCF101](https://translate.google.com/website?sl=zh-CN&tl=en&hl=zh-CN&client=webapp&u=https://www.crcv.ucf.edu/research/data-sets/ucf101/) | 0.9407         |

## Follow-up

1. At present, the code only supports Video Swin Transformer, a backbone, which is the best and more general model in our experiments. More video understanding models will be added in the future for users to choose freely.
2. Currently, only video classification is supported. In fact, this framework is general for tasks such as video object detection and video semantic segmentation, and interfaces for other video tasks will be developed in the future.
