
This is an automatic machine learning codebase for video understanding tasks (temporally only support categorization)
## Get Started
### Installation
#### Requirements
1. Python 3.6+
2. PyTorch 1.3+
3. CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
4. GCC 5+
5. mmcv 1.1.1+
6. Numpy
7. ffmpeg (4.2 is preferred)
8. decord (optional, 0.4.1+): Install CPU version by pip install decord==0.4.1 and install GPU version from source

#### Pytorch
Install PyTorch and torchvision following the official instructions, e.g.,
```
conda install pytorch torchvision -c pytorch
```
Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the PyTorch website.
#### MMCV
Install mmcv-full, we recommend you to install the pre-built package as below.
```
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```
Optionally you can choose to compile mmcv from source by the following command
```
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```
or directly run
```
pip install mmcv-full
```
#### Others

```
pip install -r build.txt
python mmaction2/setup.py develop
```

### Demo
We offer a small dataset for code test. You can run your own dataset in the same way if it is bug-free on this demo.
#### Train
```
python AutoTrain.py
```
#### Test
```
python AutoTest.py
```

#### Shift to your own dataset
You can edit the config.yaml to shift to your own dataset without any other changes. You need to organize your own data following the demo data.
