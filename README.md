[English](./README_EN.md) | 简体中文
<img src="./img/logo.png" width = "1500" alt="logo" align=center />

# AutoX是什么？
AutoX一个高效的自动化机器学习工具。
它的特点包括:
- 效果出色: AutoX在多个kaggle数据集上，效果显著优于其他解决方案(见[效果对比](#效果对比))。
- 简单易用: AutoX的接口和sklearn类似，方便上手使用。
- 通用: 适用于分类和回归问题。
- 自动化: 无需人工干预，全自动的数据清洗、特征工程、模型调参等步骤。
- 灵活性: 各组件解耦合，能单独使用，对于自动机器学习效果不满意的地方，可以结合专家知识，AutoX提供灵活的接口。
- 比赛上分点总结：整理并公开历史比赛的上分点。

# AutoX包含什么内容
- [autox_competition](autox/autox_competition/README.md): 主要针对于表格类型的数据挖掘竞赛
- [autox_server](autox/autox_server/README.md): 用于上线部署的automl服务
- [autox_interpreter](autox/autox_interpreter/README.md): 机器学习可解释功能

# 加入社区
<img src="./img/qr_code_0506.png" width = "200" height = "200" alt="AutoX社区" align=center />

# 如何为AutoX贡献
[如何为AutoX贡献](./how_to_contribute.md)

# 目录
<!-- TOC -->

- [AutoX是什么？](#AutoX是什么？)
- [AutoX包含什么内容](#AutoX包含什么内容)
- [加入社区](#加入社区)
- [目录](#目录)
- [安装](#安装)
- [如何为AutoX贡献](#如何为AutoX贡献)
- [快速上手](#快速上手)
- [效果对比](#效果对比)
- [TODO](#TODO)
- [错误排查](#错误排查)

<!-- /TOC -->
# 安装

### github仓库安装
```
git clone https://github.com/4paradigm/autox.git
## github访问速度较慢时可以通过gitee地址 https://gitee.com/poteman/autox
pip install ./autox
```

### pip安装
```
## pip安装包可能更新不及时，建议用github安装方式安装最新版本
!pip install automl-x -i https://www.pypi.org/simple/
```

# 快速上手
- [autox打比赛](autox/autox_competition/README.md)
- [autox上线部署](autox/autox_server/README.md)
- [autox可解释](autox/autox_interpreter/README.md)
- [特征工程](autox/autox_competition/feature_engineer/README.md)

# 社区案例
[汽车销量预测](./demo/汽车销量预测/README.md)

# 比赛案例
见demo文件夹

数据集下载链接：https://pan.baidu.com/s/1p38OuP8_FJp2P_wJwhdFiw?pwd=8mxf
# 效果对比
## 不同任务下的效果提升百分比
|data_type | 对比AutoGluon | 对比H2o |
|----- | ------------- | ----------- |
|binary classification | 20.44% | 2.98% |
|regression | 37.54% | 39.66% |
|time-series | 28.40% | 32.46% |

## 详细数据集对比
|data_type | single-or-multi | data_name | metric | AutoX | AutoGluon | H2o |
|----- | ------------- | ----------- |---------------- |---------------- | ----------------|----------------|
|binary classification | single-table | [Springleaf](https://www.kaggle.com/c/springleaf-marketing-response/)  | auc | 0.78865 | 0.61141 | 0.78186 |
|binary classification-nlp | single-table |[stumbleupon](https://www.kaggle.com/c/stumbleupon/)  | auc | 0.87177 | 0.81025 | 0.79039 |
|binary classification | single-table |[santander](https://www.kaggle.com/c/santander-customer-transaction-prediction/)  | auc | 0.89196 | 0.64643 | 0.88775 |
|binary classification | multi-table |[IEEE](https://www.kaggle.com/c/ieee-fraud-detection/)  | accuracy | 0.920809 | 0.724925 | 0.907818 |
|regression | single-table |[ventilator](https://www.kaggle.com/c/ventilator-pressure-prediction/)  | mae | 0.755 | 8.434 | 4.221 |
|regression | single-table |[Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity)| mae | 1137.07885 | 1173.35917 | 1163.12014 |
|regression | single-table |[zhidemai](https://www.automl.ai/competitions/19)   | mse | 1.0034 | 1.9466 | 1.1927|
|regression | single-table |[Tabular Playground Series - Aug 2021](https://www.kaggle.com/c/tabular-playground-series-aug-2021) | rmse | 7.87731 | 10.3944 | 7.8895|
|regression | single-table |[House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/)  | rmse | 0.13043 | 0.13104 | 0.13161 |
|regression | single-table |[Restaurant Revenue](https://www.kaggle.com/c/restaurant-revenue-prediction/)| rmse | 2133204.32146 | 31913829.59876 | 28958013.69639 |
|regression | multi-table  |[Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation/)| rmse | 3.72228 | 3.80801 | 22.88899 |
|regression-ts | single-table  |[Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only/)| smape | 13.79241 | 25.39182 | 18.89678 |
|regression-ts | multi-table  |[Walmart Recruiting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/)| wmae | 4660.99174 | 5024.16179 | 5128.31622 |
|regression-ts | multi-table  |[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/)| RMSPE | 0.13850 | 0.20453 | 0.35757 |
|regression-cv | single-table |[PetFinder](https://www.kaggle.com/competitions/petfinder-pawpularity-score/overview/)  | rmse | 20.1327 | 23.1732 | 21.0586 |

# AutoX成就
### 企业支持

### 比赛获奖
- [2021阿里云基础设施供应链大赛-冠军方案](https://tianchi.aliyun.com/forum/postDetail?postId=344505)


# TODO
功能开发完成后，发布相应的使用demo
- [ ] 多分类任务

若有其他希望AutoX支持的功能，欢迎提issue!
欢迎填写[用户调研问卷](https://www.wjx.cn/vj/YOwSFHN.aspx)，让AutoX变得更好!

## 错误排查
|错误信息|解决办法|
|------|------|
