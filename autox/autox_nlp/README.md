[English](./README_EN.md) | 简体中文

# autox_nlp是什么

# 快速上手

## Demo(按数据类型划分)
#### 二分类问题
Kaggle_Santander-AutoX解决方案:
- [colab](https://colab.research.google.com/drive/1HKOr3vK_Ty3Dftw2JF4SJWFtwxsBfcLz?usp=sharing)
- [kaggle-kernel](https://www.kaggle.com/poteman/autox-tutorial-santander/)

#### 回归问题
DC租金预测-AutoX解决方案:
- [colab](https://colab.research.google.com/drive/1SxK_-_6oAE8OzDitXCy2JM29F9UE0Ujj?usp=sharing)
- [DClab](https://www.dclab.run/project_content.html?type=myproject&id=5393)

# 目录
<!-- TOC -->

- [autox_nlp是什么](#autox_nlp是什么)
- [快速上手](#快速上手)
- [目录](#目录)
- [效果对比](#效果对比)

<!-- /TOC -->

# 效果对比
|data_type | single-or-multi | data_name | metric | AutoX | AutoGluon | H2o |
|----- | ------------- | ----------- |---------------- |---------------- | ----------------|----------------|
|binary classification | single-table | [Springleaf](https://www.kaggle.com/c/springleaf-marketing-response/)  | auc | 0.78865 | 0.61141 | 0.78186 |
|binary classification | multi-table |[IEEE](https://www.kaggle.com/c/ieee-fraud-detection/)  | accuracy | 0.920809 | 0.724925 | 0.907818 |
|regression | single-table  |[Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only/)| smape | 13.79241 | 25.39182 | 18.89678 |
|regression | multi-table  |[Walmart Recruiting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/)| wmae | 4660.99174 | 5024.16179 | 5128.31622 |
