English | [简体中文](./README.md)
<img src="./img/logo.png" width = "1500" alt="logo" align=center />
# What-is-AutoX?
AutoX is an efficient AutoML tool, and it is designed for the tabular data modelling for real-world datasets.
Its features include:
- SOTA: AutoX outperforms other solutions in many competition datasets(see [Evaluation](#Evaluation)).
- Easy to use: The design of interfaces is similar to sklearn.
- Generic & Universal: Supporting tabular data, including binary classification, multi-class classification and regression problems.
- Auto: Fully automated pipeline without human-intervention.
- Out of the box: Providing flexible modules which can be used alone.
- Summary of magics: Organize and publish magics of competitions. 

# What-does-AutoX-contain?
- autox_competition: mainly for tabular table data mining competitions 
- autox_server: automl service for online deployment 
- autox_interpreter: machine learning interpretable function

# Join-the-community
<img src="./img/qr_code_0429.png" width = "200" height = "200" alt="AutoX Community" align=center />  

# How-to-contribute-to-AutoX?
[how to contribute](./how_to_contribute.md)

# Table-of-Contents
<!-- TOC -->

- [What is AutoX?](#What-is-AutoX?)
- [What does AutoX contain?](#What-does-AutoX-contain?)
- [Join-the-community](#Join-the-community)
- [How to contribute for AutoX](#How-to-contribute-for-AutoX)
- [Table of Contents](#Table-of-Contents)
- [Installation](#Installation)
- [Quick Start](#Quick-Start)
- [Evaluation](#Evaluation)
- [TODO](#TODO)
- [Troubleshooting](#错误排查)  

<!-- /TOC -->
# Installation  
### github repository installation
```
1. git clone https://github.com/4paradigm/autox.git
2. cd autox
3. python setup.py install
```
### pip install
```
## The pip installation package may not be updated in time. It is recommended to install the latest version using the github installation method.
!pip install automl-x -i https://www.pypi.org/simple/
```
# Quick-Start
- [autox competition](autox/autox_competition/README_EN.md)
- [autox server](autox/autox_server/README_EN.md)
- [autox interpreter](autox/autox_interpreter/README_EN.md)

# Community case
[Car sales forecast](./demo/汽车销量预测/README.md)

# Competition case
see demo folder

# Comparison to other AutoML frameworks
## Percentage improvement under different tasks
|data_type | Compare To AutoGluon | Compare To H2o |
|----- | ------------- | ----------- |
|binary classification | 20.44% | 2.98% |
|regression | 37.54% | 39.66% |
|time-series | 28.40% | 32.46% | 

# Evaluation
| index |data_type | data_name(link)  | metric | AutoX         | AutoGluon   | H2o |
| ----- |----- | ------------- | ----------- |---------------- | ----------------|----------------|
| 1    |regression | [zhidemai](https://www.automl.ai/competitions/19)   | mse | 1.1231 | 1.9466 | 1.1927|
| 2    |regression | [Tabular Playground Series - Aug 2021](https://www.kaggle.com/c/tabular-playground-series-aug-2021)   | rmse | 7.87731 | 10.3944 | 7.8895|
| 3    |regression | [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/)   | rmse | 0.13043 | 0.13104 | 0.13161 |
| 4    |binary classification | [Titanic](https://www.kaggle.com/c/titanic/)  | accuracy | 0.77751 | 0.78229 | 0.79186 |

## Detailed dataset comparison
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

# AutoX Achievements
### Enterprise support

### Competition winning
- [2021 Alibaba Cloud Infrastructure Supply Chain Competition - Champion Scheme](https://tianchi.aliyun.com/forum/postDetail?postId=344505)


# TODO
After the function development is completed, release the corresponding demo
- [ ] Multi-classification tasks

If there are other functions that you want AutoX to support, please submit an issue! 
Welcome to fill in the [user survey questionnaire](https://www.wjx.cn/vj/YOwSFHN.aspx) to make AutoX better!

## Troubleshooting
|error message|Solution|
|------|------|
