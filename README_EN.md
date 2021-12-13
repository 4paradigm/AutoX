English | [简体中文](./README.md)

# What-is-AutoX?
AutoX is an efficient automl tool, mainly aimed at data mining competitions with tabular data.
Its features include:
- SOTA: AutoX outperforms other solutions in many competition datasets(see [Evaluation](#Evaluation)).
- Easy to use: The design of interfaces is similar to sklearn.
- Generic & Universal: Supporting tabular data, including binary classification, multi-class classification and regression problems.
- Auto: Fully automated pipeline without human-intervention.
- Out of the box: Providing flexible modules which can be used alone.
- Summary of magics: Organize and publish magics of competitions.

## interpretable-ml
AutoX covers following interpretable machine learning methods:
### Golbel interpretation
- [tree-based model](autox/interpreter/interpreter_demo/global_interpretation/global_surrogate_tree_demo.ipynb)

### Local interpretation
- [LIME](autox/interpreter/interpreter_demo/local_interpretation/lime_demo.ipynb)
- [SHAP](autox/interpreter/interpreter_demo/local_interpretation/shap_demo.ipynb)

### Influential interpretation
- [nn](autox/interpreter/interpreter_demo/influential_instances/influential_interpretation_nn.ipynb)
- [nn_sgd](autox/interpreter/interpreter_demo/influential_instances/influential_interpretation_nn_sgd.ipynb)

### Prototypes and Criticisms
- [MMD-critic](autox/interpreter/interpreter_demo/prototypes_and_criticisms/MMD_demo.ipynb)
- [ProtoDash algorithm](autox/interpreter/interpreter_demo/prototypes_and_criticisms/ProtodashExplainer.ipynb)


# Table-of-Contents
<!-- TOC -->

- [What is AutoX?](#What-is-AutoX?)
- [Table of Contents](#Table-of-Contents)
- [Installation](#Installation)
- [Architecture](#Architecture)
- [Quick Start](#Quick-Start)
- [Summary of Magics](#Summary-of-Magics)
- [Evaluation](#Evaluation)

<!-- /TOC -->
# Installation
```
1. git clone https://github.com/4paradigm/autox.git
2. cd autox
3. python setup.py install
```

# Architecture
```
├── autox
│   ├── ensemble
│   ├── feature_engineer
│   ├── feature_selection
│   ├── file_io
│   ├── join_tables
│   ├── metrics
│   ├── models
│   ├── process_data
│   └── util.py
│   ├── CONST.py
│   ├── autox.py
├── run_oneclick.py
└── demo
└── test
├── setup.py
├── README.md
```

# Quick-Start
- Full-Automl
```
from autox import AutoX
path = data_dir
autox = AutoX(target = 'loss', train_name = 'train.csv', test_name = 'test.csv', 
               id = ['id'], path = path)
sub = autox.get_submit()
sub.to_csv("submission.csv", index = False)
```
- Semi-Automl: run_demo.ipynb

# Evaluation
| index |data_type | data_name(link)  | metric | AutoX         | AutoGluon   | H2o |
| ----- |----- | ------------- | ----------- |---------------- | ----------------|----------------|
| 1    |regression | [zhidemai](https://www.automl.ai/competitions/19)   | mse | 1.1231 | 1.9466 | 1.1927|
| 2    |regression | [Tabular Playground Series - Aug 2021](https://www.kaggle.com/c/tabular-playground-series-aug-2021)   | rmse | 7.87731 | 10.3944 | 7.8895|
| 3    |regression | [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/)   | rmse | 0.13043 | 0.13104 | 0.13161 |
| 4    |binary classification | [Titanic](https://www.kaggle.com/c/titanic/)  | accuracy | 0.77751 | 0.78229 | 0.79186 |

# Data type
- cat: Categorical, Categorical variable without order.
- ord: Ordinal, Categorical variable with order.
- num: Numeric, Numeric variable.
- datetime: Time variable with Datetime format.
- timestamp: Time variable with Timestamp format.

# Pipeline
- 1.Initialize AutoX
```
1.1 Read data
1.2 Concat train and test
1.3 Identify columns type in data
1.4 Data preprocess
```
- 2.Feature engineer
```
Every feature engineer class inclues the following features:
1. auto select columns which will be executed with current operation
2. review the selected columns
3. modify the columns
4. execute the operation, and return features whose samples' number and order are consistent with orginal table.
```
- 3.Features combination
```
Combine the raw features and derived features, and return wide table.
```
- 4.train_test_split
```
Split the wide table into train and test.
```
- 5.Features filter
```
Filter the features according to the distribution of train and test.
```
- 6.Model training
```
Inputs of models are filtered features. 
model class inclues the following features:
1. get the default parameters
2. model training
3. parameters tuning
4. get the features importance
5. prediction
```
- 7.Prediction

# AutoX
## Attributes
###  info_: Information about the data set.
- info_['id']: List, unique keys to identify the sample.
- info_['target']: String, label column.
- info_['shape_of_train']: Int, the number of samples in the train set.
- info_['shape_of_test']: Int, the number of samples in the test set.
- info_['feature_type']: Dict of Dict, data type of the features.
- info_['train_name']: String, the table name of main table of train.
- info_['test_name']: String, the table name of main table of test.

### dfs_: dfs_ contains all DataFrames, including raw tables and derived tables.
- dfs_['train_test']: The combined data of train data and test data.
- dfs_['FE_feature_name']: Derived tables by feature engineering, such as FE_count, FE_groupby.
- dfs_['FE_all']: The merged table which contains raw tables and derived tables.

## Methods
- concat_train_test: concat the train and test data.
- split_train_test: split train and test data.
- get_submit: get the submission.

# Details of operations in the pipeline:
## Data IO
```
```

## Data Pre-process
```
- extract year, month, day, hour, weekday info from time columns
- delete invalid(nunique equal to 1) features 
- delete invalid (label is nan) samples
```

## Feature Engineer

- count feature
```
```

- target encoding feature


- shift feature
```
```

## Model Fitting
```
AutoX supports fellowing models:
1. Lightgbm
2. Xgboost
3. Tabnet
```

## Ensemble
```
AutoX supports two ensemble methods(Bagging will be used in default). 
1. Stacking；
2. Bagging。
```

# Summary-of-Magics
|competition|magics|
|------|------|
|kaggle criteo||
|zhidemai||

## Debug
|Log|Solution|
|------|------|

