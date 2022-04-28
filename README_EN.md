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

# How-to-contribute-for-AutoX
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

<!-- /TOC -->
# Installation
```
1. git clone https://github.com/4paradigm/autox.git
2. cd autox
3. python setup.py install
```

# Quick-Start
- [autox competition](autox/autox_competition/README_EN.md)
- [autox server](autox/autox_server/README_EN.md)
- [autox interpreter](autox/autox_interpreter/README_EN.md)


# Evaluation
| index |data_type | data_name(link)  | metric | AutoX         | AutoGluon   | H2o |
| ----- |----- | ------------- | ----------- |---------------- | ----------------|----------------|
| 1    |regression | [zhidemai](https://www.automl.ai/competitions/19)   | mse | 1.1231 | 1.9466 | 1.1927|
| 2    |regression | [Tabular Playground Series - Aug 2021](https://www.kaggle.com/c/tabular-playground-series-aug-2021)   | rmse | 7.87731 | 10.3944 | 7.8895|
| 3    |regression | [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/)   | rmse | 0.13043 | 0.13104 | 0.13161 |
| 4    |binary classification | [Titanic](https://www.kaggle.com/c/titanic/)  | accuracy | 0.77751 | 0.78229 | 0.79186 |
