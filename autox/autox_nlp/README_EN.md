English | [简体中文](./README.md)

# What-is-AutoX_nlp?


# Table-of-Contents
<!-- TOC -->

- [What is AutoX_nlp?](#What-is-AutoX_nlp?)
- [Quick Start](#Quick-Start)
- [Evaluation](#Evaluation)
- [Efficiency](#Efficiency)

<!-- /TOC -->

# Quick-Start

# Evaluation
| Task type      | Dataset name                                                                             | Evaluation Metric | AutoX                                                                       | AutoGluon                                                                     | H2o                                                                           |
|----------------|------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Regression     | [CommonlitReadability](https://www.kaggle.com/hengwdai/commonlit-readability-data-split) | RMSE              | [0.597](https://www.kaggle.com/code/hengwdai/commonlit-readability-auto3ml) | [1.022](https://www.kaggle.com/code/hengwdai/commonlit-readability-autogluon) | [1.023](https://www.kaggle.com/code/hengwdai/commonlit-readability-h2o)       |
| Regression     | [Amazonbookprice](https://www.kaggle.com/hengwdai/amazon-book-price-data-split)          | RMSE              | [629.792](https://www.kaggle.com/code/hengwdai/amazon-book-price-auto3ml)   | [687.870](https://www.kaggle.com/hengwdai/amazon-book-price-autogluon)        | [642.167](https://www.kaggle.com/code/hengwdai/amazon-book-price-h2o/)        |
| Regression     | [MercariPrice](https://www.kaggle.com/hengwdai/mercariprice-data-split)                  | RMSE              | [32.042](https://www.kaggle.com/code/hengwdai/mercariprice-auto3ml)         | [34.500](https://www.kaggle.com/code/hengwdai/mercariprice-autogluon)         | [43.960](https://www.kaggle.com/code/hengwdai/mercariprice-h2o)               |
| Classification | [Titanic](https://www.kaggle.com/competitions/titanic/data)                              | AUC               | [0.794](https://www.kaggle.com/code/hengwdai/autox-titanic)                 | [0.780](https://www.kaggle.com/code/sishihara/autogluon-tabular-for-titanic)  | [0.768](https://www.kaggle.com/code/hengwdai/titanic-solution-with-basic-h2o) |
| Classification | [Stumbleupon](https://www.kaggle.com/hengwdai/stumbleupon-data-split)                    | AUC               | [0.855](https://www.kaggle.com/code/hengwdai/stumbleupon-auto3ml)           | [0.503](https://www.kaggle.com/code/hengwdai/stumbleupon-autogluon)           | [0.707](https://www.kaggle.com/code/hengwdai/stumbleupon-h2o)                 |
| Classification | [DisasterTweets](https://www.kaggle.com/competitions/nlp-getting-started/data)           | AUC               | [0.786](https://www.kaggle.com/code/hengwdai/tweeter-autox)                 | [0.746](https://www.kaggle.com/hengwdai/tweeter-autogluon)                    | [0.721](https://www.kaggle.com/code/hengwdai/tweeter-h2o)                     |

# Efficiency

| Dataset              | Text Column     | Average Text Length | TPS    | AutoX                                                                               | AutoGluon                                                                               | H2O                                                                                    |
|----------------------|-----------------|---------------------|--------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| MercariPrice         | BrandName       | 6                   | item/s | [3480.66](https://www.kaggle.com/hengwdai/mercariprice-6-efficiency-auto3ml)        | [127.15](https://www.kaggle.com/hengwdai/mercariprice-6-efficiency-autogluon)           | [979.18](https://www.kaggle.com/hengwdai/mercariprice-6-efficiency-h2o)                |
| MercariPrice         | CategoryName    | 30                  | item/s | [2215.40](https://www.kaggle.com/hengwdai/mercariprice-30-efficiency-auto3ml)       | [118.92](https://www.kaggle.com/hengwdai/mercariprice-30-efficiency-autogluon)          | [656.80](https://www.kaggle.com/code/hengwdai/mercariprice-30-efficiency-h2o)          |
| MercariPrice         | ItemDescription | 150                 | item/s | [466.73](https://www.kaggle.com/hengwdai/mercariprice-150-efficiency-auto3ml)       | [65.46](https://www.kaggle.com/hengwdai/mercariprice-150-efficiency-autogluon)          | [183.14](https://www.kaggle.com/hengwdai/mercariprice-150-efficiency-h2o)              |
| TMDBBoxOffice        | Overview        | 300                 | item/s | [282.73](https://www.kaggle.com/code/hengwdai/tmdbboxoffice-300-efficiency-auto3ml) | [20.74](https://www.kaggle.com/code/hengwdai/tmdbboxoffice-300-efficiency-autogluon)    | [79.18](https://www.kaggle.com/hengwdai/tmdbboxoffice-300-efficiency-h2o)              |
| CommonlitReadability | Excerpt         | 1000                | item/s | [103.99](https://www.kaggle.com/hengwdai/commonlitreadability-1000-efficiency)      | [12.39](https://www.kaggle.com/hengwdai/commonlitreadability-1000-efficiency-autogluon) | [30.30](https://www.kaggle.com/code/hengwdai/commonlitreadability-1000-efficiency-h2o) |
