[English](./README_EN.md) | 简体中文

# autox_nlp是什么

# 快速上手

# 目录
<!-- TOC -->

- [autox_nlp是什么](#autox_nlp是什么)
- [快速上手](#快速上手)
- [目录](#目录)
- [效果对比](#效果对比)

<!-- /TOC -->

# 效果对比
| Task type      | Dataset name         | Evaluation Metric | AutoX                                                                                                      | AutoGluon                                                                     | H2o                                                                         |
|----------------|----------------------|-------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Regression     | CommonlitReadability | RMSE              | 0.594                                                                                                      | [0.804](https://www.kaggle.com/code/hengwdai/commonlit-readability-autogluon) | [0.998](https://www.kaggle.com/code/hengwdai/commonlit-readability-h2o)     |
| Regression     | Amazonbookprice      | RMSE              | [0.622](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/amazon-book-price/autox.ipynb) | [0.697](https://www.kaggle.com/hengwdai/amazon-book-price-autogluon)          | [0.705](https://www.kaggle.com/code/hengwdai/amazon-book-price-h2o/)        |
| Regression     | MercariPrice         | RMSE              | 25.900                                                                                                     | [34.500](https://www.kaggle.com/code/hengwdai/mercariprice-autogluon)         | [32.910](https://www.kaggle.com/code/hengwdai/mercariprice-h2o)             |
| Classification | Titanic              | AUC               | 0.794                                                                                                      | [0.780](https://www.kaggle.com/code/sishihara/autogluon-tabular-for-titanic)  | [0.768](https://www.kaggle.com/code/hengwdai/titanic-solution-with-basic-h2o) |
| Classification | Stumbleupon          | AUC               | 0.871                                                                                                      | [0.810](https://github.com/4paradigm/AutoX/blob/master/demo/stumbleupon/autogluon_stumbleupon.ipynb)                                                                     | [0.790](https://github.com/4paradigm/AutoX/blob/master/demo/stumbleupon/h2o_kaggle_stumbleupon.ipynb)                                                                   |
| Classification | DisasterTweets       | AUC               | 0.786                                                                                                      | [0.779](https://www.kaggle.com/hengwdai/tweeter-autogluon)                    | [0.721](https://www.kaggle.com/code/hengwdai/tweeter-h2o)                   |

# 处理效率对比

| Dataset              | Text Column     | Average Text Length | TPS    | AutoX   | AutoGluon | H2O     |
|----------------------|-----------------|---------------------|--------|---------|-----------|---------|
| MercariPrice         | BrandName       | 6                   | item/s | 4736.42 | 141.06    | 1940.45 |
| MercariPrice         | CategoryName    | 30                  | item/s | 2661.58 | 143.82    | 1420.02 |
| MercariPrice         | ItemDescription | 150                 | item/s | 1762.78 | 125.41    | 353.14  |
| TMDBBoxOffice        | Overview        | 300                 | item/s | 462.96  | 20.40     | 194.55  |
| CommonlitReadability | Excerpt         | 1000                | item/s | 179.15  | 18.59     | 58.46   |
