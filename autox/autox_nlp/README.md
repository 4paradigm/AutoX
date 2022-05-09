[English](./README_EN.md) | 简体中文

# autox_nlp是什么
AutoX_nlp 是针对文本数据进行处理的辅助工具。
它的特点包括：
- 效果出色：基于该工具与AutoX通用自动化建模的解决方案，在多个kaggle数据集上，效果显著优于其他解决方案(见[效果对比](#效果对比))。
- 高效处理：该工具在不同字符长度的文本数据处理上，速度显著优于其他AutoML的文本处理工具(见[处理效率对比](#处理效率对比))。
- 多方式提取：该工具支持TFIDF、Word2Vec、Glove、FastText、Bert 和 Zero-shot labeling 六种特征提取方式。
- 多样化特征：支持直接输出Embedding特征，也支持输出离散型、连续型特征。
# 快速上手

# 目录
<!-- TOC -->

- [autox_nlp是什么](#autox_nlp是什么)
- [快速上手](#快速上手)
- [目录](#目录)
- [效果对比](#效果对比)

<!-- /TOC -->

# 效果对比
| Task type      | Dataset name                                                                             | Evaluation Metric | AutoX                                                                                                              | AutoGluon                                                                     | H2o                                                                           |
|----------------|------------------------------------------------------------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Regression     | [CommonlitReadability](https://www.kaggle.com/hengwdai/commonlit-readability-data-split) | RMSE              | [0.597](https://www.kaggle.com/code/hengwdai/commonlit-readability-auto3ml)                                        | [1.022](https://www.kaggle.com/code/hengwdai/commonlit-readability-autogluon) | [1.023](https://www.kaggle.com/code/hengwdai/commonlit-readability-h2o)       |
| Regression     | [Amazonbookprice](https://www.kaggle.com/hengwdai/amazon-book-price-data-split)          | RMSE              | [629.792](https://www.kaggle.com/code/hengwdai/amazon-book-price-auto3ml)                                          | [687.870](https://www.kaggle.com/hengwdai/amazon-book-price-autogluon)        | [642.167](https://www.kaggle.com/code/hengwdai/amazon-book-price-h2o/)        |
| Regression     | [MercariPrice](https://www.kaggle.com/hengwdai/mercariprice-data-split)                  | RMSE              | [32.042](https://www.kaggle.com/code/hengwdai/mercariprice-auto3ml)                                                | [34.500](https://www.kaggle.com/code/hengwdai/mercariprice-autogluon)         | [43.960](https://www.kaggle.com/code/hengwdai/mercariprice-h2o)               |
| Classification | [Titanic](https://www.kaggle.com/competitions/titanic/data)                              | AUC               | [0.794](https://www.kaggle.com/code/hengwdai/autox-titanic)                                                        | [0.780](https://www.kaggle.com/code/sishihara/autogluon-tabular-for-titanic)  | [0.768](https://www.kaggle.com/code/hengwdai/titanic-solution-with-basic-h2o) |
| Classification | [Stumbleupon](https://www.kaggle.com/hengwdai/stumbleupon-data-split)                    | AUC               | [0.871](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/Stumbleupon/stumbleupon-auto3ml.ipynb) | [0.503](https://www.kaggle.com/code/hengwdai/stumbleupon-autogluon)           | [0.707](https://www.kaggle.com/code/hengwdai/stumbleupon-h2o)                 |
| Classification | [DisasterTweets](https://www.kaggle.com/competitions/nlp-getting-started/data)           | AUC               | [0.786](https://www.kaggle.com/code/hengwdai/tweeter-autox)                                                        | [0.746](https://www.kaggle.com/hengwdai/tweeter-autogluon)                    | [0.721](https://www.kaggle.com/code/hengwdai/tweeter-h2o)                     |

# 处理效率对比

| Dataset              | Text Column     | Average Text Length | TPS    | AutoX                                                                                                                                   | AutoGluon                                                                                                                                | H2O                                                                                                                          |
|----------------------|-----------------|---------------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| MercariPrice         | BrandName       | 6                   | item/s | [4736.42](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/autox_fe-6-speed.ipynb)           | [141.06](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/autoglueon-6-speed.ipynb)           | [1940.45](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/h2o-6-speed.ipynb)     |
| MercariPrice         | CategoryName    | 30                  | item/s | [2661.58](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/autox_fe-30-speed.ipynb)          | [143.82](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/autoglueon-30-speed.ipynb)          | [1420.02](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/h2o-30-speed.ipynb)    |
| MercariPrice         | ItemDescription | 150                 | item/s | [1762.78](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/autox_fe-150-speed.ipynb)         | [125.41](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/autoglueon-150-speed.ipynb)         | [353.14](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/MercariPrice/Efficiency/h2o-150-speed.ipynb)    |
| TMDBBoxOffice        | Overview        | 300                 | item/s | [462.96](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/TMDBBoxOffice/Efficiency/autox_fe-300-speed.ipynb)         | [20.40](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/TMDBBoxOffice/Efficiency/autoglueon-300-speed.ipynb)         | [194.55](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/TMDBBoxOffice/Efficiency/h2o-300-speed.ipynb)   |
| CommonlitReadability | Excerpt         | 1000                | item/s | [179.15](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/CommonlitReadability/Efficiency/autox_fe-1000-speed.ipynb) | [18.59](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/CommonlitReadability/Efficiency/autoglueon-1000-speed.ipynb) | [58.46](https://github.com/4paradigm/AutoX/blob/master/autox/autox_nlp/demo/CommonlitReadability/Efficiency/h2o-speed.ipynb) |
