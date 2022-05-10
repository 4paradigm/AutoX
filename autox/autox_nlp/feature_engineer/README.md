[English](./README_EN.md) | 简体中文

# autox_nlp.feature_engineer是什么
feature_engineer 是autox_nlp的特征工程模块。

<div align="center"><img height="540" src="../img/NLP_feature_eng.png" width="303"/></div> 

# 目录
<!-- TOC -->
- [调用方式](#调用方式)
- [按分词方式划分](#按分词方式划分)
- [按特征提取方式划分](#按特征提取方式划分)
- [按特征输出形式划分](#按特征输出形式划分)
- [参数介绍](#参数介绍)
- [属性介绍](#属性介绍)

<!-- /TOC -->
# 调用方式

```
from autox.autox_nlp import NLP_feature
import pandas as pd

nlp = NLP_feature()

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# Use fit to get meta_feature
meta_feature = nlp.fit(train, ['text_column_name'], use_Toknizer, embedding_mode, task, y, candidate_labels)

# Concat meta feature with raw data
for column in meta_feature.columns:
    train[column] = meta_feature[column]
    
test = nlp.transform(test)

train.to_csv('new_train.csv')
test.to_csv('new_test.csv')
```

# 快速上手
[使用demo：CommmonLit Readability prize](https://www.kaggle.com/hengwdai/quickstart-auto3ml-nlp)
## 按分词方式划分
### 空格分词
```

use_Toknizer=False

df = nlp.fit(df_train,['text_column_name'],use_Toknizer,'Word2Vec','unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
### 无监督分词器分词
```

use_Toknizer=True

df = nlp.fit(df_train,['text_column_name'],use_Toknizer,'Word2Vec','unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
## 按特征提取方式划分
### TFIDF
```

emb_mode='TFIDF'

df = nlp.fit(df_train,['text_column_name'],True,emb_mode,'unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
### Word2Vec
```

emb_mode='Word2Vec'

df = nlp.fit(df_train,['text_column_name'],True,emb_mode,'unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
### FastText
```

emb_mode='FastText'

df = nlp.fit(df_train,['text_column_name'],True,emb_mode,'unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
### Glove
```

emb_mode='Glove'

df = nlp.fit(df_train,['text_column_name'],True,emb_mode,'unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
### Bert
```

emb_mode='Bert'

df = nlp.fit(df_train,['text_column_name'],True,emb_mode,'unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
### Zero-shot Labeling
```

task='zero-shot-classification'
hypothesis = {'text_column_name':[
                        'this text is too complex',
                        'this text is easy to understand'
                        ]}

df = nlp.fit(
              df = df_train,
              text_columns_def = ['text_column_name'],
              use_tokenizer = True,
              text_columns_def = None,
              task = task, 
              y = None,
              text_columns_def = hypothesis )

df_train = nlp.transform(df_train)
test = nlp.transform(test)

```
## 按特征输出形式划分
### 直接输出embedding
```

task=embedding

train_sparse_matrix = nlp.fit(df_train,['text_column_name'],True,'Word2Vec',task)

test_sparse_matrix = nlp.transform(test)
```
### 使用target encode输出数值型特征
```

task='supervise'

df = nlp.fit(df_train,['text_column_name'],True,'Word2Vec',task)

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)

```
### 使用k means输出离散型特征
```

task='unsupervise'

df = nlp.fit(df_train,['text_column_name'],True,'Word2Vec',task)

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)

