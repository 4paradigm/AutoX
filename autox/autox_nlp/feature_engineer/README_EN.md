[English](./README_EN.md) | 简体中文

# autox_nlp.feature_engineer
The framework of the NLP_feature is shown below.


<div align="center"><img height="540" src="../img/NLP_feature_eng.png" width="303"/></div> 

# Catalogue
<!-- TOC -->
- [Calling Method](#Calling Method)
- [Different token split methods](#Different token split methods)
- [Different feature extract methods](#Different feature extract methods)
- [Different feature output methods](#Different feature output methods)
- [class NLP_feature](#NLP_feature)
<!-- /TOC -->

# Calling Method
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

# Quick Start
[demo：CommmonLit Readability prize](https://www.kaggle.com/hengwdai/quickstart-auto3ml-nlp)
##Different token split methods

### Tokenizing by space split
```

use_Toknizer=False

df = nlp.fit(df_train,['text_column_name'],use_Toknizer,'Word2Vec','unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
###Tokenizing by Tokenizer
```

use_Toknizer=True

df = nlp.fit(df_train,['text_column_name'],use_Toknizer,'Word2Vec','unsupervise')

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```
##Different feature extract methods

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
##Different feature output methods
 
### Directly output embedding
```

task='embedding'

train_sparse_matrix = nlp.fit(df_train,['text_column_name'],True,'Word2Vec',task)

test_sparse_matrix = nlp.transform(test)
```
### Use target encode to get numeric feature
```

task='supervise'

df = nlp.fit(df_train,['text_column_name'],True,'Word2Vec',task)

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)

```
### Use k-means to get category  feature
```

task='unsupervise'

df = nlp.fit(df_train,['text_column_name'],True,'Word2Vec',task)

# Concat meta feature with raw data

for column in df.columns:
    df_train[column] = df[column]

test = nlp.transform(test)
```

#NLP_feature

Text feature extraction tool processes the text with the process of word segmentation, word embedding (feature extraction) and feature dimension reduction.
## Attributes
```
· text_columns_def (list)       - Text columns' name of Dataset
· task (str)                    - Feature dimension reduction methods ('embedding'/'supervise'/'unsupervise'/'zero-shot-classification')
· y (DataFrame)                 - target column of Dataset
· use_tokenizer (bool)          - Whether to use tokenizer
· embedding_mode (str)          - Feature extract methods ('TFIDF'/'Word2Vec'/'Glove'/'FastText'/'Bert')
· candidate_labels (dict)       - while using zero-shot labeling，need to set the possible categories for each text column，and store in the dictionary
· tokenizers (dict)             - Tokenizers stored in a dictionary form，the key represents the text column names，value is the corresponding tokenizer
· embeddings (dict)             - Feature extract (word embedding) models stored in a dictionary form，the key represents the text column names，value is the corresponding model
· encoders (dict)               - Feature dimension reduction models stored in a dictionary form，the key represents the text column names，value is the corresponding model
· model_name (str)              - The pre-trained model used for word embedding (feature extraction) using Bert, or other models on the huggingface
· zero_shot_model (str)         - The pre-trained model used for feature extraction using zero-shot, and the other models on the huggingface can be used
· corpus_model (dict)           - Glove front model stored in dictionary with key represents the text column name and value the corresponding model
· device (str)                  - All inference environments that currently use deep model scenarios are automatically set to cuda if the GPU is supported
· pipline (huggingface pipline) - pipeline used for feature extraction using zero-shot
· n_clusters (int)              - The output dimension of feature dimensionality reduction using k-means
· do_mlm (bool)                 - Is mask language used for word embedding using Bert
· mlm_epochs (int)              - Training epochs pretrained with mlm for word embedding with Bert
· emb_size (int)                - Output dimensions for word embedding when using Word2Vec, FastText, Glove
```
## NLP_feature.fit
The text columns in the training data were used to obtain the pipeline for feature extraction and output the extracted training data text features.
### Parameters
```
· df (pandas.DataFrame)                                 - Training dataset containing the text columns
· text_columns_def (list)                               - Text column names stored through the list,e.g.,['text1','text2']
· use_tokenizer (bool, optional, defaults to True)      - Whether to use an unsupervised tokenizer(True,False)
· embedding_mode (str, optional, defaults to 'TFIDF')   - Feature extract (word embedding) method('TFIDF'/'Word2Vec'/'Glove'/'FastText'/'Bert')
· task (str, optional, defaults to 'unsupervise')       - Feature dimension reduction models('embedding'/'supervise'/'unsupervise'/'zero-shot-classification')
· y (pandas.DataFrame, optional, defaults to None)      - Target value column in the dataset,e.g., df['target']
· candidate_labels (dict, optional, defaults to None)   - If you use zero shot, you need to specify possible category labels for each text column,e.g.,{'text1':['label1','label2'],'text2':['label3','label2']}
```
### Return: pandas.DataFrame / dict { text_name : sparse.csr_matrix }
If embedding is selected as the dimension reduction mode, the return value format is a dictionary composed of text column names and sparse.csr_matrix, and an DataFrame composed of features of each column of text column.
## NLP_feature.transform
Feature extraction of the new test data was performed using the pipeline done with fit.The splicing of the test set with the extracted features was taken as the output.
### Parameters
```
The test dataset containing the text columns, and the columns specified by the fit entry reference 'text_columns_def' must be included in the test data.
```
### Return：pandas.DataFrame / dict { text_name : sparse.csr_matrix }
The new dataset is obtained from the input test set, and the feature column splicing converted from each column of text column.

    
    