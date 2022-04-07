import warnings

import numpy as np
import pandas as pd
from datasets import Dataset
from gensim.models import FastText, Word2Vec
from glove import Glove, Corpus
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, PreTrainedTokenizerFast)

warnings.filterwarnings('ignore')


def tokens2sentence(x):
    return ' '.join(x)

def dummy_fun(doc):
    return doc

def get_words_w2v_vec(model,sent):#使用word2vec获取整句话的vec
    return np.array(list(map(lambda x:model.wv[x],sent)),dtype=float).mean(axis=0)

def get_words_glove_vec(model,sent):#使用glove获取整句话的vec
    return np.array(list(map(lambda x:model.word_vectors[model.dictionary[x]],sent)),dtype=float).mean(axis=0)


class NLP_feature():
    def __init__(self):
        self.tokenizers = {}
        self.embedded_texts = {}
        self.model_name = 'prajjwal1/bert-tiny'
        self.embeddings = {}
        self.encoders = {}
        self.emb_size = 32
        self.use_tokenizer = False
        self.text_columns_def = None
        self.y = None
        self.task = None
        self.embedding_mode = None

    def fit(self, df, text_columns_def, use_tokenizer, embedding_mode, task, y=None):
        self.task = task
        self.use_tokenizer = use_tokenizer
        self.text_columns_def = text_columns_def
        self.embedding_mode = embedding_mode
        self.y = y
        df = df.loc[:, text_columns_def]
        ## 训练分词器，如果不使用，则默认使用空格分词
        if self.use_tokenizer:
            self.fit_tokenizers(df)
        if self.embedding_mode != 'Bert':
            for column in self.text_columns_def:
                df[f'{column}_tokenized_ids'] = self.tokenize(df, column)
        #         return df
        ## 训练embedding,初步确定五种： TFIDF、FastText、Word2Vec、Glove、 BertEmbedding,训练的embedding model数量与文本特征列数量相同,使用字典存储，索引为特征列名
        self.fit_embeddings(df)
        ## 根据task训练特征提取器,训练的encoder model数量与文本特征列数量相同,使用字典存储，索引为特征列名, 与每一列的embedding model也是一一对应
        ## 目前支持：有监督回归数值特征、无监督KNN距离特征、无监督关键词离散特征(语义，情感，。。。）
        ## 特殊任务：根据两两对比数据 生成 每个数据的具体得分，比如通过成对评论的恶意(替换成任何一种语义程度都可以)比较，生成单个评论的恶意程度
        ## 注意： 关键词离散特征和特殊任务目前只支持使用深度模型，无法自选tokenizer和embedding
        return self.fit_encoders(df, y)
        ## 使用提取器处理文本数据生成新特征

    def transform(self, df):
        return self.encoder(df)

    def fit_tokenizers(self, df):
        raw_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        raw_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
        raw_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(vocab_size=500, special_tokens=special_tokens)

        #         df = pd.concat([df_train_new[['comment_text']]])

        def micro_tokenizer(text_df, name):
            text = pd.concat([text_df])
            dataset = Dataset.from_pandas(text)

            def get_training_corpus():
                for i in range(0, len(dataset), 1000):
                    yield dataset[i: i + 1000][name]

            raw_tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw_tokenizer,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )
            return tokenizer

        for column in self.text_columns_def:
            print(f'Fitting column: {column} tokenizer')
            if self.embedding_mode != 'Bert':
                self.tokenizers.update({column: micro_tokenizer(df[[column]], column)})
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizers.update({column: tokenizer})

    def tokenize(self, df, column):
        if self.use_tokenizer:
            #             for column in self.text_columns_def:
            print(f'Tokenizing column: {column}')
            tokenizer = self.tokenizers[column]
            return list(map(lambda x: self.tokenizers[column].convert_ids_to_tokens(x),
                            self.tokenizers[column](df[column].to_list())['input_ids']))
        else:
            return [text.split(' ') for text in df[column].to_list()]

    def fit_embeddings(self, df):
        if self.embedding_mode == 'TFIDF':
            def micro_tfidf(text_df, name):
                vectorizer = TfidfVectorizer(
                    analyzer='word',
                    tokenizer=dummy_fun,
                    preprocessor=dummy_fun,
                    token_pattern=None
                )
                self.embeddings.update({name: vectorizer.fit(text_df.to_list())})

            for column in self.text_columns_def:
                print(f'Fitting column: {column} tfidf embedding')
                micro_tfidf(df[f'{column}_tokenized_ids'], column)

        elif self.embedding_mode == 'FastText':
            def micro_fasttext(text_df, name):
                model = FastText(vector_size=self.emb_size)
                #                 text_df = text_df.apply(tokens2sentence)
                model.build_vocab(text_df)
                model.train(
                    text_df, epochs=model.epochs,
                    total_examples=model.corpus_count, total_words=model.corpus_total_words,
                )
                self.embeddings.update({name: model})

            for column in self.text_columns_def:
                print(f'Fitting column: {column} fasttext embedding')
                micro_fasttext(df[f'{column}_tokenized_ids'], column)
        elif self.embedding_mode == 'Word2Vec':
            def micro_word2vec(text_df, name):
                model = Word2Vec(vector_size=self.emb_size, min_count=0)
                model.build_vocab(text_df)
                model.train(
                    text_df, epochs=model.epochs,
                    total_examples=model.corpus_count, total_words=model.corpus_total_words,
                )
                self.embeddings.update({name: model})

            for column in self.text_columns_def:
                print(f'Fitting column: {column} word2vec embedding')
                micro_word2vec(df[f'{column}_tokenized_ids'], column)
        elif self.embedding_mode == 'Glove':
            def micro_glove(text_df, name):
                corpus_model = Corpus()
                corpus_model.fit(text_df, window=10, ignore_missing=False)
                model = Glove(no_components=self.emb_size)
                model.fit(corpus_model.matrix, epochs=10)
                model.add_dictionary(corpus_model.dictionary)
                self.embeddings.update({name: model})

            for column in self.text_columns_def:
                print(f'Fitting column: {column} glove embedding')
                micro_glove(df[f'{column}_tokenized_ids'], column)
        elif self.embedding_mode == 'Bert':
            def micro_bert(text_df, name):
                text_df = text_df.apply(lambda x: x.replace('\n', ''))
                text = '\n'.join(text_df.tolist())
                with open('text.txt', 'w') as f:
                    f.write(text)
                model = AutoModelForMaskedLM.from_pretrained(self.model_name)
                model_path = f'./{name}_transformer'
                train_dataset = LineByLineTextDataset(
                    tokenizer=self.tokenizers[name],
                    file_path="text.txt",
                    block_size=128)
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizers[name], mlm=True, mlm_probability=0.15)
                training_args = TrainingArguments(
                    output_dir=model_path,
                    num_train_epochs=2,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    evaluation_strategy='no',
                    save_strategy='no',
                    report_to="none")
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=None)
                trainer.train()
                trainer.save_model(model_path)
                self.embeddings.update({name: model_path})

            for column in self.text_columns_def:
                print(f'Fitting column: {column} bert embedding')
                micro_bert(df[column], column)
        else:
            raise NotImplementedError

    def emb_text(self, raw_df, name):
        #             tokenizer = self.tokenizers[name]
        embedding = self.embeddings[name]
        embedded_text = None
        if self.embedding_mode == 'TFIDF':
            embedded_text = embedding.transform(self.tokenize(raw_df, name))
        elif self.embedding_mode == 'FastText':
            embedded_text = sparse.csr_matrix(embedding.wv[
                                                  list(map(lambda x: tokens2sentence(x), self.tokenize(raw_df, name)))
                                              ])
        elif self.embedding_mode == 'Word2Vec':
            embedded_text = sparse.csr_matrix(np.array(
                list(map(lambda x: get_words_w2v_vec(embedding, x), self.tokenize(raw_df, name)))
            ))
        elif self.embedding_mode == 'Glove':
            embedded_text = sparse.csr_matrix(np.array(
                list(map(lambda x: get_words_glove_vec(embedding, x), self.tokenize(raw_df, name)))
            ))
        elif self.embedding_mode == 'Bert':
            model = AutoModel.from_pretrained(self.embeddings[name], output_hidden_states=True)
            encoded_input = self.tokenizers[name](raw_df[name].to_list(),
                                                  max_length=128,
                                                  return_tensors='pt',
                                                  padding='max_length',
                                                  truncation=True)
            embedded_text = sparse.csr_matrix(model(**encoded_input)['hidden_states'][0].detach().numpy().mean(1))
        else:
            raise NotImplementedError
        return embedded_text

    def fit_encoders(self, df, y=None):
        if self.task == 'embedding':
            res_dict = {}
            for column in self.text_columns_def:
                res_dict.update({column: self.emb_text(df, column)})
            return res_dict
        #                 df[f"{column}_meta_feature"] = self.emb_text(df[column],column)

        elif self.task == 'regression':
            def micro_regressor(embedding_array, y):
                regressor = Ridge(random_state=42, alpha=0.8)
                regressor.fit(embedding_array, y)
                return regressor

            for column in self.text_columns_def:
                encoders = []
                folds = KFold(n_splits=5, shuffle=True, random_state=42)
                for fold_n, (train_index, valid_index) in enumerate(folds.split(df)):
                    if self.embedding_mode != 'Bert':
                        trn = df[f'{column}_tokenized_ids'][train_index]
                        vld = df[f'{column}_tokenized_ids'][valid_index]
                    else:
                        trn = self.tokenizers[column](
                            df[column][train_index].to_list(),
                            max_length=128,
                            return_tensors='pt',
                            padding='max_length',
                            truncation=True)
                        vld = self.tokenizers[column](
                            df[column][valid_index].to_list(),
                            max_length=128,
                            return_tensors='pt',
                            padding='max_length',
                            truncation=True)
                    y_trn = y.iloc[train_index]
                    if self.embedding_mode == 'TFIDF':
                        trn = self.embeddings[column].transform(trn)
                        vld = self.embeddings[column].transform(vld)
                    elif self.embedding_mode == 'FastText':
                        trn = sparse.csr_matrix(self.embeddings[column].wv[(trn.apply(tokens2sentence))])
                        vld = self.embeddings[column].wv[(vld.apply(tokens2sentence))]
                    elif self.embedding_mode == 'Word2Vec':
                        trn = sparse.csr_matrix(
                            np.array(list(map(lambda x: get_words_w2v_vec(self.embeddings[column], x), trn))))
                        vld = np.array(list(map(lambda x: get_words_w2v_vec(self.embeddings[column], x), vld)))
                    elif self.embedding_mode == 'Glove':
                        trn = sparse.csr_matrix(
                            np.array(list(map(lambda x: get_words_glove_vec(self.embeddings[column], x), trn))))
                        vld = np.array(list(map(lambda x: get_words_glove_vec(self.embeddings[column], x), vld)))
                    elif self.embedding_mode == 'Bert':
                        model = AutoModel.from_pretrained(self.embeddings[column], output_hidden_states=True)
                        trn = sparse.csr_matrix(model(**trn)['hidden_states'][0].detach().numpy().mean(1))
                        vld = sparse.csr_matrix(model(**vld)['hidden_states'][0].detach().numpy().mean(1))
                    else:
                        raise NotImplementedError
                    encoders.append(micro_regressor(trn, y_trn))

                    val = encoders[fold_n].predict(vld)
                    df.loc[valid_index, f"{column}_meta_feature"] = val
                self.encoders.update({column: encoders})
                if self.embedding_mode != 'Bert':
                    df = df.drop(columns=[f'{column}_tokenized_ids'])
            return df

    def transform(self, df):
        if self.task == 'embedding':
            for column in self.text_columns_def:
                df[f'{column}_embedded'] = self.emb_text(df, column)
            return df
        if self.task == 'regression':

            for column in self.text_columns_def:
                embedded_text = self.emb_text(df, column)
                meta_test = None
                for idx in range(5):
                    encoder = self.encoders[column][idx]
                    pred = (encoder.predict(embedded_text)) / 5
                    if idx == 0:
                        meta_test = pred
                    else:
                        meta_test += pred
                df[f'{column}_transformed'] = meta_test
            return df