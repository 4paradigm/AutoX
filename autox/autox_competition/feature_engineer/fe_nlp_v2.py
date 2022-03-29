import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
from sklearn.linear_model import Ridge
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)



class NLP_feature():
    def __init__(self):
        self.tokenizers = {}
        self.embedded_texts = {}
        self.embeddings = {}
        self.encoders = {}
        self.use_tokenizer = False
        self.text_columns_def = None

    def fit(self, df, text_columns_def, use_tokenizer, embedding_mode, task, y=None):
        self.task = task
        self.use_tokenizer = use_tokenizer
        self.text_columns_def = text_columns_def
        df = df.loc[:, text_columns_def]
        ## 训练分词器，如果不使用，则默认使用空格分词
        if self.use_tokenizer:
            self.fit_tokenizers(df)
        df = self.tokenize(df)
        ## 训练embedding,初步确定五种： TFIDF、FastText、Word2Vec、Glove、 BertEmbedding,训练的embedding model数量与文本特征列数量相同,使用字典存储，索引为特征列名
        self.fit_embeddings(df, embedding_mode)
        ## 根据task训练特征提取器,训练的encoder model数量与文本特征列数量相同,使用字典存储，索引为特征列名, 与每一列的embedding model也是一一对应
        ## 目前支持：有监督回归数值特征、无监督KNN距离特征、无监督关键词离散特征(语义，情感，。。。）
        ## 特殊任务：根据两两对比数据 生成 每个数据的具体得分，比如通过成对评论的恶意(替换成任何一种语义程度都可以)比较，生成单个评论的恶意程度
        ## 注意： 关键词离散特征和特殊任务目前只支持使用深度模型，无法自选tokenizer和embedding
        self.fit_encoders(task, y)
        ## 使用提取器处理文本数据生成新特征

    def transform(self, df):
        return self.encoder(df)

    def fit_tokenizers(self, df):
        raw_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        raw_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
        raw_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(vocab_size=50000, special_tokens=special_tokens)

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
            self.tokenizers.update({column: micro_tokenizer(df[[column]], column)})

    def tokenize(self, df):
        if self.use_tokenizer:
            for column in self.text_columns_def:
                print(f'Tokenizing column: {column}')
                df[f'{column}_tokenized_ids'] = self.tokenizers[column](df[column].to_list())['input_ids']
                df = df.drop(columns=[column])
            return df
        else:
            for column in self.text_columns_def:
                print(f'Splitting column: {column}')
                df[f'{column}_tokenized_ids'] = [text.split(' ') for text in df[column].to_list()]
                df = df.drop(columns=[column])
            return df

    def fit_embeddings(self, df, embedding_mode):
        if embedding_mode == 'TFIDF':
            def dummy_fun(doc):
                return doc

            def micro_tfidf(text_df, name):
                vectorizer = TfidfVectorizer(
                    analyzer='word',
                    tokenizer=dummy_fun,
                    preprocessor=dummy_fun,
                    token_pattern=None
                )
                self.embeddings.update({name: vectorizer.fit(text_df.to_list())})

            for column in self.text_columns_def:
                print(f'Fitting column: {column} embedding')
                micro_tfidf(df[f'{column}_tokenized_ids'], column)
                self.embedded_texts.update({column: self.embeddings[column].transform(df[f'{column}_tokenized_ids'])})
        #                 df[f'{column}_embeddings'] = (self.embeddings[column].transform(df[f'{column}_tokenized_ids'])).toarray()
        #                 df = df.drop(columns=[f'{column}_tokenized_ids'])
        #             return df
        else:
            raise NotImplementedError

    def fit_encoders(self, task, y=None):
        if task == 'embedding':
            return
        elif task == 'regression':
            def micro_regressor(embedding_array, name, y):
                regressor = Ridge(random_state=42, alpha=0.8)
                regressor.fit(embedding_array, y)
                self.encoders.update({name: regressor})

            for column in self.text_columns_def:
                micro_regressor(self.embedded_texts[column], column, y)

    def transform(self, df):
        if self.task == 'regression':
            for column in self.text_columns_def:
                tokenizer = self.tokenizers[column]
                embedding = self.embeddings[column]
                encoder = self.encoders[column]
                df[f'{column}_transformed'] = tokenizer(df[column].to_list())['input_ids']
                embedded_text = embedding.transform(df[f'{column}_transformed'])
                df[f'{column}_transformed'] = encoder.predict(embedded_text)
            return df