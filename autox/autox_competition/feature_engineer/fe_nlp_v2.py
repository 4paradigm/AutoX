import numpy as np
import pandas as pd
import warnings
from autox.autox_competition.process_data import Feature_type_recognition
from autox.autox_competition.CONST import FEATURE_TYPE
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
                          DataCollatorForLanguageModeling, pipeline,
                          Trainer, TrainingArguments, PreTrainedTokenizerFast)

warnings.filterwarnings('ignore')


def tokens2sentence(x):
    return ' '.join(x)


def dummy_fun(doc):
    return doc


def get_words_w2v_vec(model, sent):  # 使用word2vec获取整句话的vec
    return np.array(list(map(lambda x: model.wv[x], sent)), dtype=float).mean(axis=0)


def get_words_glove_vec(model, sent):  # 使用glove获取整句话的vec
    return np.array(list(map(lambda x: model.word_vectors[model.dictionary[x]], sent)), dtype=float).mean(axis=0)


class NLP_feature():
    def __init__(self):
        self.tokenizers = {}
        self.embedded_texts = {}
        self.model_name = 'prajjwal1/bert-tiny'
        self.zero_shot_model = 'typeform/mobilebert-uncased-mnli'  # joeddav/xlm-roberta-large-xnli
        self.embeddings = {}
        self.encoders = {}
        self.candidate_labels = {}
        self.pipline = None
        self.n_clusters = 2
        self.do_mlm = False
        self.emb_size = 32
        self.use_tokenizer = False
        self.text_columns_def = []
        self.y = None
        self.task = None
        self.embedding_model = None
        self.df_feature_type = None
        self.silence_cols = []
        

    def fit(self, df, task = 'embedding', embedding_model = 'TFIDF', use_tokenizer = False, silence_cols = [], 
            df_feature_type = None, y = None, candidate_labels=None):
        self.task = task
        self.use_tokenizer = use_tokenizer
        self.embedding_model = embedding_model
        self.candidate_labels = candidate_labels
        self.df_feature_type = df_feature_type
        self.silence_cols = silence_cols
        self.y = y

        if self.df_feature_type is None:
            feature_type_recognition = Feature_type_recognition()
            feature_type = feature_type_recognition.fit(df)
            self.df_feature_type = feature_type

        for feature in self.df_feature_type.keys():
            if self.df_feature_type[feature] == FEATURE_TYPE['txt'] and feature not in self.silence_cols:
                self.text_columns_def.append(feature)

        df = df.loc[:, self.text_columns_def]
        if self.task == 'zero-shot-classification':
            self.pipeline = pipeline(self.task, model=self.zero_shot_model)
            return

        ## 1、训练分词器，如果不使用，则默认使用空格分词
        if self.use_tokenizer:
            self.fit_tokenizers(df)

        ## 2、分词处理，用于下一步embedding model的训练
        if self.embedding_model != 'Bert':
            for column in self.text_columns_def:
                df[f'{column}_tokenized_ids'] = self.tokenize(df, column)
        
        ## 3、训练embedding model,
        ## 初步确定五种： TFIDF、FastText、Word2Vec、Glove、 BertEmbedding,训练的embedding model数量与文本特征列数量相同,使用字典存储，索引为特征列名
        self.fit_embeddings(df)
        
        ## 4、训练编码器，对embedding 进行特征降维，
        ## 初步确定两种编码器的选择：有监督的target encode 使用岭回归模型，2)	无监督的k-means：使用k-means算法
        if self.task == 'supervise' or self.task == 'unsupervise': 
            self.fit_encoders(df)
        ## 使用提取器处理文本数据生成新特征

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
            if self.embedding_model != 'Bert':
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
        if self.embedding_model == 'TFIDF':
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

        elif self.embedding_model == 'FastText':
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
        elif self.embedding_model == 'Word2Vec':
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
        elif self.embedding_model == 'Glove':
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
        elif self.embedding_model == 'Bert':
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
                if self.do_mlm:
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
        if self.embedding_model == 'TFIDF':
            embedded_text = embedding.transform(self.tokenize(raw_df, name))
        elif self.embedding_model == 'FastText':
            embedded_text = sparse.csr_matrix(embedding.wv[
                                                  list(map(lambda x: tokens2sentence(x), self.tokenize(raw_df, name)))
                                              ])
        elif self.embedding_model == 'Word2Vec':
            embedded_text = sparse.csr_matrix(np.array(
                list(map(lambda x: get_words_w2v_vec(embedding, x), self.tokenize(raw_df, name)))
            ))
        elif self.embedding_model == 'Glove':
            embedded_text = sparse.csr_matrix(np.array(
                list(map(lambda x: get_words_glove_vec(embedding, x), self.tokenize(raw_df, name)))
            ))
        elif self.embedding_model == 'Bert':
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

    def fit_encoders(self, df):

        def micro_regressor(embedding_array, y):
            regressor = Ridge(random_state=42, alpha=0.8)
            regressor.fit(embedding_array, y)
            return regressor

        def micro_cluster(embedding_array):
            cluster = KMeans(n_clusters=self.n_clusters)
            cluster.fit(embedding_array)
            return cluster

        for column in self.text_columns_def:
            encoders = []
            folds = KFold(n_splits=5, shuffle=True, random_state=42)
            for fold_n, (train_index, valid_index) in enumerate(folds.split(df)):
                if self.embedding_model != 'Bert':
                    trn = df[f'{column}_tokenized_ids'][train_index]
                    vld = df[f'{column}_tokenized_ids'][valid_index]
                else:
                    trn = self.tokenizers[column](df[column][train_index].to_list(), max_length=128, 
                            return_tensors='pt', padding='max_length', truncation=True)
                    vld = self.tokenizers[column](df[column][valid_index].to_list(), max_length=128, 
                            return_tensors='pt', padding='max_length', truncation=True)
                if self.embedding_model == 'TFIDF':
                    trn = self.embeddings[column].transform(trn)
                    vld = self.embeddings[column].transform(vld)
                elif self.embedding_model == 'FastText':
                    trn = sparse.csr_matrix(self.embeddings[column].wv[(trn.apply(tokens2sentence))])
                    vld = self.embeddings[column].wv[(vld.apply(tokens2sentence))]
                elif self.embedding_model == 'Word2Vec':
                    trn = sparse.csr_matrix(
                        np.array(list(map(lambda x: get_words_w2v_vec(self.embeddings[column], x), trn))))
                    vld = np.array(list(map(lambda x: get_words_w2v_vec(self.embeddings[column], x), vld)))
                elif self.embedding_model == 'Glove':
                    trn = sparse.csr_matrix(
                        np.array(list(map(lambda x: get_words_glove_vec(self.embeddings[column], x), trn))))
                    vld = np.array(list(map(lambda x: get_words_glove_vec(self.embeddings[column], x), vld)))
                elif self.embedding_model == 'Bert':
                    model = AutoModel.from_pretrained(self.embeddings[column], output_hidden_states=True)
                    trn = sparse.csr_matrix(model(**trn)['hidden_states'][0].detach().numpy().mean(1))
                    vld = sparse.csr_matrix(model(**vld)['hidden_states'][0].detach().numpy().mean(1))
                else:
                    raise NotImplementedError

                if self.task == 'supervise':
                    y_trn = self.y.iloc[train_index]
                    encoders.append(micro_regressor(trn, y_trn))
                elif self.task == 'unsupervise':
                    encoders.append(micro_cluster(trn))

            self.encoders.update({column: encoders})

    def transform(self, df):
        if self.task == 'embedding':
            for column in self.text_columns_def:
                df[f'{column}_embedded'] = self.emb_text(df, column)
            return df
        elif self.task == 'supervise' or self.task == 'unsupervise':
            for column in self.text_columns_def:
                embedded_text = self.emb_text(df, column)
                meta_test = None
                for idx in range(5):
                    encoder = self.encoders[column][idx]
                    if self.task == 'supervise':
                        pred = (encoder.predict(embedded_text)) / 5
                    elif self.task == 'unsupervise':
                        pred = np.eye(self.n_clusters)[encoder.predict(embedded_text)]
                    if idx == 0:
                        meta_test = pred
                    else:
                        meta_test += pred
                    if self.task == 'unsupervise':
                        meta_test = np.argmax(meta_test, axis=1)
                #                     meta_test = pd.DataFrame(np.eye(self.n_clusters)[np.argmax(meta_test,axis=1)])
                #                     for idx in range(self.n_clusters):
                #                         df[f'{column}_transformed_class{idx}'] =  meta_test[idx]
                df[f'{column}_transformed'] = meta_test
            return df
        elif self.task == 'zero-shot-classification':
            for column in self.text_columns_def:
                pred_labels = []
                candidate_labels = self.candidate_labels[column]
                for text in df[column]:
                    results = classifier(text, candidate_labels)
                    pred_labels.append(results['labels'][np.argmax(results['scores'])])
                df[f'{column}_transformed'] = pred_labels
            return df
        else:
            raise NotImplementedError
    def fit_transform(self, df, task = 'embedding', embedding_model = 'TFIDF', use_tokenizer = False, silence_cols = [], 
            df_feature_type = None, y = None, candidate_labels=None):
        self.fit(self, df, task = task, embedding_model = embedding_model, use_tokenizer = use_tokenizer, silence_cols = silence_cols, 
            df_feature_type = df_feature_type, y = y, candidate_labels=candidate_labels)
        return self.transform(df)
