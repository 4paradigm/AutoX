import numpy as np
import gc
import pandas as pd
import torch
from gensim.models import FastText, Word2Vec

try:
    os.system('pip install glove-python-binary')
    from glove import Glove, Corpus

    Glove_installed = True
except Exception as e:
    print("Your environment is not support to install glove-python-binary, so the 'embedding_mode=Glove' method is \
          not supported")
    Glove_installed = False
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, PreTrainedTokenizerFast, pipeline)
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as DataSet
from sklearn.linear_model import Ridge
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import warnings

warnings.filterwarnings('ignore')

from tqdm.auto import tqdm


def tokens2sentence(x):
    return ' '.join(x)


def dummy_fun(doc):
    return doc


def get_words_w2v_vec(model, sent):  # 使用word2vec获取整句话的vec
    return np.array(list(map(lambda x: model.wv[x], sent)), dtype=float).mean(axis=0)


def get_words_glove_vec(model, sent):  # 使用glove获取整句话的vec
    return np.array(list(map(lambda x: model.word_vectors[model.dictionary[x]], sent)), dtype=float).mean(axis=0)


def prepare_input(tokenizer, text):
    inputs = tokenizer(text,
                       max_length=128,
                       return_tensors='pt',
                       padding='max_length',
                       truncation=True)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = texts.values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.tokenizer,
                               self.texts[item])
        return inputs


def model_infer(model, data, device):
    preds = []
    model.eval()
    model.to(device)
    for inputs in data:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(**inputs)
        preds.append(y_preds['hidden_states'][0].detach().to('cpu').numpy().mean(1))
    predictions = np.concatenate(preds, axis=0)
    return predictions


class NLP_feature():
    def __init__(self):
        self.tokenizers = {}
        self.embedded_texts = {}
        self.model_name = 'prajjwal1/bert-tiny'
        self.zero_shot_model = 'typeform/mobilebert-uncased-mnli'  # joeddav/xlm-roberta-large-xnli
        self.embeddings = {}
        self.corpus_model = {}
        self.encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.candidate_labels = {}
        self.pipline = None
        self.n_clusters = 16
        self.do_mlm = False
        self.mlm_epochs = 2
        self.emb_size = 32
        self.use_tokenizer = False
        self.text_columns_def = None
        self.y = None
        self.task = None
        self.embedding_mode = None

    def fit_transform(self, df, text_columns_def, use_tokenizer=True, embedding_mode='TFIDF', task='unsupervise',
                      y=None,
                      candidate_labels=None):
        self.task = task
        self.use_tokenizer = use_tokenizer
        self.text_columns_def = text_columns_def
        self.embedding_mode = embedding_mode
        self.candidate_labels = candidate_labels
        self.y = y
        self.param_check()
        df = df.loc[:, text_columns_def]
        for col in text_columns_def:
            df[col] = df[col].apply(str)
        if self.task == 'zero-shot-classification':
            self.pipeline = pipeline(self.task, model=self.zero_shot_model)
            return
        if self.use_tokenizer:
            self.fit_tokenizers(df)
        if self.embedding_mode != 'Bert':
            for column in self.text_columns_def:
                df[f'{column}_tokenized_ids'] = self.tokenize(df, column)
        self.fit_embeddings(df)
        return self.fit_encoders(df, y)

    def param_check(self):
        if Glove_installed is False and self.embedding_mode == 'Glove':
            raise NotImplementedError("Your environment is not support to install glove-python-binary, \
            so the 'embedding_mode=Glove' method is not supported")

    def fit_tokenizers(self, df):
        raw_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        raw_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
        raw_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(vocab_size=500, special_tokens=special_tokens)

        def micro_tokenizer(text_df, name):
            text = pd.concat([text_df])
            dataset = DataSet.from_pandas(text)

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
                model = Word2Vec(vector_size=self.emb_size, min_count=-1)
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
                self.corpus_model.update({name: corpus_model})

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
                model.to(self.device)
                model_path = f'./{name}_transformer'
                train_dataset = LineByLineTextDataset(
                    tokenizer=self.tokenizers[name],
                    file_path="text.txt",
                    block_size=128)
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizers[name], mlm=True, mlm_probability=0.15)
                training_args = TrainingArguments(
                    output_dir=model_path,
                    num_train_epochs=self.mlm_epochs,
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
        raw_df[name] = raw_df[name].apply(str)
        embedding = self.embeddings[name]
        embedded_text = None
        if self.embedding_mode == 'TFIDF':
            embedded_text = embedding.transform(self.tokenize(raw_df, name))
        elif self.embedding_mode == 'FastText':
            embedded_text = sparse.csr_matrix(embedding.wv[
                                                  list(map(lambda x: tokens2sentence(x), self.tokenize(raw_df, name)))
                                              ])
        elif self.embedding_mode == 'Word2Vec':
            text_df = self.tokenize(raw_df, name)
            embedding.build_vocab(text_df)
            embedding.train(
                text_df, epochs=embedding.epochs,
                total_examples=embedding.corpus_count, total_words=embedding.corpus_total_words,
            )
            embedded_text = sparse.csr_matrix(np.array(
                list(map(lambda x: get_words_w2v_vec(embedding, x), text_df))
            ))
        elif self.embedding_mode == 'Glove':
            text_df = self.tokenize(raw_df, name)
            corpus_model = self.corpus_model[name]
            corpus_model.fit(text_df, window=10, ignore_missing=False)
            embedding.fit(corpus_model.matrix, epochs=10)
            embedding.add_dictionary(corpus_model.dictionary)
            embedded_text = sparse.csr_matrix(np.array(
                list(map(lambda x: get_words_glove_vec(embedding, x), text_df))
            ))
        elif self.embedding_mode == 'Bert':
            model = AutoModel.from_pretrained(self.embeddings[name], output_hidden_states=True)
            encoded_input = TrainDataset(self.tokenizers[name], raw_df[name])
            embedded_text = sparse.csr_matrix(model_infer(model, encoded_input, self.device))
        else:
            raise NotImplementedError
        return embedded_text

    def fit_encoders(self, df, y=None):
        if self.task == 'embedding':
            res_dict = {}
            for column in self.text_columns_def:
                print(f'Updating column: {column} embeddings output')
                res_dict.update({column: self.emb_text(df, column)})
            return res_dict

        elif self.task == 'supervise' or self.task == 'unsupervise':
            def micro_regressor(embedding_array, y):
                regressor = Ridge(random_state=42, alpha=0.8)
                regressor.fit(embedding_array, y)
                return regressor

            def micro_cluster(embedding_array):
                cluster = KMeans(n_clusters=self.n_clusters)
                cluster.fit(embedding_array)
                return cluster

            for column in self.text_columns_def:
                print(f'Fitting column: {column} encoder')
                encoders = []
                folds = KFold(n_splits=5, shuffle=True, random_state=42)
                for fold_n, (train_index, valid_index) in enumerate(folds.split(df)):
                    if self.embedding_mode != 'Bert':
                        trn = df[f'{column}_tokenized_ids'][train_index]
                        vld = df[f'{column}_tokenized_ids'][valid_index]
                    else:
                        trn = TrainDataset(self.tokenizers[column], df[column][train_index])
                        vld = TrainDataset(self.tokenizers[column], df[column][valid_index])
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
                        trn = sparse.csr_matrix(model_infer(model, trn, self.device))
                        vld = sparse.csr_matrix(model_infer(model, vld, self.device))
                    else:
                        raise NotImplementedError

                    if self.task == 'supervise':
                        y_trn = y.iloc[train_index]
                        encoders.append(micro_regressor(trn, y_trn))
                    else:
                        encoders.append(micro_cluster(trn))

                    val = encoders[fold_n].predict(vld)
                    df.loc[valid_index, f"{column}_meta_feature"] = val
                self.encoders.update({column: encoders})
                if self.embedding_mode != 'Bert':
                    df = df.drop(columns=[f'{column}_tokenized_ids'])
            return df.drop(columns=self.text_columns_def)

    def transform(self, df):
        df = df.loc[:, self.text_columns_def]
        if self.task == 'embedding':
            res_dict = {}
            for column in self.text_columns_def:
                print(f'Updating column: {column} embeddings output')
                res_dict.update({column: self.emb_text(df, column)})
            return res_dict
        if self.task == 'supervise' or self.task == 'unsupervise':
            for column in self.text_columns_def:
                print(f'Transforming column: {column}')
                embedded_text = self.emb_text(df, column)
                meta_test = None
                if self.task == 'supervise':
                    for idx in range(5):
                        encoder = self.encoders[column][idx]
                        pred = (encoder.predict(embedded_text)) / 5
                        if idx == 0:
                            meta_test = pred
                        else:
                            meta_test += pred
                else:
                    for idx in range(5):
                        encoder = self.encoders[column][idx]
                        pred = np.eye(self.n_clusters)[encoder.predict(embedded_text)]
                        if idx == 0:
                            meta_test = pred
                        else:
                            meta_test += pred
                    meta_test = np.argmax(meta_test, axis=1)
                #                     meta_test = pd.DataFrame(np.eye(self.n_clusters)[np.argmax(meta_test,axis=1)])
                #                     for idx in range(self.n_clusters):
                #                         df[f'{column}_transformed_class{idx}'] =  meta_test[idx]
                df[f'{column}_meta_feature'] = meta_test
                df.drop(columns=[column])
            return df
        elif self.task == 'zero-shot-classification':
            for column in self.text_columns_def:
                pred_labels = []
                classifier = self.pipeline
                candidate_labels = self.candidate_labels[column]
                for text in df[column]:
                    text = text[:128]
                    results = classifier(text, candidate_labels)
                    pred_labels.append(results['labels'][np.argmax(results['scores'])])
                df[f'{column}_meta_feature'] = pred_labels
                df.drop(columns=[column])
            return df
        else:
            raise NotImplementedError
