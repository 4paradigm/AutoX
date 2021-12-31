import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing


def preprocess_data(df, schema, process_label=False):
    df = df.copy()
    preprocess_dict = {}
    for fea in schema['features']:
        if fea['name'] != schema['label']:
            preprocess_dict[fea['name']] = {"type": fea['type']}
            if fea['type'] == "Category":

                preprocess_dict[
                    fea['name']]["processor"] = preprocessing.LabelEncoder()
                df[fea['name']] = preprocess_dict[
                    fea['name']]["processor"].fit_transform(
                        df[fea['name']].astype(str))
            else:
                preprocess_dict[
                    fea['name']]["processor"] = preprocessing.StandardScaler()
                df[fea['name']] = preprocess_dict[
                    fea['name']]["processor"].fit_transform(df[[fea['name']]])
    if process_label:
        label = schema['label']
        if label not in df:
            raise Exception("{} not in df!".format(label))
        else:
            preprocess_dict[label] = {"type" : "label", "processor" : preprocessing.LabelEncoder()}
            df[label] = preprocess_dict[label]["processor"].fit_transform(df[label])

    return df, preprocess_dict


class CountEndoder(object):
    def __init__(self):
        self.kv = None
    
    def fit(self, df):
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        self.kv = df.value_counts().to_dict()
    
    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        return df.map(self.kv).fillna(0)
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

def preprocess_data_for_tree(df, schema):
    df = df.copy()
    preprocess_dict = {}
    for fea in schema['features']:
        if fea['name'] != schema['label']:
            preprocess_dict[fea['name']] = {"type": fea['type']}
            if fea['type'] == "Category":
                preprocess_dict[
                    fea['name']]["processor"] = CountEndoder()
                df['count({})'.format(fea['name'])] = preprocess_dict[
                    fea['name']]["processor"].fit_transform(
                        df[fea['name']].astype(str))
                df.drop(fea['name'], axis=1, inplace=True)
    return df, preprocess_dict