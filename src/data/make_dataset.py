# -*- coding: utf-8 -*-

import pandas
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder


def create_datasets(input_file: str, train_output_file: str, test_output_file: str):
    df = pandas.read_csv(input_file)
    x_tr, y_tr = df.dropna(subset=['Status']).drop(['Status', 'Id'], 1), df.Status.dropna()
    train_df = resample(x_tr, y_tr)
    train_df.to_csv(train_output_file)
    x_predict = df[df.Status.isna()].drop(['Status', 'Id'], 1)
    x_predict.to_csv(test_output_file)


def to_cats(df: pandas.DataFrame):
    for col in df.columns:
        if col.startswith('Type'):
            df[col] = df[col].apply(lambda x: int(x * 10))
    return df


def resample(x, y):
    x_, y_ = SMOTE().fit_resample(x, y)
    x_df = pandas.DataFrame(x_, columns=x.columns)
    x_df['labels'] = pandas.Series(y_)
    return x_df


def ohe(x):
    enc = OneHotEncoder(categories='auto')
    return pandas.DataFrame(enc.fit_transform(x).todense())
