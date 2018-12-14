import os

import pandas
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from yellowbrick import ROCAUC

from src.models.base import rf
from src.settings import PROCESSED_DATA_PATH, MODELS_PATH, REPORTS_PATH
from src.transformers.build_features import ToCatTransformer, DataFrameOHETransformer


def train_clf(x, save=True, report=True):
    path = os.path.join(PROCESSED_DATA_PATH, x)
    df = pandas.read_csv(path).drop('Unnamed: 0', 1)
    x, y = df.drop('labels', 1), df.labels

    pipe = Pipeline([
        ('to_cat', ToCatTransformer()),
        ('ohe', DataFrameOHETransformer()),
        ('rf', rf)
    ])
    pipe.fit(x, y)

    if save:
        store(pipe)

    if report:
        produce_report(pipe, x, y)


def produce_report(pipeline, x, y):
    xtr, xte, ytr, yte = train_test_split(x, y)
    m = ROCAUC(pipeline)
    m.fit(xtr, ytr)
    m.score(xte, yte)
    m.poof(os.path.join(REPORTS_PATH, 'ROCAUC.png'))
    with open(os.path.join(REPORTS_PATH, 'results.txt'), 'w') as rep:
        cr = classification_report(yte, m.predict(xte))
        rep.write(cr)


def store(mdl):
    p = os.path.join(MODELS_PATH, 'pipe.joblib')
    joblib.dump(mdl, p)
