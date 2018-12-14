"""
Load model stored from path and generate predictions
"""

import os

import pandas
from sklearn.externals import joblib

from src.settings import MODELS_PATH, PROCESSED_DATA_PATH, OUTPUT_DATA_PATH


def predict_results(test_file):
    path = os.path.join(PROCESSED_DATA_PATH, test_file)
    df = pandas.read_csv(path).set_index('Id')
    idx = df.index
    mdl = joblib.load(os.path.join(MODELS_PATH, 'pipe.joblib'))
    results = mdl.predict_proba(df)[:, 1]
    df['Status'] = results
    df.index = idx
    df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'output.csv'))