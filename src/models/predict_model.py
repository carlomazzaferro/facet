import os

import pandas
from sklearn.externals import joblib
from src.settings import MODELS_PATH, PROCESSED_DATA_PATH, OUTPUT_DATA_PATH


def predict_results(test_file):
    path = os.path.join(PROCESSED_DATA_PATH, test_file)
    df = pandas.read_csv(path).drop('Unnamed: 0', 1)
    mdl = joblib.load(os.path.join(MODELS_PATH, 'pipe.joblib'))
    results = mdl.predict(df)
    df['Status'] = results
    df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'output.csv'))