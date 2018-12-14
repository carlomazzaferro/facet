import os

import hug
import pandas
from sklearn.externals import joblib

from src.settings import MODELS_PATH


@hug.post('/predictions')
def server(body):
    df = pandas.DataFrame([body.values()], columns=body.keys())
    mdl = joblib.load(os.path.join(MODELS_PATH, 'pipe.joblib'))
    results = mdl.predict(df)
    return {'prediction': results[0]}
