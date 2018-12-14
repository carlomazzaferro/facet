"""
Generate train and test datasets, doin
"""

import pandas
from imblearn.over_sampling import SMOTE


def create_datasets(input_file: str, train_output_file: str, test_output_file: str):
    df = pandas.read_csv(input_file)
    x_tr, y_tr = df.dropna(subset=['Status']).drop(['Status', 'Id'], 1), df.Status.dropna()
    train_df = resample(x_tr, y_tr)
    train_df.to_csv(train_output_file)
    x_predict = df[df.Status.isna()].set_index('Id').drop(['Status'], 1)
    x_predict.to_csv(test_output_file)


def resample(x, y):
    """ Resampling strategy, as explained in the notebook """
    x_, y_ = SMOTE().fit_resample(x, y)
    x_df = pandas.DataFrame(x_, columns=x.columns)
    x_df['labels'] = pandas.Series(y_)
    return x_df
