"""
Custom data transformers. Wrapped in the scikit-leatn api so that they can be placed inside a pipeline transformer
"""

import pandas
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder


class ToCatTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def to_cats(df: pandas.DataFrame):
            for col in df.columns:
                if col.startswith('Type'):
                    df[col] = df[col].apply(lambda x: int(x * 10))
            return to_cats(df)
        return X


class DataFrameOHETransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.ohe = None
        self.fit_est = None
        self.features = None

    def fit(self, X, y=None):
        self.ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
        self.fit_est = self.ohe.fit(X)
        return self

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)
        return self.transform(x, y)

    def transform(self, X, y=None):
        return pandas.DataFrame(self.fit_est.transform(X).todense())



