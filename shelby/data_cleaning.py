import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TypesChecker(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def transform(self, data):
        for col in self.num_cols:
            data[col] = data[col].astype('float32')

        for col in self.cat_cols:
            data[col] = data[col].astype('O')

        return data

    def fit(self, *_):
        return self


class CatNanFiller(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, method='top'):
        self.method = method
        self.cat_cols = cat_cols

    def transform(self, data):
        if self.method == 'top':
            for col in self.cat_cols:
                data[col] = data[col].fillna(data[col].describe()['top'])
        else:
            for col in self.cat_cols:
                data[col] = data[col].fillna('NO_VALUE')

        return data

    def fit(self, *_):
        return self


class NumNanFiller(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, method='mean'):
        self.method = method
        self.num_cols = num_cols

    def transform(self, data):
        if self.method == 'mean':
            for col in self.num_cols:
                data[col] = data[col].fillna(data[col].mean())

        else:
            for col in self.num_cols:
                data[col] = data[col].fillna(data[col].median())

        return data

    def fit(self, *_):
        return self