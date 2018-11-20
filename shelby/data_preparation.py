import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class NumBinner(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_bin, cat_columns, num_columns):
        self.columns_to_bin = columns_to_bin
        self.cat_columns = cat_columns
        self.num_columns = num_columns

    def transform(self, data):
        for col in self.columns_to_bin:
            bins = [data[col].quantile(x/100) for x in range(0, 101, 25)]
            data[col] = pd.cut(data[col].values, bins)
            data[col] = data[col].astype('object')
            self.cat_columns.append(col)
            self.num_columns.remove(col)

        return self.cat_columns, self.num_columns, data

    def fit(self, *_):
        return self


class NumClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_clip, low_q=0.03, hight_q=0.97):
        self.columns_to_clip = columns_to_clip
        self.low_q = low_q
        self.hight_q = hight_q

    def transform(self, data):
        for col in self.columns_to_clip:
            low_val = data[col].quantile(low_q)
            hight_val = data[col].quantile(hight_q)

            data[[col]] = data[[col]].clip(low_val, hight_val)

        return data

    def fit(self, *_):
        return self


class NumLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_lg):
        self.columns_to_lg = columns_to_lg

    def transform(self, data):
        for col in self.columns_to_lg:
            data[col] = np.log1p(data[col].values)

        return data

    def fit(self, *_):
        return self


# Reverse
class NumExpTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_exp):
        self.columns_to_exp = columns_to_exp

    def transform(self, data):
        for col in self.columns_to_exp:
            data[col] = np.expm1(data[col])

        return data

    def fit(self, *_):
        return self



class CatDummifier(BaseEstimator, TransformerMixin):
    """
    Drop first to get k-1 dummies
    """
    def __init__(self, cat_columns, drop_first=True):
        self.cat_columns = cat_columns
        self.drop_first = drop_first

    def transform(self, data):
        if self.drop_first:
            return pd.get_dummies(data, columns=self.cat_columns, drop_first=self.drop_first)
        else:
            return pd.get_dummies(data, columns=self.cat_columns)

    def fit(self, *_):
        return self


class CatLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_columns):
        self.cat_columns = cat_columns

    def transform(self, data):
        for col in self.cat_columns:
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes

        return data

    def fit(self, *_):
        return self


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale, method='standart_scaler'):
        self.columns_to_scale = columns_to_scale
        self.method = method

    def transform(self, data):
        if self.method == 'standart_scaler':
            for col in self.columns_to_scale:
                data[col] = StandardScaler().fit_transform(data[[col]].values)

        elif self.method == 'minmax':
            for col in self.columns_to_scale:
                data[col] = MinMaxScaler().fit_transform(data[[col]].values)

        else:
            for col in self.columns_to_scale:
                data[col] = RobustScaler().fit_transform(data[[col]].values)

        return data

    def fit(self, *_):
        return self


class ArraysExtractor(BaseEstimator, TransformerMixin):
    """
    Split stacked train and test into X_train, y_train, X_finall
    """
    def __init__(self, target_col, test_index):
        self.target_col = target_col
        self.test_index = test_index

    def transform(self, data):
        X_train = data.iloc[:self.test_index].values
        X_finall = data.iloc[self.test_index:].values
        y_train = self.target_col.values

        return X_train, y_train, X_finall

    def fit(self, *_):
        return self


class DfToNdarray(BaseEstimator, TransformerMixin):
    """
    Get values from DataFrame
    """
    def __init__(self):
        pass

    def transform(self, data):
        return data.values

    def fit(self, *_):
        return self


class SkewRemover(BaseEstimator, TransformerMixin):
    def __init__(self,  columns_to_remove_skewn, lam=.15, skew_thresh=.75, method='log'):
        self.method = method
        self.cols = columns_to_remove_skewn
        self.thresh = skew_thresh
        self.lam = lam

    def transform(self, data):
        skewed_feats = data[self.cols].apply(lambda x: skew(x))
        skewed_feats = skewed_feats[abs(skewed_feats) > self.thresh]

        if self.method == 'log':
            for feat in skewed_feats.index:
                data[feat] = np.log1p(data[feat])

        else:
            for feat in skewed_feats.index:
                data[feat] = boxcox1p(data[feat], self.lam)

        return data

    def fit(self, *_):
        return self


























