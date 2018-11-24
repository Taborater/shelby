# -*- coding: utf-8 -*-
"""Data Preparation
(part of shelby module)

This module contains custom transformers (based on sklearn TransformerMixin) for data preparation operations:
- Data binnig for numeric columns.
- Data clipping for numeric columns.
- Categorical features dummifier.
- Categorical features label encoder.
- Data scaler for numeric columns.
- Skew remover for numeric columns.
- Array extractor.

Todo:
    ?* Move methods from __init__ doc to class doc
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class NumBinner(BaseEstimator, TransformerMixin):
    """Binnig data in specified numerical columns values using 3 uniform quantiles (from 0 to 1).

    Usage example:
        `new_cat_cols, new_num_cols, modified_df = NumBinner(columns_to_bin, cat_cols, num_cols).fit_transform(df)`
    """

    def __init__(self, columns_to_bin, cat_columns, num_columns):
        """NumBinner __init__.

        Args:
            columns_to_bin (list): list of numerical columns names in which data should be binned.
            cat_columns (list): list of categorical columns names.
            num_columns (list): list of numerical columns names.

        """
        self.columns_to_bin = columns_to_bin
        self.cat_columns = cat_columns
        self.num_columns = num_columns


    def transform(self, data):
        """Bin data in specified columns.

        Args:
            data (pandas.DataFrame): DataFrame containing data for binning.

        Returns:
            data (pandas.DataFrame): DataFrame with binned columns.

        """
        # Iterate through columns for binning.
        for col in self.columns_to_bin:

            # Generate bins using quantiles from 0 to 1 with step .25
            bins = [data[col].quantile(x/100) for x in range(0, 101, 25)]

            # Bin columns
            data[col] = pd.cut(data[col].values, bins)

            # Change column type and move it from numerical columns list to categorical columns list
            data[col] = data[col].astype('object')
            self.cat_columns.append(col)
            self.num_columns.remove(col)

        return self.cat_columns, self.num_columns, data


    def fit(self, *_):
        """Fit the transformer."""
        return self


class NumClipper(BaseEstimator, TransformerMixin):
    """Clipping data in specified numerical columns using low_quantile and high_quantile (params) as min and max values.

    Usage example:
        `df = NumClipper(columns_to_clip, low_q=.3, high_q=.97).fit_transform(df)`
    """

    def __init__(self, columns_to_clip, low_q=0.03, hight_q=0.97):
        """NumClipper __init__,

        Args:
            columns_to_clip (list): list of numerical columns names in which data should be clipped.
            low_q (float): low percentel specifies minimum value for clipping.
            high_q (float): high percentel specifies maximum value for clipping.

        """
        self.columns_to_clip = columns_to_clip
        self.low_q = low_q
        self.hight_q = hight_q


    def transform(self, data):
        """Clip data in spicified columns.

        Args:
            data (pandas.DataFrame): DataFrame containing data for binning.

        Returns:
            data (pandas.DataFrame): DataFrame with clipped data.

        """
        # Iterate  through columns to clip
        for col in self.columns_to_clip:

            # Define quantile values
            low_val = data[col].quantile(self.low_q)
            hight_val = data[col].quantile(self.hight_q)

            # Clip data
            data[[col]] = data[[col]].clip(low_val, hight_val)

        return data


    def fit(self, *_):
        """Fit the transformer."""
        return self



class CatDummifier(BaseEstimator, TransformerMixin):
    """Generating dummy features for specified categorical columns.

    Usage example:
        `dummified_df = CatDummifier(cat_cols, drop_first=True).fit_transform(df)`
    """

    def __init__(self, cat_columns, drop_first=True):
        """CatDummifier __init__.

        Args:
            cat_columns (list): list of categorical columns names.
            drop_first (bool): if true - generate k-1 one-hot columns from k categories.
        """
        self.cat_columns = cat_columns
        self.drop_first = drop_first

    def transform(self, data):
        """Dummifier categorical columns.

        Args:
            data (pandas.DataFrame): DataFrame containing data for generating dummies.

        Returns:
            data (pandas.DataFrame): DataFrame with dummified categorical columns.
        """
        if self.drop_first:
            return pd.get_dummies(data, columns=self.cat_columns, drop_first=self.drop_first)
        else:
            return pd.get_dummies(data, columns=self.cat_columns)

    def fit(self, *_):
        """Fit the transformer."""
        return self


class CatLabelEncoder(BaseEstimator, TransformerMixin):
    """Encoding categorical data in specified columns by mapping each category to number

    Usage example:
        `le_df = CatLabelEncoder(cat_cols).fit_transform(df)`
    """

    def __init__(self, cat_columns):
        """CatLabelEncoder __init__.

        Args:
            cat_columns (list): list of categorical columns names.
        """
        self.cat_columns = cat_columns


    def transform(self, data):
        """Encode categorical columns.

        Args:
            data (pandas.DataFrame): DataFrame containing data for encoding.

        Returns:
            data (pandas.DataFrame): DataFrame with encoded categorical columns.

        """
        # Iterate  through columns to encode
        for col in self.cat_columns:
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes

        return data

    def fit(self, *_):
        return self


class Scaler(BaseEstimator, TransformerMixin):
    """Scaling data in specified numerical columns with specified method

    Usage example:
        `scaled_df = Scaler(columns_to_scale, method='minmax').fit_transform(df)`
    """
    def __init__(self, columns_to_scale, method='standart_scaler'):
        """Scaler __init__.

        Args:
            columns_to_scale (list): list containing names of columns to scale.
            method (str): specifies name of scaling method (default 'standart_scaler')
            ['standart_scaler', 'minmax', 'robust'].

        """
        self.methods = ['standart_scaler', 'minmax', 'robust']
        self.columns_to_scale = columns_to_scale
        self.method = method

        # Raise assertion if passed method of scaling is unknow
        assert self.method in self.methods, f'Unknow method(use one of {self.methods})'


    def transform(self, data):
        """Scaling specified features.

        Args:
            data (pandas.DataFrame): DataFrame containing data for scaling.

        Returns:
            data (pandas.DataFrame): DataFrame with scaled numerical columns.

        """
        if self.method == 'standart_scaler':
            for col in self.columns_to_scale:
                data[col] = StandardScaler().fit_transform(data[[col]].values)

        elif self.method == 'minmax':
            for col in self.columns_to_scale:
                data[col] = MinMaxScaler().fit_transform(data[[col]].values)

        elif self.method == 'robust':
            for col in self.columns_to_scale:
                data[col] = RobustScaler().fit_transform(data[[col]].values)

        return data


    def fit(self, *_):
        """Fit the transformer."""
        return self


class SkewRemover(BaseEstimator, TransformerMixin):
    """Removing skew from data in specified numerical columns with specified method.

    Usage example:
        `nonskew_df = SkewRemover(columns_to_remove_skew, lam=.15, skew_thresh=.75, method='boxcox').fit_transform(df)`
    """
    def __init__(self, columns_to_remove_skewn, lam=.15, skew_thresh=.75, method='log'):
        """SkewRemover __init__

        Args:
            columns_to_remove_skewn (list): list containing names of columns to unskew.
            lam (float): BoxCox transformation parameter.
            skew_thresh (float): threshold for detecting skewed columns.
            method (str): specifies name of transformation method (default 'log')
            ['log', 'boxcox'].
        """
        self.methods = ['log', 'boxcox']
        self.method = method
        self.cols = columns_to_remove_skewn
        self.thresh = skew_thresh
        self.lam = lam

        # Raise assertion if passed method of transformation is unknow
        assert self.method in self.methods, f'Unknow method(use one of {self.methods})'


    def transform(self, data):
        """Remove skew from data.

        Args:
            data (pandas.DataFrame): DataFrame containing data for transformation.

        Returns:
            data (pandas.DataFrame): DataFrame with transfored data.
        """
        # Compute skew of each feature
        skewed_feats = data[self.cols].apply(lambda x: skew(x))

        # Filter features to transform by skew threshold
        skewed_feats = skewed_feats[abs(skewed_feats) > self.thresh]

        if self.method == 'log':
            for feat in skewed_feats.index:
                data[feat] = np.log1p(data[feat])

        elif self.method == 'boxcox':
            for feat in skewed_feats.index:
                data[feat] = boxcox1p(data[feat], self.lam)

        return data


    def fit(self, *_):
        """Fit the transformer."""
        return self


class ArraysExtractor(BaseEstimator, TransformerMixin):
    """Spliting stacked train and test pandas DataFrames into X_train, y_train, X_finall numpy arrays.

    It's convenient to finish preprocessing pipeline with this transformer.

    Usage example:
        `X, y, X_finall = ArraysExtractor(target_col, test_index).fit_transform(full_df)`
    """
    def __init__(self, target_col, test_index):
        """ArraysExtractor __init__.

        Args:
            target_col (str): name of target column.
            test_index (int): indicates from which index in full_df df_test values appears.
        """
        self.target_col = target_col
        self.test_index = test_index


    def transform(self, data):
        """Split data into X_train, y_train, X_finall.

        Args:
            data (pandas.DataFrame): DataFrame containing data for transformation.

        Returns:
            X (numpy ndarray): array with values from train df.
            y (numpy ndarray): array containing labels from train df.
            X_finall (numpy ndarray): array with values from test df.
        """
        X = data.iloc[:self.test_index].values
        X_finall = data.iloc[self.test_index:].values
        y = self.target_col.values

        return X, y, X_finall


    def fit(self, *_):
        """Fit the transformer."""
        return self


























