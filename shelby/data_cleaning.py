# -*- coding: utf-8 -*-
"""Data Cleaning
(part of shelby module)

This module contains custom transformers (based on sklearn TransformerMixin) for basic data cleaning operations:
correcting columns types, filling NaN values in numeric and categorical columns.

Usage example:
    ``
    # correct types
    correct_types_df = TypesChecker(num_cols, cat_cols).fit_transform(df)

    # fill NaNs
    cat_nans_filled_df = CatNanFiller(cat_cols, method='top').fit_transform(df)
    num_nans_filled_df = NumNanFiller(num_cols, method='mean').fit_transform(df)

    ``
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TypesCorrector(BaseEstimator, TransformerMixin):
    """Correcting pandas DataFrame columns types using predefined lists of number and categorical columns."""

    def __init__(self, num_cols, cat_cols):
        """TypeChecker __init__.

        Args:
            num_cols (list): list of numerical columns names.
            cat_cols (list): list of categorical columns names.

        """
        self.num_cols = num_cols
        self.cat_cols = cat_cols


    def transform(self, data):
        """Redefine columns types according to num_cols and cat_cols.

        Args:
            data (pandas.DataFrame): DataFrame to check.

        Returns:
            data (pandas.DataFrame): DataFrame with corrected types.

        """
        for col in self.num_cols:
            data[col] = data[col].astype('float32')

        for col in self.cat_cols:
            data[col] = data[col].astype('O')

        return data

    def fit(self, *_):
        """Fit the transformer."""
        return self



class CatNanFiller(BaseEstimator, TransformerMixin):
    """Filling NaNs in categorical columns with most frequent value of special indicator value."""

    def __init__(self, cat_cols, method='top'):
        """CatNanFiller __init__.

        Args:
            cat_cols (list): list of categorical columns names.
            method (str): specifies how to fill NaNs (default 'top')
            ['top' - fill with more frequent, 'indicator' - fill with special value].

        """
        # Define known methods list
        self.methods = ['top', 'indicator']
        self.method = method
        self.cat_cols = cat_cols

        # Raise assertion if passed method of filling NaNs is unknow
        assert self.method in self.methods, f'Unknow method(use one of {self.methods})'


    def transform(self, data):
        """Fill NaNs in categorical columns.

        Args:
            data (pandas.DataFrame): DataFrame to fill.

        Returns:
            data (pandas.DataFrame): DataFrame with filled NaNs in categorical columns.

        """
        if self.method == 'top':
            for col in self.cat_cols:
                data[col] = data[col].fillna(data[col].describe()['top'])

        elif self.method == 'indicator':
            for col in self.cat_cols:
                data[col] = data[col].fillna('NO_VALUE')

        return data


    def fit(self, *_):
        """Fit the transformer."""
        return self



class NumNanFiller(BaseEstimator, TransformerMixin):
    """Filling NaNs in numerical columns with mean, median or custom value."""

    def __init__(self, num_cols, method='mean'):
        """NumNanFiller __init__.

        Args:
            num_cols (list): ;ist of numerical columns names.
            method (str or float or int): specifies how to fill NaNs (default 'mean')
            ['mean', 'median', int, float].

        """
        # Define known methods list
        self.methods = ['mean', 'median']
        self.method = method
        self.num_cols = num_cols

        # Raise assertion if passed method of filling NaNs is unknow
        assert (self.method in self.methods) or (type(self.method) in [float, int]), f'Unknow method(use number or one of {self.methods})'


    def transform(self, data):
        """Fill NaNs in numerical columns.

        Args:
            data (pandas.DataFrame): DataFrame to fill.

        Returns:
            data (pandas.DataFrame): DataFrame with filled NaNs in numerical columns.

        """
        if self.method == 'mean':
            for col in self.num_cols:
                data[col] = data[col].fillna(data[col].mean())

        elif self.method == 'median':
            for col in self.num_cols:
                data[col] = data[col].fillna(data[col].median())

        else:
            for col in self.num_cols:
                data[col] = data[col].fillna(self.method)

        return data

    def fit(self, *_):
        """Fit the transformer."""
        return self