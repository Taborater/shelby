# -*- coding: utf-8 -*-
"""Basics
(part of shelby package)

This module contains basic table data manipulations:
- separate_cols function for separating set of columns into two subsets - numerical columns and categorical columns.
- train_test_stack function for stacking two pandas DataFrames verticaly.

Usage example:
    ``
    # separating
    cat_cols, num_cols = shelby.basics.separate_cols(df)

    # stacking data
    full_df, target_col, test_start_index = shelby.basics.train_test_stack(df_train, df_test, 'SalePrice')
    ``

"""
import pandas as pd


def separate_cols(df, unique_thresh=10, return_probably_cat=False):
    """Separate pandas DataFrame columns set into two subsets: categorical columns and numerical columns.

    If return_probably_cat=True - return one more subset:
    probably categorical columns (when dtype is 'object' but #unique_vals(col) > unique_thres).

    It's important to use TypesCorrector from data_cleaning after this func, because some categorical columns
    may appear as 'int' or 'float' columns, type checker will fix it for given DataFrame.

    Args:
        df (pandas.DataFrame): df whose columns will be divided.
        unique_thresh (int): categorical threshold [IF #unique_vals(col) <= unique_thres THEN col is categorical].
        return_probably_cat (bool): if true then returns probably cat cols, else concat them with categorical columns.

    Returns:
        cat_cols (list): list of categorical columns.
        num_cols (list): list of numerical columns.
        [optional] probably_cat_cols (list): list of probably categorical columns.

    """

    # Define empty lists for all types of columns
    cat_cols = []
    probably_cat = []
    num_cols = []

    # Iterate through names of all columns
    for col in df.columns:

        # Case1: categorical column
        if df[col].dtype == 'O' and df[col].nunique() <= unique_thresh:
            cat_cols.append(col)

        # Case2: probably categorical column
        elif df[col].dtype == 'O' and df[col].nunique() > unique_thresh:
            probably_cat.append(col)

        # Case3: numerical column which is probably categorical
        elif df[col].dtype != 'O' and df[col].nunique() < unique_thresh:
            probably_cat.append(col)

        # Case4: numerical column
        else:
            num_cols.append(col)

    # Return 3 or 2 subsets
    if return_probably_cat:
        return cat_cols, num_cols, probably_cat

    else:
        cat_cols += probably_cat
        return cat_cols, num_cols


def train_test_stack(df_train, df_test, target_col_name):
    """Stacks two pandas DataFrames verticaly.

    Args:
        df_train (pandas.DataFrame).
        df_test (pandas.DataFrame).
        target_col_name (str): name of target column, necessary to remove target column from train.

    Returns:
        full_df (pandas.DataFrame): resulting pandas DataFrame.
        target_col (numpy ndarray): array containing df_train target column values.
        test_start_index (int): indicates from which index in full_df df_test values appears(for later separation).

    """
    full_df = pd.concat([df_train.drop(target_col_name, axis=1), df_test])
    target_col = df_train[target_col_name]
    test_start_index = df_train.shape[0]

    return full_df, target_col, test_start_index