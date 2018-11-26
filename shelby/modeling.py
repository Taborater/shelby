# -*- coding: utf-8 -*-
"""Modeling
(part of shelby module)

This module contains functions and custom estimators (based on sklearn Mixins) for easy access to ML evaluations:
- Function for model tunning based on grid search.
- Function for generating out-of-fold predictions for one estimator.
- Function for generating out-of-fold predictions array for multiple estimators.
- Function for model n_loops x n_folds validation.
- Function for validation multiple models.
- Averaging Models estimator.

Todo:
    * Extend model_validator with different cv strategies (stratified and etc.).
    * Use model_validator as grid_search_model_tunner cv_strategy.
"""

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import make_scorer
from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


def grid_search_model_tunner(estimator, X, y, cv_strategy, metric, params, verbose=False):
    """Tune given model with given parameters using specified cv strategy.

    Args:
        estimator (sklearn estimator): model to tune.
        X (numpy ndarray): array on which model tunes.
        y (numpy ndarray): labels corresponding to X.
        cv_strategy (sklearn _split class): specifies cross validation split type.
        params (dict): dict of params.
        verbose (bool): if True - print logs.

    Returns:
        (sklearn estimator): tunned model instance.
    """
    if verbose:
        print(f'Get {estimator.__class__.__name__}\nTune params....')

    # Make scorer fot GridSearchCV
    scorer = make_scorer(metric)
    gs = GridSearchCV(estimator, params, scoring=scorer, cv=cv_strategy,
                      n_jobs=-1)
    gs.fit(X, y)

    if verbose:
        print(f'Best score: {gs.best_score_}\nBest params: {gs.best_params_}')

    return gs.best_estimator_


def get_oof_prediction(estimator, X_train, y_train, X_test, cv_strategy):
    """Generate out-of-fold predictions for given train and test array using given estimator.

    Args:
        estimator (sklearn estimator): model for predictions.
        X_train (numpy ndarray): train array.
        y_train (numpy ndarray): labels corresponding to X.
        X_test (numpy ndarray): test array.
        cv_strategy (sklearn _split class): specifies cross validation split type.

    Returns:
        oof_train (numpy ndarray): out-of-fold predictions for train array.
        oof_test (numpy ndarray): out-of-fold predictions for test array.

    """
    nfolds = cv_strategy.n_splits
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]

    # Init zero arrays for finall predictions
    oof_train = np.zeros((ntrain, ))
    oof_test = np.zeros((ntest, ))

    # Init empty array for oof_test_kf predictions
    # (later the result will be averaged to get oof_test)
    oof_test_kf = np.empty((nfolds, ntest))

    for i, (train_index, test_index) in enumerate(cv_strategy.split(X_train)):
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]

        estimator.fit(x_tr, y_tr)

        oof_train[test_index] = estimator.predict(x_te)

        # Predict labels for all test items
        oof_test_kf[i, :] = estimator.predict(X_test)

    # Average predictions vertically (axis=0), through folds
    oof_test = oof_test_kf.mean(axis=0)

    # Reshape to get column array
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def get_oof_array(estimators_array, X_train, y_train, X_test, cv_strategy):
    """Generate out-of-fold predictions array for multiple estimators using same X_train for
    all estimators.

    Args:
        estimators_array (array of sklearn estimators): models for predictions.
        X_train (numpy ndarray): train array.
        y_train (numpy ndarray): labels corresponding to X.
        X_test (numpy ndarray): test array.
        cv_strategy (sklearn _split class): specifies cross validation split type.

    Returns:
        oof_train_array (numpy ndarray): array of out-of-folds predictions for train of each estimator.
        oof_test_array (numpy ndarray): array of out-of-folds predictions for test of each estimator.
    """
    nmodels = len(estimators_array)
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]

    # Init zero arrays for finall predictions
    oof_train_array = np.zeros((ntrain, nmodels))
    oof_test_array = np.zeros((ntest, nmodels))

    for i, estimator in enumerate(estimators_array):
        # Use get_oof_prediction for each estimator
        oof_train, oof_test = get_oof_prediction(estimator, X_train,
                                                 y_train, X_test, cv_strategy)

        oof_train_array[:, i] = oof_train.flatten()
        oof_test_array[:, i] = oof_test.flatten()

    return oof_train_array, oof_test_array


def model_validator(estimator, X, y, metric, seeds, X_holdout=None, y_holdout=None, n_splits=5, n_loops=5, verbose=False):
    """Model n_loops x n_folds validation.

    Args:
        estimator (sklearn estimator): model to tunne.
        X (numpy ndarray): array for validation.
        y (numpy ndarray): labels corresponding to X.
        metric (func): metric func(y_pred, y_true) -> score (float).
        seeds (list / numpy ndarray): list of random seeds, for each loop of n_loops.
        X_holdout (numpy ndarray): array for holdout score (default None).
        y_holdout (numpy ndarray): labels corresponding to X_holdout (default None).
        n_splits (int): number of splits in cross-validation.
        n_loops (int): number of cross-validation loops.
        verbose (bool): if True - print logs.

    Returns:
        all_scores (numpy ndarray): n_loops x n_folds arrays of scores.
        mean_score (float): mean score.
        std_score (float): standart deviation of score.

    """
    # Make scorer from metric for cross_val_score
    scorer = make_scorer(metric)

    # Each seed is used for one iteration for random splits in each loop.
    assert len(seeds) == n_loops, 'len(seeds) must be equal to loops'

    # Init array for scores
    all_scores = []
    holdout_score = None
    for i in range(n_loops):
        cv = KFold(n_splits, shuffle=True, random_state=seeds[i])
        scores = cross_val_score(estimator, X, y, scoring=scorer, cv=cv)
        all_scores.append(scores)

    # If holdout set provided - compute score on it
    if not(X_holdout is None) and not(y_holdout is None):
        estimator.fit(X, y)
        holdout_pred = estimator.predict(X_holdout)
        holdout_score = metric(y_holdout, holdout_pred)

        if verbose:
            print(f'holdout score: {holdout_score}')

    # Create array from list and eval statistics
    all_scores = np.array(all_scores)
    mean_score = all_scores.mean()
    std_score = all_scores.std()

    if verbose:
        print(f'cv mean: {mean_score}\ncv std: {std_score}')

    return all_scores, mean_score, std_score, holdout_score



def validate_multiple_models(estimators, X, y, metric, seeds, X_holdout=None, y_holdout=None, n_splits=5, n_loops=50, verbose=False):
    """Validate multiple models using model_validator n_models times.

    Args:
        estimators (array of sklearn estimators): models to validate.
        X (numpy ndarray): array for validation.
        y (numpy ndarray): labels corresponding to X.
        metric (func): metric func(y_pred, y_true) -> score (float).
        seeds (list / numpy ndarray): list of random seeds, for each loop of n_loops.
        X_holdout (numpy ndarray): array for holdout score (default None).
        y_holdout (numpy ndarray): labels corresponding to X_holdout (default None).
        n_splits (int): number of splits in cross-validation.
        n_loops (int): number of cross-validation loops.

    Returns:
        pandas DataFrame: contains score's mean and std for each model.

    """
    # Init array for means and stds of estimator's scores
    scores_res = np.zeros((3, len(estimators)))

    for ix, est in enumerate(estimators):
        _, mean, std, holdout_score = model_validator(est, X, y, metric, seeds, X_holdout, y_holdout, n_splits, n_loops)
        if verbose:
            print(f'{est.__class__.__name__}\nScore mean: {mean}\nScore std: {std}\n=========')

        scores_res[0, ix] = mean
        scores_res[1, ix] = std
        scores_res[2, ix] = holdout_score

    # Get estimators names for finall DataFrame columns
    list_of_est_names = [est.__class__.__name__ for est in estimators]

    # Get DataFrame with score info (mean, std) and holdout score
    result_df = pd.DataFrame(data = scores_res, columns = list_of_est_names, index = ['mean', 'std', 'holdout'])

    # If holdount didn't provided - drop holdout row
    result_df.dropna(inplace=True)

    return result_df


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """Averaging models predictions. """

    def __init__(self, estimators):
        """AveragingModels __init__.

        Args:
            estimators (array of sklearn estimators): models to average.
        """
        self.estimators = estimators


    def fit(self, X, y):
        """Fit AveragingModels estimator"""

        # Train cloned base models
        for est in self.estimators:
            est.fit(X, y)

        return self


    def predict(self, X):
        """Make predictions using base models and average them."""

        # Stack predictions in columns
        predictions = np.column_stack([
            est.predict(X) for est in self.estimators
        ])

        # Average horizontaly
        return np.mean(predictions, axis=1)




