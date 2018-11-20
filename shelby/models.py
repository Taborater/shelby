from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import make_scorer
from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator
import numpy as np

def model_tunner(model, X, y, cv_strategy, metric, params, verbose=False):
    if verbose:
        print(f'Get {model.__class__}\nTune params....')

    scorer = make_scorer(metric)
    gs = GridSearchCV(model, params, scoring=scorer, cv=cv_strategy,
                      n_jobs=-1)
    gs.fit(X, y)

    if verbose:
        print(f'Best score: {gs.best_score_}\nBest params: {gs.best_params_}')

    return gs.best_estimator_


def get_oof_prediction(estimator, X_train, y_train, X_test, cv):
    nfolds = cv.n_splits
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]

    oof_train = np.zeros((ntrain, ))
    oof_test = np.zeros((ntest, ))
    oof_test_skf = np.empty((nfolds, ntest))

    for i, (train_index, test_index) in enumerate(cv.split(X_train)):
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]

        estimator.fit(x_tr, y_tr)

        oof_train[test_index] = estimator.predict(x_te)
        oof_test_skf[i, :] = estimator.predict(X_test)

    oof_test = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def get_oof_array(estimators_array, X_train, y_train, X_test, cv):
    """
    Work only if X_train equal for all estimators in estimators_array.
    If not - use get_off_prediction separatelty for all estimators
    """
    nmodels = len(estimators_array)
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]

    oof_train_array = np.zeros((ntrain, nmodels))
    oof_test_array = np.zeros((ntest, nmodels))

    for i, estimator in enumerate(estimators_array):
        oof_train, oof_test = get_oof_prediction(estimator, X_train,
                                                 y_train, X_test, cv)

        oof_train_array[:, i] = oof_train.flatten()
        oof_test_array[:, i] = oof_test.flatten()

    return oof_train_array, oof_test_array


def model_validator(estimator, X, y, metric, seeds, nsplits=5, loops=50, verbose=False):
    scorer = make_scorer(metric)

    assert len(seeds) == loops, 'len(seeds) must be equal to loops'

    all_scores = []
    for i in range(loops):
        cv = KFold(nsplits, shuffle=True, random_state=seeds[i])
        scores = cross_val_score(estimator, X, y, scoring=scorer, cv=cv)
        all_scores.append(scores)

    all_scores = np.array(all_scores)
    mean_score = all_scores.mean()
    std_score = all_scores.std()

    if verbose:
        print(f'Scores mean: {mean_score}\nScores std: {std_score}')
    return all_scores, mean_score, std_score


def validate_multiple_models(estimators, X, y, metric, seeds, nsplits, loops):
    for est in estimators:
        _, mean, std = smodels.model_validator(est, X, y, metric, seeds, nsplits, loops)
        print(f'{est.__class__.__name__}\nScore mean: {mean}\nScore std: {std}\n=========')


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)




