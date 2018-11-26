"""Shelby package
Modules:
	- basics.
	- pull_push_data.
	- data_cleaning.
	- data_preparation.
	- modeling.

Dependencies:
	- numpy [ndarray and some statistics evaluating].
	- pandas [DataFrames].
	- scipy [skew calculations, transformations (BoxCox)].
	- sklearn [estimators/transformers mixins, metrics, cross-validation, scalers and etc].
	- os [os.path mostly].

Asperities:
	Sometimes using shelby along with xgboost/lightgbm in jupyther crashes kernel.
	To solve this problem use next lines:

		import os
		os.environ['KMP_DUPLICATE_LIB_OK']='True'
"""
from shelby.basics import separate_cols, train_test_stack
from shelby.pull_push_data import read_data, write_data
from shelby.data_cleaning import CatNanFiller, NumNanFiller, TypesCorrector
from shelby.data_preparation import ArraysExtractor, CatDummifier, CatLabelEncoder, \
									MinMaxScaler, NumBinner, NumClipper, Scaler, \
									SkewRemover

from shelby.modeling import grid_search_model_tunner, get_oof_prediction, get_oof_array, \
 							model_validator, validate_multiple_models, AveragingModels


if __name__ == '__main__':
	print('Shelby - for tabular data processing')