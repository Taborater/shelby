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


if __name__ == '__main__':
	print('Shelby - for tabular data processing')