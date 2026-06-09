from pathlib import Path
WHEELS = Path("/kaggle/input/autogluon-1-4-0-offline")  # <- your dataset

!pip install --no-index --find-links="{WHEELS}" \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"

!pip install --no-index --find-links="{WHEELS}" \
   "bitsandbytes>=0.46.1"

!pip install --no-index --find-links="{WHEELS}" \
   "mlforecast==0.14.0" "optuna==4.3.0"


!pip install --no-index --find-links="{WHEELS}" \
    "autogluon.timeseries"


from autogluon.tabular import TabularDataset, TabularPredictor

data_root = '/kaggle/input/playground-series-s5e10/'
train_data = TabularDataset(data_root + 'train.csv')
test_data = TabularDataset(data_root + 'test.csv')
test_X  = test_data.drop(["id"], axis=1)


TARGET = 'accident_risk'

hyperparameters = {
	#'NN_TORCH': [{}],
	'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}],
	'XGB': [{}],
	#'FASTAI': [{}],
	#'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
	#'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
}

predictor = TabularPredictor(label=TARGET).fit(train_data=train_data, presets='medium', hyperparameters=hyperparameters)



import pandas as pd
import os

sub   = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))

pred = predictor.predict(test_data)
sub[TARGET] = pred
display(sub.head())

sub.to_csv("submission.csv", index=False)

