!python -m pip install -qq --no-index --find-links=/kaggle/input/library-for-cibmtr \
autogluon \
lifelines


from metric import score

from autogluon.tabular import TabularDataset, TabularPredictor

import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import TargetEncoder, LabelEncoder, OneHotEncoder, RobustScaler

from xgboost import XGBRegressor

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from xgboost import XGBRegressor
import xgboost as xgb

import time
import random

from eda_utility_library import categorize_columns, plot_pie_charts, violin_plots, missing_data_summary



pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# Read the input
train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


# C-index utility to use with valid
def c_index(valid, preds):
    y_true = valid[['ID', 'efs', 'efs_time', 'race_group']].copy()
    y_pred = valid[['ID']].copy()
    y_pred['prediction'] = preds
    
    m = score(y_true, y_pred, 'ID')
    return m


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    """
    Transform using survival probability estimates
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    # Get survival probabilities at each time point
    y = kmf.survival_function_at_times(df[time_col]).values
    
    # Adjust for censoring
    # censored_mask = df[event_col] == 0
    #y[censored_mask] = y[censored_mask] * 1.2  # Increase survival prob for censored
    
    return y

train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')


RMV = ['ID', 'efs', 'efs_time', 'y']
BASIC = list(set(train.columns) - set(RMV))
print(f'There are {len(BASIC)} basic features: {BASIC}')


# Print the column types that exist in the given data
categorized_columns = categorize_columns(train, rmv=RMV)
for col_type in categorized_columns.keys():
    if len(categorized_columns[col_type]) > 0:
        print(col_type)


CATEGORICAL = categorized_columns['categorical']
DISCRETE = categorized_columns['discrete']
CONTINUOUS = categorized_columns['continuous']


for col in CATEGORICAL:
    train[col].fillna('NAN', inplace=True)
    test[col].fillna('NAN', inplace=True)


# Train OHE
ohe = OneHotEncoder(handle_unknown='error', sparse_output=False)
dummy = ohe.fit_transform(train[CATEGORICAL+DISCRETE])
OHE_COLUMNS = ohe.get_feature_names_out()

# Apply OHE to train and test
train[OHE_COLUMNS] = dummy
test[OHE_COLUMNS] = ohe.transform(test[CATEGORICAL+DISCRETE])


te = TargetEncoder(random_state=42)

train_te = te.fit_transform(train[BASIC], train['y'])
TE_COLUMNS = ['TE_' + col for col in BASIC]

train[TE_COLUMNS] = train_te
test[TE_COLUMNS] = te.transform(test[BASIC])


# Subset into lower than the median and upper than the median
lower = train[train['y'] < train['y'].median()]
upper = train[train['y'] >= train['y'].median()]

train['lower'] = train['y'] < train['y'].median()


clf_FEATURES = list(OHE_COLUMNS)+DISCRETE+list(TE_COLUMNS)+['lower']
regr_FEATURES = list(OHE_COLUMNS)+DISCRETE+list(TE_COLUMNS)+['y']


clf = TabularPredictor(label='lower')
clf.fit(train[clf_FEATURES], 
        time_limit=60)


lower_model = TabularPredictor(label='y')
lower_model.fit(lower[regr_FEATURES], 
                time_limit=60)


upper_model = TabularPredictor(label='y')
upper_model.fit(upper[regr_FEATURES], 
                time_limit=60)


clf_FEATURES.remove('lower')
regr_FEATURES.remove('y')


lower_idx = clf.predict(test[clf_FEATURES])
upper_idx = lower_idx == False


lower_IDs = test[lower_idx]['ID']
lower_preds = lower_model.predict(test[lower_idx][regr_FEATURES])

upper_IDs = test[upper_idx]['ID']
upper_preds = upper_model.predict(test[upper_idx][regr_FEATURES])


df_lower = pd.DataFrame({'ID': lower_IDs, 'prediction': lower_preds})
df_upper = pd.DataFrame({'ID': upper_IDs, 'prediction': upper_preds})

ss = pd.concat([df_lower, df_upper], ignore_index=True)

ss.to_csv('submission.csv', index=False)

