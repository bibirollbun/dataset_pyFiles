!pip install flaml --no-index --find-links=file:/kaggle/input/flaml-2-3-3 


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import warnings
warnings.filterwarnings("ignore")

from flaml import AutoML
from metric import score
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import rankdata 
from lifelines import NelsonAalenFitter
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


def transform_target(df, time_col='efs_time', event_col='efs'):
    
    naf = NelsonAalenFitter(alpha=0.05, nelson_aalen_smoothing=False) 
    
    naf.fit(df[time_col], df[event_col])
    
    y = - naf.cumulative_hazard_at_times(df[time_col]).values
    
    return y 


train['y'] = transform_target(train, time_col='efs_time', event_col='efs') 
train.loc[train.efs == 0, 'y'] -= 0.15
train.head() 


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


%%time

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof = np.zeros(len(train))
pred_test = np.zeros(len(test))
valid_scores = [] 

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    automl = AutoML() 

    automl_settings = {'time_budget': 1800*3,  
                       'metric': 'rmse',
                       'task': 'regression',
                       'estimator_list': ['xgboost', 'catboost', 'lgbm', 'xgb_limitdepth'],
                       'ensemble': True,
                       'log_file_name': '',
                       'seed': 1,
                      } 
    
    automl.fit(x_train, y_train, **automl_settings, verbose = False) 

    # INFER OOF
    oof[test_index] = automl.predict(x_valid)
    # INFER TEST
    pred_test += automl.predict(x_test)
    
    rmse = mean_squared_error(y_valid, automl.predict(x_valid), squared=False) 
    print('Fold', i, '==> RMSE is ==>', rmse) 
    
    valid_scores.append(rmse) 

# COMPUTE AVERAGE TEST PREDS
pred_test /= FOLDS

print("#"*25)
print(f'Mean Validation RMSE= {np.mean(valid_scores):.5f}')
print(f'Std Validation RMSE= {np.std(valid_scores):.5f}') 


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.
 
It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.
 
The concordance index is a value between 0 and 1 where:
 
0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for FLAML NelsonAalenFitter =",m) 


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = pred_test
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head() 

