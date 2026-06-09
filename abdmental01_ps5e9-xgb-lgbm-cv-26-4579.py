%%time

import pandas as pd 
import numpy as np


%%time

SEED = 42

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase

train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

def fe(df: pd.DataFrame) -> pd.DataFrame:

    df["is_high_energy"] = (df["Energy"] > 0.7).astype(int)
    df["is_acoustic"] = (df["AcousticQuality"] > 0.5).astype(int)
    df["is_live"] = (df["LivePerformanceLikelihood"] > 0.5).astype(int)

    return df

train = fe(train)
test = fe(test)

train.head()


%%time

base = AbdBase(train_data=train, test_data=test, target_column='BeatsPerMinute',gpu=True,
                 problem_type="regression", metric="rmse", seed=SEED,
                 n_splits=5,early_stop=True,num_classes=0,cat_features=False,
                 fold_type='KF')


%%time

base.X_train.head()


%%time

ParamsXgb = {
    'n_estimators': 10000, 'max_depth': 5, 'learning_rate': 0.0030677782048625733, 'min_child_weight': 4,
    'subsample': 0.6010093554676529, 'enable_categorical': True}

results_XGB_1 = base.Train_ML(ParamsXgb,'XGB',e_stop=50)


%%time

ParamsLGBM = {'learning_rate': 0.012013096537711447, 'max_depth': 10, 'num_leaves': 42, 'min_child_samples': 84,
 'min_child_weight': 0.006666403354184525, 'lambda_l1': 6.2410039929105645, 'lambda_l2': 9.412108273470349,
 'min_split_gain': 0.6110995969369165, 'feature_fraction': 0.886822213153831, 'bagging_fraction': 0.8727411086148172,
 'bagging_freq': 10,"objective": "regression","metric": "rmse","boosting_type": "gbdt", "n_estimators": 10000}

results_LGBM_1 = base.Train_ML(ParamsLGBM, 'LGBM', e_stop=50) # 26.4579


sum([-0.2,1.2])


%%time

from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

oof_preds = results_XGB_1[0] * -0.1 + results_LGBM_1[0] * 1.1
score = rmse(base.y_train,oof_preds)
score 


%%time

test_preds = results_XGB_1[1]* -0.1 + results_LGBM_1[1] * 1.1
mp = np.clip(test_preds, 46.718, 206.037)

sample['BeatsPerMinute'] = mp
sample.to_csv('submission.csv', index=False)
sample.head()

