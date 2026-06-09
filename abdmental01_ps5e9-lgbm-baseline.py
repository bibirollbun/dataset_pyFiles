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

def fe(df: pd.DataFrame, n_pca=2) -> pd.DataFrame:

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
                 fold_type='RKF')


%%time

base.X_train.head()


%%time

ParamsLGBM = {'learning_rate': 0.012013096537711447, 'max_depth': 10, 'num_leaves': 42, 'min_child_samples': 84,
 'min_child_weight': 0.006666403354184525, 'lambda_l1': 6.2410039929105645, 'lambda_l2': 9.412108273470349,
 'min_split_gain': 0.6110995969369165, 'feature_fraction': 0.886822213153831, 'bagging_fraction': 0.8727411086148172,
 'bagging_freq': 10,"objective": "regression","metric": "rmse","boosting_type": "gbdt", "n_estimators": 10000}

results_LGBM_1 = base.Train_ML(ParamsLGBM, 'LGBM', e_stop=50) # 26.4579


%%time

mp = np.clip(results_LGBM_1[1], 46.718, 206.037)

sample['BeatsPerMinute'] = mp
sample.to_csv('submission.csv', index=False)
sample.head()

