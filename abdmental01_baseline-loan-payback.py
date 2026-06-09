%%time

import pandas as pd 
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase
SEED = 42


%%time

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
original = pd.read_csv("/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv")

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

COLS = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate','gender','marital_status',
 'education_level','employment_status','loan_purpose','grade_subgrade']

ORIG = []

for col in COLS:
    mean_map = original.groupby(col)['loan_paid_back'].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    new_count_col_name = f"orig_count_{col}"
    count_map = original.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

FEATURES = COLS + ORIG

train.head()


%%time

from sklearn.metrics import roc_auc_score

def ROC_AUC(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba)

CATS = ['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']

encode_c = {'cat_c': CATS}

base = AbdBase(train_data=train[FEATURES + ['loan_paid_back']], test_data=test[FEATURES], target_column='loan_paid_back',gpu=True, prob=True, test_prob=True,
                 problem_type="classification", metric="custom", seed=SEED,ohe_fe=False,ordinal_encoder=encode_c,
                 n_splits=5,early_stop=True,num_classes=2,cat_features=False,custom_metric=ROC_AUC,
                 fold_type='SKF')


%%time

ParamsXgb = {'n_estimators': 50000,'max_depth': 3, 'learning_rate': 0.012316885641402132, 'min_child_weight': 5}

results_XGB = base.Train_ML(ParamsXgb,'XGB',e_stop=50)


%%time

def save_outputs(base_file_name, oof, pred):
    oof_df = pd.DataFrame(oof)
    pred_df = pd.DataFrame(pred)

    oof_df.to_csv(f"{base_file_name}_OOF.csv", index=False)
    pred_df.to_csv(f"{base_file_name}_PREDS.csv", index=False)

save_outputs('XGB_0.9255',results_XGB[0], results_XGB[1])

sample['loan_paid_back'] = results_XGB[1]
sample.to_csv('submission.csv', index=False)
sample.head() 

