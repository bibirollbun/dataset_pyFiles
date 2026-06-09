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

train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

original['y'] = original['y'].map({'no': 0, 'yes': 1})

COLS = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome',]

def TARGET_MEAN_FEATURES_ORIGINAL(train_df, test_df, col, target='y'):
    
    new_col = f"{c}_mean_target_orig"
    target_map = original.groupby(col)[target].mean()
    mean = train['y'].mean()
    mapping_count = original[col].value_counts()
    
    train_df[f"{col}_count"] = train_df[col].map(mapping_count).fillna(0)
    test_df[f"{col}_count"] = test_df[col].map(mapping_count).fillna(0)
    
    train_df[new_col] = train_df[col].map(target_map).fillna(mean)
    test_df[new_col] = test_df[col].map(target_map).fillna(mean)
    
    return train_df, test_df

for c in COLS:
    train, test = TARGET_MEAN_FEATURES_ORIGINAL(train,test,c)

%time

def NEW_FE(df):
    
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
    df['job_edu'] = df['job'].astype(str) + "_" + df['education'].astype(str)
    df['contacted_before'] = (df['pdays'] != -1).astype(int)
    df['age_squared'] = df['age'] ** 2

    df['duration_sin'] = np.sin(2*np.pi * df['duration'] / 400)
    df['duration_cos'] = np.cos(2*np.pi * df['duration'] / 400)

    month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_map).astype('int')

    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    df.drop('month_num',axis=1,inplace=True)
    
    return df

train = NEW_FE(train)
test = NEW_FE(test)

train.head()


%%time

from sklearn.metrics import roc_auc_score

def ROC_AUC(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba)


cat_cols = ['job','marital', "education", 'contact', 'poutcome','month','default','housing','loan','job_edu']

encode_c = {'cat_c': cat_cols}

base = AbdBase(train_data=train, test_data=test, target_column='y',gpu=True, prob=True, test_prob=True,
                 problem_type="classification", metric="custom", seed=SEED,ohe_fe=False,ordinal_encoder=encode_c,
                 n_splits=7,early_stop=True,num_classes=2,cat_features=False,custom_metric=ROC_AUC,
                 fold_type='SKF')


%%time

ParamsLgb = {'n_estimators': 40000, 'learning_rate': 0.0358306214515723, 'num_leaves': 228, 'max_depth': 6,
             'min_child_samples': 83, 'subsample': 0.8700304020753131, 'colsample_bytree': 0.6169349166144594,
             'reg_alpha': 3.700714656885025, 'reg_lambda': 4.709578317972932,"objective": "binary",
             "metric": "binary_logloss"}

results_Lgb_1 = base.Train_ML(ParamsLgb,'LGBM',e_stop=150)


%%time

def save_outputs(base_file_name, oof, pred):
    oof_df = pd.DataFrame(oof)
    pred_df = pd.DataFrame(pred)

    oof_df.to_csv(f"{base_file_name}_OOF.csv", index=False)
    pred_df.to_csv(f"{base_file_name}_PREDS.csv", index=False)

save_outputs('LGBM_0.9744',results_Lgb_1[0], results_Lgb_1[1])

sample['y'] = results_Lgb_1[1]
sample.to_csv('submission.csv', index=False)
sample.head() 

