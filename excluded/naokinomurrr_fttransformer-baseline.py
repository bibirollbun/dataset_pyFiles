# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
# linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/pip-install-pytorch-tabular/scikit_learn-1.6.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index
!pip install /kaggle/input/pip-install-pytorch-tabular/torchmetrics-1.5.2-py3-none-any.whl --no-index
!pip install /kaggle/input/pip-install-pytorch-tabular/pytorch_lightning-2.4.0-py3-none-any.whl --no-index
!pip install /kaggle/input/pip-install-pytorch-tabular/pytorch_tabnet-4.1.0-py3-none-any.whl --no-index
!pip install /kaggle/input/pip-install-pytorch-tabular/einops-0.7.0-py3-none-any.whl --no-index
!pip install /kaggle/input/pip-install-pytorch-tabular/pytorch_tabular-1.1.1-py2.py3-none-any.whl --no-index

!pip install /kaggle/input/lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/lifelines/formulaic-1.1.1-py3-none-any.whl
!pip install /kaggle/input/lifelines/lifelines-0.30.0-py3-none-any.whl


from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig


# Scikit-learn関連
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    make_scorer
)
import warnings
warnings.filterwarnings("ignore")


# その他
from itertools import product
from lifelines.utils import concordance_index
import pandas.api.types

import warnings
warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from sklearn.preprocessing import quantile_transform
from scipy.stats import rankdata
import torch
from category_encoders import TargetEncoder
import gc


class ParticipantVisibleError(Exception):
    pass
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
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

def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in ['int64', 'int32']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')

        elif col_type == 'object':
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])

            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Reduced memory usage: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")

    return df

def transform_quantile(data, time_col, event_col):

    time = data[time_col].values
    event = data[event_col].values

    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(-time[event == 1].reshape(-1, 1)).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_dead.min() - 0.3
    data['y'] = transformed

    return data

train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
sample = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


train = transform_quantile(train,"efs_time","efs")

features = [c for c in train.columns if c not in ["ID","race_group","efs","efs_time"]]
print(len(features))

#実質的にカテゴリ型のデータタイプを変換する
cat_cols =  ['dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'hla_match_c_high', 'hla_high_res_8', 'tbi_status', 'arrhythmia', 'hla_low_res_6', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe', 'prim_disease_hct', 'hla_high_res_6', 'cmv_status', 'hla_high_res_10', 'hla_match_dqb1_high', 'tce_imm_match', 'hla_nmdp_6', 'hla_match_c_low', 'rituximab', 'hla_match_drb1_low', 'hla_match_dqb1_low', 'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'year_hct', 'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hla_match_a_high', 'hepatic_severe', 'prior_tumor', 'hla_match_b_low', 'peptic_ulcer', 'hla_match_a_low', 'gvhd_proph', 'rheum_issue', 'sex_match', 'hla_match_b_high', 'race_group', 'comorbidity_score', 'karnofsky_score', 'hepatic_mild', 'tce_div_match', 'donor_related', 'melphalan_dose', 'hla_low_res_8', 'cardiac', 'hla_match_drb1_high', 'pulm_moderate', 'hla_low_res_10']
num_cols = ['donor_age', 'age_at_hct']
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

def preprocess(data):
    for c in num_cols:
        scaler = StandardScaler()
        data[c] = scaler.fit_transform(data[[c]])
# 欠損値の処理
    for col in cat_cols:
        data[col] = data[col].astype("category")
        if "missing" not in data[col].cat.categories:
                data[col] = data[col].cat.add_categories("missing")
        data[col] = data[col].fillna("missing")
        data[col] = data[col].astype(str)
    for col in num_cols:
        mean_value = data[col].mean()
        data[col] = data[col].fillna(mean_value)
    return data

train = preprocess(train)
test = preprocess(test)



#cat_colsL : 順位性あり（ラベルエンコーディングする）
cat_colsL = [
    'dri_score',
    'cyto_score',
    'hla_match_c_high',
    'hla_high_res_8',
    'hla_low_res_6',
    'hla_high_res_6',
    'hla_high_res_10',
    'hla_match_dqb1_high',
    'hla_nmdp_6',
    'hla_match_c_low',
    'hla_match_drb1_low',
    'hla_match_dqb1_low',
    'cyto_score_detail',
    'comorbidity_score',
    'karnofsky_score',
    'hla_low_res_8',
    'hla_low_res_10',
    'hla_match_a_high',
    'hla_match_b_low',
    'peptic_ulcer',
    'hla_match_a_low',
     'hla_match_drb1_high'
]
# cat_colsB 順位無し
cat_colsB = [
    'psych_disturb',
    'diabetes',
    'tbi_status',
    'arrhythmia',
    'graft_type',
    'vent_hist',
    'renal_issue',
    'pulm_severe',
    'prim_disease_hct',
    'cmv_status',
    'tce_imm_match',
    'rituximab',
    'prod_type',
    'conditioning_intensity',
    'ethnicity',
    'year_hct',
    'obesity',
    'mrd_hct',
    'in_vivo_tcd',
    'tce_match',
    'hepatic_severe',
    'prior_tumor',
    'gvhd_proph',
    'rheum_issue',
    'sex_match',
    'hla_match_b_high',
    'race_group',
    'hepatic_mild',
    'tce_div_match',
    'donor_related',
    'melphalan_dose',
    'cardiac',
    'pulm_moderate',
]

# cat_colsB 順位無しの分類
cat_cols_binary = [col for col in cat_colsB if 'Yes' in  train[col].dropna().unique()] #Yes or Noのカテゴリ変数
cat_cols_c = [col for col in cat_colsB if 'Yes' not in  train[col].dropna().unique()] #それ以外のやつ

print("cat_cols_binary:", cat_cols_binary)
print(cat_cols_c)

for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

#cat_colsLをラベルエンコーディング
le = LabelEncoder()
for c in cat_colsL:
    train[c] = le.fit_transform(train[c])
    test[c] = le.fit_transform(test[c])

train[cat_cols_binary] = train[cat_cols_binary].replace("Not done", "missing")
test[cat_cols_binary] = test[cat_cols_binary].replace("Not done", "missing")

for c in cat_cols_binary:
    train[c] = train[c].replace({'Yes': 1, 'No': 0, 'missing': -1})
    train[c] = train[c].astype('int8')

    test[c] = test[c].replace({'Yes': 1, 'No': 0, 'missing': -1})
    test[c] = test[c].astype('int8')



from sklearn.preprocessing import TargetEncoder
# def CE(train, test):
#     for c in cat_cols_c:
#         train[c] = train[c].astype(str)
#         test[c] = test[c].astype(str)
#         te = TargetEncoder()
#         te.fit(train[[c]],train['y'])
#         train[c] = te.transform(train[[c]])
#         test[c] = te.transform(test[[c]])
#     return train, test
# train,test = CE(train, test)


train = reduce_memory_usage(train)
test = reduce_memory_usage(test)


import psutil
import os
import time

mem = psutil.virtual_memory()
print(f"Total Memory: {mem.total / 1e9:.2f} GB")
print(f"Available Memory: {mem.available / 1e9:.2f} GB")
print(f"Used Memory: {mem.used / 1e9:.2f} GB")
print(f"使用率{mem.used*100/mem.total:.2f}%")


for c in cat_cols_c:
    train[c] = train[c].astype(str)
    te = TargetEncoder()
    te.fit(train[[c]],train["y"])                
    train[c] = te.transform(train[[c]])



train = reduce_memory_usage(train)


from datetime import timedelta


data_config = DataConfig(
    target=['y'],
    continuous_cols=num_cols,
    categorical_cols=cat_cols)

optimizer_config = OptimizerConfig(
    optimizer_params = {'weight_decay':1e-3}
)

trainer_config = TrainerConfig(
    auto_lr_find=False,
    batch_size=64,
    max_epochs=10,
    accelerator="auto",
    precision=32,
    seed=42,
    early_stopping='valid_loss',
    early_stopping_min_delta=0.0001
    
)

model_config = FTTransformerConfig(
    task="regression",
    embedding_dropout=0.1,
    learning_rate=1e-3,
    attn_feature_importance=False
)


def cv_score_infer_ft(train, test):
    features = [c for c in train.columns if c not in ["y", "ID", "efs", "efs_time"]]
    
    X = train[features]
    test_x = test[features]
    y = train["y"]

    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds_ft = np.zeros(test_x.shape[0])
    fold_scores = []
    
    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )


    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        start_time = time.time()
        print(f"Starting Fold {fold+1}...")
    
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model.fit(train=pd.concat([X_train, y_train], axis=1))
        end_time = time.time()
        print(f"Fold {fold+1} completed in {end_time - start_time:.2f} seconds")
        torch.cuda.empty_cache()

        preds_valid = model.predict(pd.concat([X_valid, y_valid], axis=1)).values.reshape(-1)
        preds_test = model.predict(test_x).values.reshape(-1)
        
        preds_ft += preds_test / kf.n_splits
        test_data = pd.concat([test_x, pd.Series([0] * len(test_x), name="y")], axis=1)
        preds_ft += preds_test / kf.n_splits

        solution = train.iloc[valid_idx][["ID", "efs", "efs_time", "race_group"]]
        submission = pd.DataFrame({'ID': train.iloc[valid_idx]["ID"], 'prediction': preds_valid})
        c_index = score(solution.copy(), submission.copy(), row_id_column_name="ID")
        print(f"Fold {fold+1} : Score: {c_index:.5f}")
        fold_scores.append(c_index)
        print(f"使用率{mem.used*100/mem.total:.2f}%")
        del X_train, X_valid, y_train, y_valid
        gc.collect()
        torch.cuda.empty_cache()
    mean_fold_scores = np.mean(fold_scores)
    print(mean_fold_scores)
    return preds_ft


# train_s = train.sample(n=1000, random_state=42)
preds_ft = cv_score_infer_ft(train, test)


sub = pd.DataFrame({
    'ID': test['ID'],
    'prediction': preds_ft
})
sub.to_csv('submission.csv')
sub.head()







