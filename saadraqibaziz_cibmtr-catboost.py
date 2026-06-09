# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
import copy

# label Encoding & filling missing values
from sklearn.preprocessing import LabelEncoder,StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from lifelines.utils import concordance_index

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score,cross_val_predict, GridSearchCV, RandomizedSearchCV
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold




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
        if not pd.api.types.is_numeric_dtype(submission[col]):
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


def replace_the_lowest(df, column, replace_with_index, info=False):
    smalles_list = df[column].value_counts().nsmallest(replace_with_index)
    smallest = smalles_list.index[0]
    replace_with = smalles_list.index[-1]
    col = df[column].replace(smallest, replace_with)
    
    if(info):
        print(f"befor --> {smalles_list},\n after --> {col.value_counts().nsmallest(replace_with_index-1)}")
    return col

def replace_classes_with_lower_freq(df, column, threshold, replace_with):
    category_counts = df[column].value_counts()
    low_freq_classes = category_counts[category_counts < threshold].index
    return df[column].replace(low_freq_classes, replace_with)


train_path = r"/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
train_data = pd.read_csv(train_path)  # removing ID from the dataset

y_true = train_data[["ID","efs","efs_time","race_group"]].copy()
y_pred = train_data[["ID"]].copy()

train_data.drop(labels="ID", axis=1, inplace=True)
test_path = r"/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
test_data = pd.read_csv(test_path)  # removing ID from the dataset
test_id = test_data["ID"]
efs = train_data["efs"]
efs_time = train_data["efs_time"]

test_data.drop(labels="ID", axis=1, inplace=True)


merge_df = pd.concat([train_data, test_data], ignore_index=True)
merge_df


remove_columns = ["ethnicity", "cyto_score", "mrd_hct", "efs", "efs_time"]

imputer_fill_na = ["dri_score","melphalan_dose", "hla_low_res_8", "hla_match_drb1_high", "hla_low_res_6",
                   "hla_high_res_6","hla_low_res_10",
                   "cmv_status","tce_imm_match", "hla_nmdp_6", "hla_match_c_low", "rituximab", "hla_match_drb1_low",
                    "hla_match_dqb1_low","cyto_score_detail", "conditioning_intensity", "in_vivo_tcd", "tce_match",
                    "hla_match_a_high", "hla_match_b_low", "hla_match_a_low", "gvhd_proph", "sex_match",
                    "hla_match_b_high", "comorbidity_score", "karnofsky_score", "tce_div_match","hla_match_c_high",
                    "hla_high_res_8", "vent_hist", "hla_high_res_10", "hla_match_dqb1_high"]

col_replaced_class_idx = {"dri_score":3, "hla_match_c_high":2,"hla_low_res_6":2,
                            "hla_high_res_6":2, "hla_match_dqb1_high":2, "hla_nmdp_6":2,
                            "hla_match_c_low":2, "hla_match_dqb1_low":2, "hla_match_a_high":2,
                             "hla_match_b_low":2, "hla_match_a_low":2,"hla_match_b_high":2,
                            "karnofsky_score":2,"hla_match_drb1_high":2, "hla_low_res_10":2,
                            "year_hct": 2} 

col_replaced_with_other = {"hla_high_res_8": (29, 6.0),"tbi_status":(100, "other"), "prim_disease_hct":(100, "other"),
                            "hla_high_res_10":(26,7.0),"tce_imm_match":(100, "other"), 
                            "conditioning_intensity": (100, "other"),
                           'gvhd_proph': (100, "other"), "hla_low_res_8":(25, 6.0)
                           }
think = ["vent_hist","renal_issue", "pulm_severe","prim_disease_hct", "cyto_score_detail","tce_div_match"]

special_feature_engineering = {"hepatic_severe":(["nan","Not done"], "Unknown"),
                               "peptic_ulcer":(["nan","Not done"], "Unknown"),"prior_tumor":(["nan","Not done"], "Unknown"),
                               "rheum_issue":(["nan","Not done"], "Unknown"),"hepatic_mild":(["nan","Not done"], "Unknown"),
                               "donor_related":(["nan", "Multiple donor (non-UCB)"], "other"), "cardiac":(["nan","Not done"], "Unknown"),
                               "pulm_moderate":(["nan","Not done"], "Unknown"), "psych_disturb":(["nan","Not done"], "Unknown"),
                               "diabetes":(["nan","Not done"], "Unknown"), "arrhythmia":(["nan","Not done"], "Unknown"),
                               "renal_issue":(["nan","Not done"], "Unknown"),"obesity":(["nan","Not done"], "Unknown"),
                               "pulm_severe":(["nan","Not done"], "Unknown")}

col = "peptic_ulcer"
print(train_data[col].value_counts(), train_data[col].isna().sum())
sns.countplot(x=col, data=train_data)
plt.xticks(rotation=90)
plt.show()



# 1st
# remove the columns
merge_df = merge_df.drop(labels=remove_columns, axis=1)
merge_copy = copy.deepcopy(merge_df)


# 2nd
for key, value in col_replaced_class_idx.items():
    merge_copy[key] = replace_the_lowest(merge_copy, key,value)


# 3rd
for key, value in col_replaced_with_other.items():
    merge_copy[key] = replace_classes_with_lower_freq(merge_copy, key, value[0], value[1])


# 4th
merge_copy["year_hct"] = merge_copy["year_hct"]-2000


# 5th
for key, value in special_feature_engineering.items():
    if(len(value[0])==1):
        merge_copy[key] = merge_copy[key].fillna(value[1])
    else:
        merge_copy[key] = merge_copy[key].replace(value[0][1], value[1])
        merge_copy[key] = merge_copy[key].fillna(value[1])


# 6th 
categorical = merge_copy.select_dtypes(include="object").columns
Oencoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

merge_copy[categorical] = Oencoder.fit_transform(merge_copy[categorical])

columns = merge_copy.columns[merge_copy.isna().any()].tolist()
# columns = merge_copy.columns
impute = KNNImputer(n_neighbors=5)
merge_copy[columns] = impute.fit_transform(merge_copy[columns]).round().astype(np.float64)
# merge_copy["donor_age"] = impute.fit_transform(merge_copy[["donor_age"]])
# 


# 7th
col_to_scale = ["donor_age", "age_at_hct"]

scaler = StandardScaler()
merge_copy[col_to_scale] = scaler.fit_transform(merge_copy[col_to_scale])


np.sum(merge_copy.isna().sum())


submission = merge_copy.iloc[len(train_data):].copy()
submission


df = merge_copy.iloc[:len(train_data)].reset_index(drop=True)
df


def get_result_of_metrics(ytest, ypred):
    mae = mean_absolute_error(ytest, ypred)
    rmse = mean_squared_error(ytest, ypred) **0.5
    r2 = r2_score(ytest, ypred)
    
    return mae, rmse, r2


dataset = pd.concat([df, efs, efs_time], axis=1)
xtrain, xtest, ytrain, ytest= train_test_split(dataset, dataset["efs"], test_size=0.30, random_state=42, shuffle=True)


import optuna

def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 3, 10),
        "bagging_temperature": trial.suggest_uniform("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_int("random_strength", 1, 10),
        "loss_function": "RMSE",
        "eval_metric": "AUC",
    }

    model = CatBoostRegressor(**params, verbose=0)
    model.fit(xtrain.drop(labels=["efs_time", "efs"], axis=1), ytrain, 
              eval_set=(xtest.drop(labels=["efs_time", "efs"], axis=1), ytest), early_stopping_rounds=50)
    
    return model.best_score_["validation"]["AUC"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best parameters:", study.best_params)



Variables = dataset.drop(labels=["efs_time", "efs"], axis=1).columns
FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(df))
pred_cat = np.zeros(len(df))

for i, (train_index, test_index) in enumerate(kf.split(df)):

    print(f"# Fold {i+1}")
    
    x_train = dataset[Variables].loc[train_index].copy()
    y_train = dataset['efs'].loc[train_index]    
    x_valid = dataset[Variables].loc[test_index].copy()
    y_valid = dataset['efs'].loc[test_index]
    x_test = dataset[Variables].copy()

    cat_model = CatBoostRegressor(**study.best_params, verbose=0)
    cat_model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)]
    )
    
    oof_cat[test_index] = cat_model.predict(x_valid)
    pred_cat += cat_model.predict(x_test)

pred_cat /= FOLDS


model = CatBoostRegressor(**study.best_params, verbose=0)
model.fit(xtrain.drop(labels=["efs_time", "efs"], axis=1), ytrain)
predicts = model.predict(xtest.drop(labels=["efs_time", "efs"], axis=1))
mae, rmse, r2 = get_result_of_metrics(ytest, predicts)
c_index = concordance_index(xtest['efs_time'],-predicts, xtest['efs'])
print("metrics: ", mae, rmse, r2, c_index)
print(model.predict(submission))



y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = model.predict(submission)
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head(3)

