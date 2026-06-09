pip install optbinning


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.filterwarnings('ignore')
os.environ['LGBM_LOG_LEVEL'] = '-1'

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import skew, kurtosis


import pandas as pd
import numpy as np
import random
import shap
from sklearn.compose import make_column_selector


from sklearn.metrics import accuracy_score
from optbinning import OptimalBinning


from xgboost import XGBClassifier
import xgboost as xgb
import optuna


SEED_VALUE = 32

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

pd.set_option('future.no_silent_downcasting', True)


train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"
sub_path = "/kaggle/input/playground-series-s5e12/sample_submission.csv"
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sub = pd.read_csv(sub_path)


df_train.shape, df_test.shape, df_sub.shape


target = "diagnosed_diabetes"
s_temp = df_train[target].copy()
df_train.drop(target, axis = 1, inplace = True)
df_train["train"], df_test["train"] = 1, 0
df_train[target] = s_temp.copy()
df = pd.concat([df_train, df_test])


df.drop("id", inplace = True, axis = 1)
df_train = 0
df_test = 0


## Metabolic Risk Score
df["MRS_risk"] = df["waist_to_hip_ratio"] * 0.30 + df["triglycerides"] * 0.25 + df["systolic_bp"] * 0.20 + df["hdl_cholesterol"] * 0.15 + df["bmi"] * 0.10
df["MRS_risk"] = df["MRS_risk"].astype(float)

## Pulse Pressure Risk Score Binning
df["sd_risk"] = df["systolic_bp"] / df["diastolic_bp"]

## AGE & FAMILY HISTORY INTERACTION
df["af_risk"] = df["age"] + df["family_history_diabetes"] * df["age"] * 0.15

## AGE & FAMILY HISTORY INTERACTION
df["af_risk2"] = df["age"] + df["family_history_diabetes"] * df["age"] * 0.30

## Colesterol Rate
df["chol_rate"] = df["cholesterol_total"] / df["hdl_cholesterol"]
df["chol_rate"] = df["chol_rate"].astype(float)

## Screen Sleep Rate
df["ss_rate"] = df["screen_time_hours_per_day"] / df["sleep_hours_per_day"]
df["ss_rate"] = df["ss_rate"].astype(float)
df.drop("sleep_hours_per_day", axis = 1, inplace = True)

## BMI & WAIST TO HIP RATIO
df["bw_rate"] = df["bmi"] / df["waist_to_hip_ratio"]

## AGE & PHYSICAL ACTIVITY MINUTES PER WEEK
df["ap_risk"] = df["physical_activity_minutes_per_week"] + df["physical_activity_minutes_per_week"] * df["age"] * 0.15

## AGE & PHYSICAL ACTIVITY MINUTES PER WEEK
df["ap_risk2"] = df["physical_activity_minutes_per_week"] + df["physical_activity_minutes_per_week"] * df["age"] * 0.30

## AGE & TRIGLYCERIDES
df["at_risk"] = df["triglycerides"] + df["triglycerides"] * df["age"] * 0.15

## AGE & TRIGLYCERIDES
df["at_risk2"] = df["triglycerides"] + df["triglycerides"] * df["age"] * 0.30

## History Risk Count
df["HRC_count"] = df["family_history_diabetes"] + df["hypertension_history"] + df["cardiovascular_history"]


cat_feats = make_column_selector(dtype_include = object)(df)


for feat in cat_feats:
    dict_name = feat + "_dict"
    dict_name = {}
    list = df[feat].unique().tolist()
    for value in list:
        pos = df[(df[feat] == value) & (df[target] == 1) & (df["train"] == 1)].shape[0]
        neg = df[(df[feat] == value) & (df[target] == 0) & (df["train"] == 1)].shape[0]
        dict_name[value] = pos/neg
    dict_name = dict(sorted(dict_name.items(), key = lambda item: item[1], reverse = False))
    i = 0
    for key in dict_name:
        dict_name[key] = i + 1
        i += 1
    print(dict_name)
    df[feat] = df[feat].replace(dict_name).astype(float)
    df[feat] = df[feat].astype(float)


int_feats = make_column_selector(dtype_include = "int64")(df)
df_int_2_exclude = ["family_history_diabetes", "hypertension_history", "cardiovascular_history", "train"]


## Integer Numbers
for feat in int_feats:
    if feat not in df_int_2_exclude:
        skewness = df[df["train"] == 1][feat].skew()
        kurt = kurtosis(df[df["train"] == 1][feat])
        if (abs(skewness) > 0.5) & (abs(kurt) > 3):
            print(feat, "---->", skewness," - ", kurt)


df["physical_activity_minutes_per_week"] = np.log(df["physical_activity_minutes_per_week"])


## Binning for Optimal Prediction
for feat in int_feats:
    if feat not in df_int_2_exclude:
        x, y = df[df["train"] == 1][feat], df[df["train"] == 1][target]
        optb = OptimalBinning(name = feat, dtype = "numerical", solver = "cp", max_n_prebins = 10, min_prebin_size = 0.05, time_limit = 50)
        optb.fit(x, y)
        df[f'{feat}_bin_label'] = optb.transform(df[feat], metric="bins")
        norm_counts = df[f'{feat}_bin_label'].value_counts(normalize = True)
        df[feat] = df[f'{feat}_bin_label'].map(norm_counts)
        df = df.drop(columns=[f'{feat}_bin_label'])


num_feats = make_column_selector(dtype_include = "float64")(df)
df_num_2_exclude = ["diagnosed_diabetes", "HRC_count"]


## Integer Numbers
for feat in num_feats:
    if feat not in df_num_2_exclude:
        skewness = df[df["train"] == 1][feat].skew()
        kurt = kurtosis(df[df["train"] == 1][feat])
        if (abs(skewness) > 0.5) & (abs(kurt) > 3):
            print(feat, "---->", skewness," - ", kurt)


df["ap_risk"] = np.log(df["ap_risk"])
df["ap_risk2"] = np.log(df["ap_risk2"])


target = "diagnosed_diabetes"
total_feats = df.columns.tolist()
total_feats.remove(target)
total_feats.remove("train")
df_train_x = df[df["train"] == 1][total_feats]
df_train_y = df[df["train"] == 1][target]
df_test_x = df[df["train"] == 0][total_feats]
df_train_x.shape, df_train_y.shape


cols_max = df_train_x.max(axis = 0)
cols_min = df_train_x.min(axis = 0)


df_train_x_norm = (df_train_x - cols_min) / (cols_max - cols_min)
df_test_x_norm = (df_test_x - cols_min) / (cols_max - cols_min)


df_train_x_norm.shape, df_test_x_norm.shape, df_train_y.shape


df_train = pd.concat([df_train_x_norm, df_train_y], axis = 1)
df_test = df_test_x_norm


df_train['activity_bins'] = pd.cut(df_train['physical_activity_minutes_per_week'], bins = 5, labels = False)


df_train['stratify_key'] = (df_train['diagnosed_diabetes'].astype(str) + "_" + df_train['family_history_diabetes'].astype(str) + "_" + df_train['activity_bins'].astype(str))


X_train, X_val, y_train, y_val = train_test_split(df_train.drop(columns=['diagnosed_diabetes', 'stratify_key', 'activity_bins']),
                                                  df_train['diagnosed_diabetes'], test_size = 0.2, random_state = 42, stratify = df_train['stratify_key'])


print(f"Stratifying Complete. df Train: {len(X_train)}, df Val: {len(X_val)}")


len(X_train.columns.tolist()), len(X_val.columns.tolist()), y_train.shape, y_val.shape


train_feats = X_train.columns.tolist()


## LGBM
weak_feats = ["alcohol_consumption_per_week", "cholesterol_total", "screen_time_hours_per_day", "diastolic_bp", "systolic_bp", "smoking_status", "income_level",
              "education_level", "ethnicity", "gender", "triglycerides", "hdl_cholesterol", "hypertension_history", "af_risk2", "sd_risk", "MRS_risk",
              "cardiovascular_history", "employment_status", "ss_rate", "at_risk2", "HRC_count"]


for feat in weak_feats:
    if feat not in train_feats:
        print(feat)
    else:
        train_feats.remove(feat)


X_train = X_train[train_feats]
X_val = X_val[train_feats]


data_para_df = {'feature': X_train.columns, 'importance': xgb_model.feature_importances_}
importance_df = pd.DataFrame(data_para_df).sort_values(by='importance', ascending = False)


num_features = len(importance_df)
altura_grafico = max(6, num_features * 0.25) 

plt.figure(figsize=(10, altura_grafico))
plt.barh(importance_df['feature'], importance_df['importance'], color = 'darkblue')
plt.gca().invert_yaxis()  # La más importante arriba
plt.xlabel('Importancia (Gini/Gain)')
plt.title(f'Importancia de las {num_features} variables - XGBoost')
plt.grid(axis='x', linestyle='--', alpha=0.7)


plt.tight_layout()
plt.show()

# 4. Extraer lista de importancia cero para el siguiente paso
zero_importance_xgb = importance_df[importance_df['importance'] == 0]['feature'].tolist()
print(f"Se encontraron {len(zero_importance_xgb)} variables con importancia cero.")


# Esto muestra todas las variables con una barrita visual al lado del número
importance_df.style.background_gradient(cmap='YlOrRd', subset=['importance'])


weak_feats = ["at_risk", "ap_risk2", "af_risk2", "gender", "education_level", "ethnicity", "HRC_count", "hdl_cholesterol",
              "diastolic_bp", "triglycerides", "employment_status", "cholesterol_total", "alcohol_consumption_per_week",
              "hypertension_history", "smoking_status", "income_level", "systolic_bp"]


%%time
def objective(trial):
    xgb_params = {
        'verbosity': 0,
        'objective': 'binary:logistic', 
        'eval_metric': 'logloss',
        'tree_method': 'hist',          
        'device': 'cuda',
 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50), 
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),      
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0)
    }

    xgb_model = XGBClassifier(**xgb_params, early_stopping_rounds = 100)

    xgb_model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = False)

    y_preds = xgb_model.predict(X_val)
    return accuracy_score(y_val, y_preds)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective, n_trials=50)

print("Mejores parámetros XGB:", study_xgb.best_params)
print("Mejor Accuracy XGB:", study_xgb.best_value)

















y_preds = xgb_model.predict(df_test_x_norm[train_feats])
y_preds.shape


# Create the histogram
plt.hist(y_preds, bins = 30, edgecolor = 'black') # 'bins' controls the number of bars, 'edgecolor' adds distinction
plt.title('Distribution of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


df_sub["diagnosed_diabetes"] = y_preds
df_sub["diagnosed_diabetes"].value_counts()


df_sub.to_csv("submission_XII.csv", index = False)




