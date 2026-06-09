# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory|
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_ds = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv") 
test_ds = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


train_ds.describe()


train_ds


cat_cols = train_ds.select_dtypes(["object", "category"]).columns

num_cols =  []
for col in train_ds.columns:
    if col not in cat_cols and col !="id" and col !="loan_paid_back":
        num_cols.append(col)


num_cols


import matplotlib 
import seaborn as sns
import matplotlib.pyplot as plt


for col in num_cols:
    sns.boxplot(x = train_ds[col])
    plt.title(col)
    plt.show()


for col in cat_cols:
    sns.barplot(x = col, y= "loan_paid_back", data = train_ds)
    plt.title(col)
    plt.show()


ordinal_cols = ["grade_subgrade", "loan_purpose", "employment_status", "education_level"]
one_hot_cols = ["gender", "marital_status"]


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split


one_hot = OneHotEncoder(sparse = False, handle_unknown = "ignore").fit(train_ds[one_hot_cols])
ordinal = OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1).fit(train_ds[ordinal_cols])


test_ds


one_hot_train = one_hot.fit_transform(train_ds[one_hot_cols])

one_hot_train = pd.DataFrame(one_hot_train, 
                          columns=one_hot.get_feature_names_out(one_hot_cols))

train_ds = pd.concat([train_ds.drop(one_hot_cols, axis=1), one_hot_train], axis=1)

one_hot_test = one_hot.transform(test_ds[one_hot_cols])
one_hot_test = pd.DataFrame(
    one_hot_test,
    columns = one_hot.get_feature_names_out(one_hot_cols)
)

test_ds = pd.concat([test_ds.drop(one_hot_cols,axis = 1),one_hot_test],axis= 1)


train_ds[ordinal_cols] = ordinal.transform(train_ds[ordinal_cols])
test_ds[ordinal_cols] = ordinal.transform(test_ds[ordinal_cols])


train_ds.describe()


for col in num_cols:
    sns.boxplot(x = train_ds[col])
    plt.title(col)
    plt.show()


scaler = RobustScaler().fit(train_ds[num_cols])

train_ds[num_cols] = scaler.transform(train_ds[num_cols])
test_ds[num_cols] = scaler.transform(test_ds[num_cols])


for col in num_cols:
    sns.boxplot(x = train_ds[col])
    plt.title(col)
    plt.show()


train_ds.columns


X  = train_ds[['annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate', 'education_level', 'employment_status',
       'loan_purpose', 'grade_subgrade', 'gender_Female',
       'gender_Male', 'gender_Other', 'marital_status_Divorced',
       'marital_status_Married', 'marital_status_Single',
       'marital_status_Widowed']]
y = train_ds["loan_paid_back"]

X_train, X_val, y_train, y_val = train_test_split(X,y ,test_size = 0.1, shuffle =True, random_state = 67)


!pip install LazyPredict


from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor            


from lazypredict.Supervised import LazyRegressor

clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=roc_auc_score)
models, predictions = clf.fit(X_train[:10000], X_val[:10000], y_train[:10000], y_val[:10000])

print(models)


xgb_model = XGBRegressor(objective =  'binary:logistic',
    eval_metric ='auc',
    max_depth= 7,
    colsample_bytree=  0.8,
    subsample = 0.8,
    n_estimators = 5000,
    learning_rate = 0.01,
    random_state = 42,
    n_jobs = -1,
    device= 'cuda',
    enable_categorical=True,

    scale_pos_weight =0.8, # usefull for unbalanced data
    min_samples_split = 5,
    alpha = 2.5,
    max_bin = 512).fit(X, y)


log_params =  {
            "objective": "binary",
            "eval_metric": "auc",
            "device": "gpu",
            "learning_rate": 0.01,
            "n_estimators": 5000,
            "max_depth": 7,
            "subsample": 0.90,
            "colsample_bytree": 0.60,
            "reg_lambda": 1.25,
            "reg_alpha": 0.001,
            "verbosity": -1,  # suppress verbose LGBM logs
            "random_state": 42,
}
lgb_model = LGBMRegressor(**log_params).fit(X, y)


roc_auc = roc_auc_score(y_val, lgb_model.predict(X_val))
print("ROC AUC:", roc_auc)


test_X = test_ds.drop("id",axis = 1)


xgb_preds = xgb_model.predict(test_X)
lgb_preds = lgb_model.predict(test_X)


final_preds = xgb_preds *0.6 + lgb_preds *0.4


sample_sub.loan_paid_back = final_preds


sample_sub.loc[sample_sub["loan_paid_back"] <0, "loan_paid_back"] = 0


sample_sub.to_csv("f1.csv",index = False)


sample_sub


sample_sub.loc[]

