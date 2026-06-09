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


#Core Libraries 
import pandas as pd
import numpy as np
import random
import warnings
from scipy import stats

#Visualization Libraries 

import matplotlib.pyplot as plt
import seaborn as sns

#machine Learning Libraries 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder ,FunctionTransformer
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import make_scorer,accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier ,HistGradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,IsolationForest
from sklearn.compose import ColumnTransformer


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


df_train.head(5)


df_test.head(5)



df_sub


df_train.info()


df_train.dtypes


df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])


df_train


pip install pydantic-settings


pip install "pydantic==1.*"


from pandas_profiling import ProfileReport

profile = ProfileReport(df_train,explorative=True,config_file="")

# Save the report as an HTML file
profile.to_file("profile_report.html")

# Or display it in a Jupyter Notebook
profile.to_notebook_iframe()


#3.1 Select the categorical and numerical columns


categorical = df_train.select_dtypes(include = ['object']).columns
numerical = df_train.select_dtypes(include = ['int64']).columns


categorical


numerical


#3.2 Find and Handaling Missing Value


categorical.isnull().sum()


for col in categorical:
    print(df_train[col].value_counts().head(10))
    print(f"Missing value  : {df_train[col].isnull().sum()}")
    print("_"*40)


numerical.isnull().sum()


for col in numerical:
    print(df_train[col].value_counts().head(10))
    print(f"Missing value  : {df_train[col].isnull().sum()}")
    print("_"*40)


# 3.3 Check the skewness of columns


print(df_train[numerical].skew())


# A skewness between -0.5 and +0.5 is generally considered approximately symmetric.
# No transformations are required based on the skew values you provided.


for col in numerical:
    sns.histplot(df_train[col],kde = True ,bins =10 ,label =col )
    plt.xlabel("values")
    plt.ylabel("Frequency")
    plt.title("skewness of numerical columns")
    plt.legend()
    plt.show()


# 3.4 Find the corr matrix for numerical columns


sns.heatmap(df_train.corr(numeric_only = True) ,annot = True)


# 3.5 Encoding the categorical columns


# Apply Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in categorical:
    df_train[col] = le.fit_transform(df_train[col])

print(df_train.head())



df_train


sns.heatmap(df_train.corr(numeric_only = True) ,annot = True)


# 3.6 Find the important features


correlation_matrix = df_train.corr(numeric_only=True)

# Extract correlation values for Depression
target_correlation = correlation_matrix["Fertilizer Name"].sort_values(ascending=False)
# Display correlation values
print("Feature correlation with Fertilizer Name")
print(target_correlation)


# Visualizing correlation using bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=target_correlation.index, y=target_correlation.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.ylabel("Correlation Coefficient")
plt.title("Feature Correlation with Fertilizer Name")
plt.show()


#4.1 Box plot 

plt.figure(figsize = (10,6 ))
sns.boxplot(data = df_train)
plt.title("Box plot for outlier Detection")
plt.show()


# 4.2.Histogram 

plt.figure(figsize = (10,6))
plt.hist(df_train ,bins= 10 , edgecolor = 'black')
plt.title("Histogram for outlier Detection")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# 3.Isolation Forest Decision Score PLot

isolation_forest  = IsolationForest(contamination = 0.04 ,random_state =42)
isolation_forest.fit(df_train)


scores = isolation_forest.decision_function(df_train)

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, edgecolor='black')
plt.title("Isolation Forest Decision Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


#  Isolation Forest decision score histogram follows a normal distribution, it generally means the dataset does not contain extreme outliers or anomalies. This can indicate that most points in your dataset are behaving consistently, without significant deviations.


# 4.4  define the outlier label -1 or 1 . if there will be outlier its 1 otherwise -1
outlier_label =isolation_forest.fit_predict(df_train)


outlier_label


non_outlier = outlier_label!=-1
non_outlier.sum()


have_outlier = outlier_label==-1
have_outlier.sum()


# 4.5 Filter out outlier from X_train Data


df_train = df_train[non_outlier]


df_train


df_train.shape
#!750000


# 4.6 After clean outlier Isolation Forest Decision Score PLot

isolation_forest  = IsolationForest(contamination = 0.04 ,random_state =42)
isolation_forest.fit(df_train)


scores = isolation_forest.decision_function(df_train)

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, edgecolor='black')
plt.title("Isolation Forest Decision Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


X = df_train.drop(['Fertilizer Name'] ,axis = 1)
y = df_train['Fertilizer Name']


X


y


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state =42,stratify=y)


X_train


from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# === Label Encode Targets ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)


y_train_enc.shape


from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import gc

# === Custom MAP@3 Function ===
def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    for i, p in enumerate(predicted):
        if p == actual:
            score = 1.0 / (i + 1)
            break
    return score

def mapk(actuals, predicteds, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)])

# === Model Parameters ===

cat_params = {
    'iterations': 200,
    'depth': 3,
    'learning_rate': 0.05,
    'loss_function': 'MultiClass',
    'eval_metric': 'TotalF1',
    'early_stopping_rounds': 50,
    'random_seed': 42,
    'verbose': False,
    'task_type': 'CPU'
}

lgb_params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.05,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1
}

xgb_params = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'min_child_weight': 50,
    'n_estimators': 300,
    'n_jobs': -1,
    'random_state': 42
}

# === Stratified K-Fold Setup ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Score Containers ===
cat_map_scores, cat_f1_scores = [], []
lgb_map_scores, lgb_f1_scores = [], []
xgb_map_scores, xgb_f1_scores = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # === CatBoost ===
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    cat_proba = cat_model.predict_proba(X_val)
    cat_top3 = np.argsort(cat_proba, axis=1)[:, -3:][:, ::-1]
    cat_preds = cat_model.predict(X_val)

    cat_mapk = mapk(y_val.values, cat_top3, k=3)
    cat_f1 = f1_score(y_val, cat_preds, average='macro')

    cat_map_scores.append(cat_mapk)
    cat_f1_scores.append(cat_f1)
    print(f"[CatBoost] Fold {fold+1} - MAP@3: {cat_mapk:.4f}, F1: {cat_f1:.4f}")

    # === LightGBM ===
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(0)
        ]
    )
    lgb_proba = lgb_model.predict_proba(X_val)
    lgb_top3 = np.argsort(lgb_proba, axis=1)[:, -3:][:, ::-1]
    lgb_preds = lgb_model.predict(X_val)

    lgb_mapk = mapk(y_val.values, lgb_top3, k=3)
    lgb_f1 = f1_score(y_val, lgb_preds, average='macro')

    lgb_map_scores.append(lgb_mapk)
    lgb_f1_scores.append(lgb_f1)
    print(f"[LightGBM] Fold {fold+1} - MAP@3: {lgb_mapk:.4f}, F1: {lgb_f1:.4f}")

    # === XGBoost ===
    xgb_model = XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    xgb_proba = xgb_model.predict_proba(X_val)
    xgb_top3 = np.argsort(xgb_proba, axis=1)[:, -3:][:, ::-1]
    xgb_preds = xgb_model.predict(X_val)

    xgb_mapk = mapk(y_val.values, xgb_top3, k=3)
    xgb_f1 = f1_score(y_val, xgb_preds, average='macro')

    xgb_map_scores.append(xgb_mapk)
    xgb_f1_scores.append(xgb_f1)
    print(f"[XGBoost] Fold {fold+1} - MAP@3: {xgb_mapk:.4f}, F1: {xgb_f1:.4f}")

    gc.collect()

# === Final Results ===
print("\n==== Final CV Results ====")
print("ðŸ”¸ CatBoost MAP@3: {:.4f} Â± {:.4f}".format(np.mean(cat_map_scores), np.std(cat_map_scores)))
print("ðŸ”¸ CatBoost F1: {:.4f} Â± {:.4f}".format(np.mean(cat_f1_scores), np.std(cat_f1_scores)))
print("ðŸ”¸ LightGBM MAP@3: {:.4f} Â± {:.4f}".format(np.mean(lgb_map_scores), np.std(lgb_map_scores)))
print("ðŸ”¸ LightGBM F1: {:.4f} Â± {:.4f}".format(np.mean(lgb_f1_scores), np.std(lgb_f1_scores)))
print("ðŸ”¸ XGBoost MAP@3: {:.4f} Â± {:.4f}".format(np.mean(xgb_map_scores), np.std(xgb_map_scores)))
print("ðŸ”¸ XGBoost F1: {:.4f} Â± {:.4f}".format(np.mean(xgb_f1_scores), np.std(xgb_f1_scores)))



y_train


y_train


print(X_test.shape)  # Should be (144000, num_features)





