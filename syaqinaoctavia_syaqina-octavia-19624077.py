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


!pip install optuna


# ===== 1. Basic Python Tools =====
import os
import sys
import gc
import random
import time
import joblib
import warnings

# ===== 2. Data Handling =====
import optuna

# ===== 3. Visualization =====
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import missingno as msno


# ===== 4. Feature Engineering =====
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder, OneHotEncoder, FunctionTransformer
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ===== 5. Sklearn Modules =====
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, GroupKFold,
    RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, mean_squared_error, r2_score, classification_report
)

# ===== 6. Classical ML Models =====
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# ===== 7. Boosting Models =====
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor


TRAIN_PATH = "/kaggle/input/sparta-2024-data-science-competition/train.csv"
TEST_PATH = "/kaggle/input/sparta-2024-data-science-competition/test.csv"
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)


# Menampilkan jumlah dan persentase missing value
missing_df = df_train.isnull().sum().to_frame(name='missing_count')
missing_df['missing_percent'] = 100 * missing_df['missing_count'] / len(df_train)
print(missing_df[missing_df['missing_count'] > 0])


# HEATMAP


num_cols = df_train.select_dtypes(include = ["int", "float"]).columns.tolist()

plt.figure(figsize=(10, 8))
sns.heatmap(df_train[num_cols].corr(), annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi Antar Variabel Kuantitatif')
plt.show()
def plot_full_heatmap(df, figsize=(18, 14), annot=False, cmap="coolwarm"):
    # Ambil hanya kolom numerik
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    corr = df[num_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap,
                square=True, cbar_kws={"shrink": 0.7}, linewidths=0.5)
    plt.title("Full Correlation Heatmap (All Numeric Columns)", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_full_heatmap(df_train)


fig, axes = plt.subplots(nrows=len(num_cols), figsize=(5, 5 * len(num_cols)))

for i, col in enumerate(num_cols):
    axes[i].hist(df_train[col])
    axes[i].set_title(f"Histogram of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

#spacing
plt.tight_layout()

plt.show()


col_drop = [
    'name', 'description',
    'neighborhood_overview', 'host_id',
    'host_name', 'host_since',
    'host_location', 'host_about',
    'bathrooms_text', 'amenities'
]

df_train.drop(columns=col_drop, inplace=True)
df_test.drop(columns=col_drop, inplace=True)


train_direct = df_train[
    df_train['estimated_occupancy_l365d'].notnull() &
    df_train['estimated_revenue_l365d'].notnull() &
    (df_train['estimated_occupancy_l365d'] > 0) &
    (df_train['estimated_revenue_l365d'] > 0)
].copy()

train_predict = df_train[
    df_train['estimated_occupancy_l365d'].isnull() |
    df_train['estimated_revenue_l365d'].isnull() |
    (df_train['estimated_occupancy_l365d'] <= 0) |
    (df_train['estimated_revenue_l365d'] <= 0)
].copy()

test_direct = df_test[
    df_test['estimated_occupancy_l365d'].notnull() &
    df_test['estimated_revenue_l365d'].notnull() &
    (df_test['estimated_occupancy_l365d'] > 0) &
    (df_test['estimated_revenue_l365d'] > 0)
].copy()

# Data yang harus diprediksi (salah satu atau dua-duanya null)
test_predict = df_test[
    df_test['estimated_occupancy_l365d'].isnull() |
    df_test['estimated_revenue_l365d'].isnull() |
    (df_test['estimated_occupancy_l365d'] <= 0) |
    (df_test['estimated_revenue_l365d'] <= 0)
].copy()


def transform(train, test):

    persen_cols = ['host_response_rate', 'host_acceptance_rate']
    for col in persen_cols:
        test[col] = test[col].str.replace('%', '').astype(float)
        train[col] = train[col].str.replace('%', '').astype(float)

    for col in test.columns:
      if test[col].dtype == "object":
          le = LabelEncoder()
          le.fit(list(train[col].astype(str)) + list(test[col].astype(str)))
          train[col] = le.transform(train[col].astype(str))
          test[col] = le.transform(test[col].astype(str))
    return train, test

train, test = transform(train_predict, test_predict)
train_2, test_2 = transform(train_direct, test_direct)


X = train.drop(["price", "id"], axis=1)
y = train["price"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 250),
#         "max_depth": trial.suggest_int("max_depth", 5, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
#         "random_state": 42,
#         "tree_method": "hist",
#         "verbosity": 0
#     }

#     model = XGBRegressor(**params)

#     # Sampel kecil (opsional)
#     X_sample = X_train
#     y_sample = y_train

#     score = cross_val_score(model, X_sample, y_sample, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
#     return score.mean()

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=75)

# print("Best params (XGBoost):", study.best_params)
# print("Best RMSE (XGBoost):", study.best_value)


# params = study.best_params
# params = {'n_estimators': 149, 'max_depth': 8, 'learning_rate': 0.09117598265919911, 'subsample': 0.8405265574101041, 'colsample_bytree': 0.9152405414016016, 'reg_alpha': 0.3202558383381904, 'reg_lambda': 0.10852521488719769}
# params = {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.09661706418039724, 'subsample': 0.7372231358415464, 'colsample_bytree': 0.9754542040968331, 'reg_alpha': 0.1335614207838155, 'reg_lambda': 0.46065505709564925}
params = {'n_estimators': 249, 'max_depth': 10, 'learning_rate': 0.0697486159443857, 'subsample': 0.8892018954755347, 'colsample_bytree': 0.8555090586656093, 'reg_alpha': 0.14705202485580957, 'reg_lambda': 0.3553729616909302}

model = xgb.XGBRegressor(
    **params,
    random_state=42
  )

# Latih model
model.fit(X_train, y_train)

# # Prediksi
y_pred = model.predict(X_val)

# Evaluasi
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")


#### Ada Imputasi


test_2['price'] = test_2['estimated_revenue_l365d'] / test_2['estimated_occupancy_l365d']

X_test_predict = test.copy() 
X_test_predict.drop(columns= ["id"] , inplace=True)

# Prediksi
test['price'] = model.predict(X_test_predict)

df_combined = pd.concat([test, test_2])


submission = pd.DataFrame({
    'id': df_combined['id'],    
    'price': df_combined['price']
})

# Simpan ke file CSV
submission.to_csv('submission.csv', index=False)

