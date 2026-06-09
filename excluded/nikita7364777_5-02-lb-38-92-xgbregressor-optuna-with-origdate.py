playground_series_s5e2_path = '/kaggle/input/playground-series-s5e2'
souradippal_student_bag_price_prediction_dataset_path = '/kaggle/input/student-bag-price-prediction-dataset'
cdeotte_feature_engineering_with_rapids_lb_38_847_path ='/kaggle/input/feature-engineering-with-rapids-lb-38-847'
print('Data source import complete.')


import os
import glob

import numpy as np
import pandas as pd
# Uploading this library to Kaggle was not successful
# WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by
# 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x79afbbb38df0>:
# Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/skimpy/
#from skimpy import skim

import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import *
from sklearn.isotonic import IsotonicRegression

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


PATH_1 = playground_series_s5e2_path
PATH_2 = souradippal_student_bag_price_prediction_dataset_path
PATH_3 = cdeotte_feature_engineering_with_rapids_lb_38_847_path


train = pd.read_csv(PATH_1 + "/train.csv")


# skim(train)


train2 = pd.read_csv(PATH_1 + "/training_extra.csv")


# skim(train2)


train = pd.concat([train, train2], axis = 0, ignore_index = True)
print("Train dataset shape",train.shape)


test = pd.read_csv(PATH_1 + "/test.csv")


# skim(test)


orig = pd.read_csv(PATH_2 + "/Noisy_Student_Bag_Price_Prediction_Dataset.csv")


# skim(orig)


orig


orig['Weight Capacity (kg)'].value_counts()


orig = orig.groupby("Weight Capacity (kg)").Price.mean()
orig.name = "Orig_Price"
orig


train


train = train.merge(orig, on = "Weight Capacity (kg)", how="left")
test = test.merge(orig, on = "Weight Capacity (kg)", how="left")
train


tmp = train.groupby("Weight Capacity (kg)")[['Price','Orig_Price']].agg(["mean","count"])
tmp = tmp.iloc[:, : -1]
tmp.columns = ['Price','Count','Orig_Price']

tmp = tmp.loc[(tmp['Count']>100) & (~tmp.Orig_Price.isna())]
print(tmp.shape)
tmp.head()


tmp['Orig_Price'].min(), tmp['Orig_Price'].max()


round(np.corrcoef(tmp.Price, tmp.Orig_Price)[0, 1], 3)


~tmp.Orig_Price.isna()


plt.scatter(tmp.Orig_Price,tmp.Price,s=1)
a,b = np.polyfit(tmp.loc[~tmp.Orig_Price.isna()].Orig_Price, tmp.loc[~tmp.Orig_Price.isna()].Price, deg = 1)

x = np.arange(15,150)
y = b + (a*x)

plt.plot(x,y,'--',color='black',linewidth=3)

r = np.corrcoef(tmp.Price, tmp.Orig_Price)[0,1]

plt.xlabel("Original Dataset Price")
plt.ylabel("Synthetic Dataset Price")
plt.show()


plt.hist(tmp.loc[~tmp.Orig_Price.isna()].Price, bins=100)
plt.title("Train data Price histogram")
plt.show()


plt.hist(tmp.loc[~tmp.Orig_Price.isna()].Orig_Price,bins=100)
plt.title("Original data Price histogram")
plt.show()


orig = pd.read_csv(PATH_2 + "/Noisy_Student_Bag_Price_Prediction_Dataset.csv")

orig = orig.loc[(orig["Weight Capacity (kg)"] > 5) & (orig["Weight Capacity (kg)"] < 30)]
orig.columns = [f"orig_{c}" for c in orig.columns]

train = train.merge(orig.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")
train = train.drop("id", axis = 1)

test = test.merge(orig.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")


train.head()


test.head()


# skim(train)


# skim(test)


CATS = []
for c in train.columns:
    if train[c].dtype == 'object':
        CATS.append(c)

print(f"There are {len(CATS)} categorical columns in Train DF:")
print(CATS)

NUMS = ['Weight Capacity (kg)','Orig_Price']

print(f"There are {len(NUMS)} numerical columns in Train DF:")
print(NUMS)

FEATURES = CATS + NUMS


encoder = OneHotEncoder(sparse=False)
encoded_cats = encoder.fit_transform(train[CATS])
encoded_cats_df = pd.DataFrame(encoded_cats, columns = encoder.get_feature_names_out(CATS))
train_encoded = pd.concat([encoded_cats_df, train[NUMS].reset_index(drop=True)], axis=1)
train_encoded.head()


encoder = OneHotEncoder(sparse=False)
encoded_cats = encoder.fit_transform(test[CATS])
encoded_cats_df = pd.DataFrame(encoded_cats, columns = encoder.get_feature_names_out(CATS))
test_encoded = pd.concat([encoded_cats_df, test[NUMS].reset_index(drop=True)], axis=1)
test_encoded.head()


len(test.columns)


len(test_encoded.columns)


len(train.columns)


len(train_encoded.columns)


len(train_encoded.columns)-len(train.columns)


# Conttrol Encoding
len_un = []
for i in CATS:
    len_un.append(len(train[i].unique()))

print(f'Lengnth CATS unique values: {sum(len_un)}')
# True 64 + 2 = 66


X = train_encoded
y = train['Price']
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(test_encoded), columns = test_encoded.columns)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_train


X_test


def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int("n_estimators", 7000, 14000, step=500),
        'max_depth': trial.suggest_int("max_depth", 4, 9, step=1),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        'subsample': trial.suggest_float("subsample", 0.5, 0.95),
        'gamma': trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 0.9),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': 42
    }

    model = XGBRegressor(**xgb_params)

    # Конфигурация кросс-валидации
    cv = ShuffleSplit(n_splits=5, test_size = 0.3, random_state = 42)
    rmse_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        # Разделение данных
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Обучение и предсказание
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], early_stopping_rounds=100,verbose=False)
        y_pred = model.predict(X_fold_val)
        # Расчет RMSE
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        rmse_scores.append(fold_rmse)

    # Возвращаем средний RMSE по всем фолдам
    return np.mean(rmse_scores)


#study_1 = optuna.create_study(direction='minimize')
#study_1.optimize(objective, n_trials = 15)


best_params_1 = {'n_estimators': 9500, 'max_depth': 8, 'learning_rate': 0.007731765910270286,
                 'reg_alpha': 0.0005377607859519973, 'subsample': 0.6370599590496365, 'gamma': 0.000280784561306808,
                 'colsample_bytree': 0.8884837331590358, 'min_child_weight': 7, 'reg_lambda': 0.00046424654761538465,
                 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist', 'device': 'cuda', 'seed': 42}
final_model_1 = XGBRegressor(**best_params_1)


'''
final_model_1.fit(X, y)
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(final_model_1, max_num_features = 20, importance_type='gain', ax=ax)
plt.title("Top 20 Feature Importances XGBoost")
plt.show()
'''


# final_model_1.fit(X, y)
# y_pred = final_model_1.predict(X_test)


'''
sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = y_pred
sub.to_csv(f"submission.csv", index=False)
sub.head()
'''

