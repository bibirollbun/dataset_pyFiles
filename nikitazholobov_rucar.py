import sys
sys.path.append('/kaggle/input/russian-car-plates-prices-prediction/')
from supplemental_english import *


import pandas as pd


train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")


test.head()


train_nodate = train.drop(columns = ['date'], axis=1)
test_nodate = test.drop(columns = ['date'], axis=1)


price = train_nodate['price']
train_noprice = train_nodate.drop(columns = ['price'], axis=1)
test_noprice = test_nodate.drop(columns = ['price'], axis=1)


train_noprice.head()


ids = test_noprice['id']
train_noid = train_noprice.drop(columns = ['id'], axis=1)
test_noid = test_noprice.drop(columns = ['id'], axis=1)


train_noid.head()


def extract_parts(plate):
    nums = plate[0] + plate[4] + plate[5]
    chars = plate[1] + plate[2] + plate[3]
    region = plate[6:]
    return nums, chars, region

train_separate = train_noid.copy()
test_separate = test_noid.copy()
train_separate[['chars', 'nums', 'region']] = train_separate['plate'].apply(lambda x: pd.Series(extract_parts(x)))
test_separate[['chars', 'nums', 'region']] = test_separate['plate'].apply(lambda x: pd.Series(extract_parts(x)))
train_separate = train_separate.drop(columns=['plate'], axis=1)
test_separate = test_separate.drop(columns=['plate'], axis=1)


train_separate.head()


train_sep_reg = train_separate.copy()
test_sep_reg = test_separate.copy()

for region, codes in REGION_CODES.items():
    train_sep_reg[region] = train_sep_reg['region'].isin(codes).astype(int)
    test_sep_reg[region] = test_sep_reg['region'].isin(codes).astype(int)


train_sep_reg.head()


train_gov = train_sep_reg.copy()
test_gov = test_sep_reg.copy()

def get_government_data(row):
    for (letters, num_range, region), values in GOVERNMENT_CODES.items():
        if row['chars'] == letters and num_range[0] <= int(row['nums']) <= num_range[1] and row['region'] == region:
            return values[1:]
    return (1, 0, 0)

train_gov[['forbidden_to_buy', 'road_advantage', 'significance_level']] = train_gov.apply(get_government_data, axis=1, result_type='expand')
test_gov[['forbidden_to_buy', 'road_advantage', 'significance_level']] = test_gov.apply(get_government_data, axis=1, result_type='expand')


train_gov.head()


train_sep_chars = train_gov.copy()
test_sep_chars = test_gov.copy()

train_sep_chars[['char1', 'char2', 'char3']] = train_sep_chars['chars'].apply(lambda x: [ord(c) for c in x]).apply(pd.Series)
test_sep_chars[['char1', 'char2', 'char3']] = test_sep_chars['chars'].apply(lambda x: [ord(c) for c in x]).apply(pd.Series)
train_sep_chars = train_sep_chars.drop(columns=['chars'])
test_sep_chars = test_sep_chars.drop(columns=['chars'])


train_sep_chars.head()


train_sep_chars.info()


train_nums = train_sep_chars.copy()
test_nums = test_sep_chars.copy()

train_nums['nums'] = train_nums['nums'].astype(int)
test_nums['nums'] = test_nums['nums'].astype(int)
train_nums['region'] = train_nums['region'].astype(int)
test_nums['region'] = test_nums['region'].astype(int)


train_X = train_nums.copy()
test_X = test_nums.copy()

train_X.columns = train_X.columns.str.replace(' ', '_')
test_X.columns = test_X.columns.str.replace(' ', '_')


price


import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from tqdm import tqdm

# SMAPE метрика
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

X = train_X
y = price
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
models_count = 20

xgb_preds, cat_preds, lgb_preds = [], [], []
xgb_models, cat_models, lgb_models = [], [], []  # Хранение моделей
xgb_weights, cat_weights, lgb_weights = None, None, None  # Хранение весов
w_xgb, w_cat, w_lgb = None, None, None  # Глобальные веса

# Параметры моделей
xgb_params = [
    {'n_estimators': 110, 'learning_rate': 0.1, 'max_depth': 10, 'subsample': 1.0, 'colsample_bytree': 1.0},
    {'n_estimators': 100, 'learning_rate': 0.09, 'max_depth': 9, 'subsample': 1.0, 'colsample_bytree': 1.0},
    {'n_estimators': 120, 'learning_rate': 0.11, 'max_depth': 11, 'subsample': 1.0, 'colsample_bytree': 1.0},
    {'n_estimators': 105, 'learning_rate': 0.095, 'max_depth': 8, 'subsample': 1.0, 'colsample_bytree': 1.0}
]
cat_params = [
    {'n_estimators': 110, 'learning_rate': 0.1, 'max_depth': 10, 'subsample': 1.0},
    {'n_estimators': 100, 'learning_rate': 0.09, 'max_depth': 9, 'subsample': 1.0},
    {'n_estimators': 120, 'learning_rate': 0.11, 'max_depth': 11, 'subsample': 1.0},
    {'n_estimators': 105, 'learning_rate': 0.095, 'max_depth': 8, 'subsample': 1.0}
]
lgb_params = [
    {'n_estimators': 110, 'learning_rate': 0.1, 'num_leaves': 31, 'min_child_samples': 5, 'max_depth': 10},
    {'n_estimators': 100, 'learning_rate': 0.09, 'num_leaves': 50, 'min_child_samples': 4, 'max_depth': 9},
    {'n_estimators': 120, 'learning_rate': 0.11, 'num_leaves': 70, 'min_child_samples': 5, 'max_depth': 11},
    {'n_estimators': 105, 'learning_rate': 0.095, 'num_leaves': 35, 'min_child_samples': 4, 'max_depth': 8}
]

# Обучение моделей
for i in tqdm(range(models_count), desc="Training models"):
    params = xgb_params[i % len(xgb_params)]
    xgb_model = xgb.XGBRegressor(**params, random_state=i)
    xgb_model.fit(X_train, y_train)
    xgb_models.append(xgb_model)
    xgb_preds.append(xgb_model.predict(X_val))

    params = cat_params[i % len(cat_params)]
    cat_model = cb.CatBoostRegressor(**params, random_state=i, verbose=0)
    cat_model.fit(X_train, y_train)
    cat_models.append(cat_model)
    cat_preds.append(cat_model.predict(X_val))

    params = lgb_params[i % len(lgb_params)]
    lgb_model = lgb.LGBMRegressor(**params, random_state=i, verbose=0)
    lgb_model.fit(X_train, y_train)
    lgb_models.append(lgb_model)
    lgb_preds.append(lgb_model.predict(X_val))

# Оптимизация весов внутри групп
def optimize_group(weights, preds):
    w = np.array(weights)
    w /= w.sum()
    ensemble_pred = np.sum(w[:, None] * preds, axis=0)
    return smape(y_val, ensemble_pred)

# Функция для поиска лучших весов в группе
def get_best_group_pred(preds):
    initial_weights = np.ones(len(preds)) / len(preds)
    result = minimize(optimize_group, initial_weights, args=(np.array(preds)), bounds=[(0, 1)] * len(preds), constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    best_weights = result.x / result.x.sum()
    return np.sum(best_weights[:, None] * np.array(preds), axis=0), best_weights

if xgb_weights is None:
    xgb_pred, xgb_weights = get_best_group_pred(xgb_preds)
if cat_weights is None:
    cat_pred, cat_weights = get_best_group_pred(cat_preds)
if lgb_weights is None:
    lgb_pred, lgb_weights = get_best_group_pred(lgb_preds)

print("xgb weights", xgb_weights)
print("cat weights", cat_weights)
print("lgb weights", lgb_weights)

# Оптимизация весов между группами
def optimize_final(weights):
    w1, w2, w3 = weights
    final_pred = w1 * xgb_pred + w2 * cat_pred + w3 * lgb_pred
    return smape(y_val, final_pred)

result = minimize(optimize_final, [1/3, 1/3, 1/3], bounds=[(0, 1), (0, 1), (0, 1)], constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
w_xgb, w_cat, w_lgb = result.x

print(f"Optimized Group Weights: XGBoost={w_xgb:.3f}, CatBoost={w_cat:.3f}, LightGBM={w_lgb:.3f}")

final_pred = w_xgb * xgb_pred + w_cat * cat_pred + w_lgb * lgb_pred
final_smape = smape(y_val, final_pred)
print(f'Final Ensemble SMAPE: {final_smape:.2f}%')


xgb_test_preds = [model.predict(test_X) for model in xgb_models]
cat_test_preds = [model.predict(test_X) for model in cat_models]
lgb_test_preds = [model.predict(test_X) for model in lgb_models]

xgb_final = np.sum(xgb_weights[:, None] * np.array(xgb_test_preds), axis=0)
cat_final = np.sum(cat_weights[:, None] * np.array(cat_test_preds), axis=0)
lgb_final = np.sum(lgb_weights[:, None] * np.array(lgb_test_preds), axis=0)

final_test_pred = w_xgb * xgb_final + w_cat * cat_final + w_lgb * lgb_final


len(test_X)


len(ids)


sub = pd.DataFrame({"id": ids, "price": final_test_pred})
sub.to_csv("submission.csv", index = False)

