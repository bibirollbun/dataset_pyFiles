# Base
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import lightgbm as lgb
import ydf
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import optuna
import scipy



# Plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode(connected = True)
import calendar

# Options
sns.set(style='whitegrid')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*use_inf_as_na option is deprecated.*')


sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')
target = 'accident_risk'

print(f'Train shape {train.shape}, Test shape {test.shape}')


train[:5]


train.describe()


train.describe(include=['object'])


plt.figure(figsize=(16,8))
sns.histplot(train['accident_risk'], kde=True, bins= 50)
plt.title('Distribution accident_risk')
plt.show()


num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
train[num_cols].hist(bins=35, figsize=(16,8))
plt.show()


def add_features(df):
    df = df.copy()
    
    # Определим числовые и категориальные признаки внутри функции
    num_features = df.select_dtypes(exclude=['object', 'bool', 'string', 'category']).columns.tolist()
    cat_features = df.select_dtypes(include=['object', 'bool', 'string', 'category']).columns.tolist()

    # 1. Квантилизация числовых признаков
    for col in num_features:
        df[f"{col}_quartile"] = pd.cut(df[col], bins=4, labels=False, include_lowest=True).astype('category')

    # 2. Полиномиальные признаки (если есть)
    for col in ['curvature', 'speed_limit']:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2

    # 3. Логическое правило
    if 'speed_limit' in df.columns and 'lighting' in df.columns:
        df["is_high_speed_night"] = ((df["speed_limit"] > 60) & (df["lighting"] == "night")).astype(int)

    # 4. Мета-признак (без внешних данных!)
    def f(X):
        return (
            0.3 * X["curvature"] +
            0.2 * (X["lighting"] == "night").astype(int) +
            0.1 * (X["weather"] != "clear").astype(int) +
            0.2 * (X["speed_limit"] >= 60).astype(int) +
            0.1 * (X["num_reported_accidents"] > 2).astype(int)
        )
    
    def clipped_meta(X):
        mu = f(X)
        sigma = 0.05
        a, b = -mu / sigma, (1 - mu) / sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b

    df["meta"] = clipped_meta(df)

    # 5. Приведение категорий
    cat_cols = df.select_dtypes(include=['object', 'bool', 'string', 'category']).columns
    df[cat_cols] = df[cat_cols].astype('category')
    
    return df




y = train[target]
X = train.drop(columns=[target])

# Feature engineering — без orig!
X_fe = add_features(X)
test_fe = add_features(test)

# Выравнивание колонок (на случай, если в test нет каких-то значений)
common_cols = X_fe.columns.intersection(test_fe.columns)
X_fe = X_fe[common_cols]
test_fe = test_fe[common_cols]

# Обработка пропусков в категориях
cat_cols = X_fe.select_dtypes(include='category').columns
# X_fe[cat_cols] = X_fe[cat_cols].fillna('NaN')
# test_fe[cat_cols] = test_fe[cat_cols].fillna('NaN')

# Лог-трансформация y
y_log = np.log1p(y)

# Удаление выбросов (опционально — можешь закомментировать)
Q1 = y_log.quantile(0.25)
Q3 = y_log.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
mask = (y_log >= lower) & (y_log <= upper)
X_clean = X_fe[mask].reset_index(drop=True)
y_clean = y_log[mask].reset_index(drop=True)


# ----------------------------
# 6. Кросс-валидация и обучение
# ----------------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

models = {
    "LGBM": lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='regression',
        metric='rmse'
    ),
    "YDF": ydf.GradientBoostedTreesLearner(
        label="target",
        task=ydf.Task.REGRESSION,
        num_trees=1000,
        growing_strategy="BEST_FIRST_GLOBAL",
        max_depth=9,
        subsample=0.8,
        random_seed=42
    )
}

oof_preds = {}
test_preds = {}

for name, model in models.items():
    print(f"\n{'='*20}\nTraining {name}\n{'='*20}")
    
    oof = np.zeros(len(X_clean))
    test_pred = np.zeros(len(test_fe))
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_clean, y_clean)):
        X_tr, X_val = X_clean.iloc[trn_idx], X_clean.iloc[val_idx]
        y_tr, y_val = y_clean.iloc[trn_idx], y_clean.iloc[val_idx]
        
        if name == "YDF":
            # YDF требует pandas DataFrame с целевой колонкой
            train_df = X_tr.copy()
            train_df["target"] = y_tr
            val_df = X_val.copy()
            val_df["target"] = y_val
            
            ydf_model = model.train(train_df)
            val_pred = ydf_model.predict(val_df)
            test_pred_fold = ydf_model.predict(test_fe)
        else:
            # LGBM / XGB работают напрямую
            if name == "LGBM":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
                )

            val_pred = model.predict(X_val)
            test_pred_fold = model.predict(test_fe)
        
        oof[val_idx] = val_pred
        test_pred += test_pred_fold / n_splits
        
        score = rmse(y_val, val_pred)
        print(f"Fold {fold+1}: RMSE = {score:.6f}")
    
    oof_score = rmse(y_clean, oof)
    print(f"OOF RMSE: {oof_score:.6f}")
    
    oof_preds[name] = oof
    test_preds[name] = test_pred

# ----------------------------
# 7. Простое усреднение (без обучения мета-модели)
# ----------------------------
final_test_pred = np.mean(list(test_preds.values()), axis=0)

# Обратное преобразование (если использовали log1p)
final_test_pred = np.expm1(final_test_pred)


pd.DataFrame({"id": test.index, target: final_test_pred}).to_csv("submission.csv", index=False)

