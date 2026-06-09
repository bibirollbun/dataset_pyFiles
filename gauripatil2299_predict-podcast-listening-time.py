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


import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings



train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')




y = train['Listening_Time_minutes']
train.drop(columns=['Listening_Time_minutes'], inplace=True)



plt.figure(figsize=(10, 5))
sns.histplot(y, bins=50, kde=True, color='dodgerblue')
plt.title("Target Distribution - Listening Time (Minutes)")
plt.xlabel("Listening Time (minutes)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



df = pd.concat([train, test], axis=0)


categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 
                    'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in categorical_cols:
    df[col] = df[col].astype(str)  # Ensure string
    df[col] = LabelEncoder().fit_transform(df[col])


print("Train Columns:\n", train.columns)
print("\nSample rows:\n", train.head())



df['guest_host_pop_ratio'] = df['Guest_Popularity_percentage'] / (df['Host_Popularity_percentage'] + 1)
df['has_guest'] = df['Guest_Popularity_percentage'].notna().astype(int)
df['episode_length_log'] = np.log1p(df['Episode_Length_minutes'])
df['total_popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage'].fillna(0)
df['ad_density'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)




plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()



X_train = df.iloc[:len(y), :].copy()
X_test = df.iloc[len(y):, :].copy()



folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X_train))
oof_cb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
preds_lgb = np.zeros(len(X_test))
preds_cb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))



lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'verbosity': -1,
    'random_state': 42
}

# Store model for SHAP
lgb_models = []

for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X_train.iloc[val_idx], y.iloc[val_idx]
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
    model_lgb = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_valid],
    num_boost_round=10000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)  # Optional: adjust or remove
    ]
)

    oof_lgb[val_idx] = model_lgb.predict(X_val)
    preds_lgb += model_lgb.predict(X_test) / folds.n_splits
    lgb_models.append(model_lgb)



lgb.plot_importance(lgb_models[0], max_num_features=20, importance_type='gain', figsize=(10, 6))
plt.title("LightGBM Feature Importance (Fold 0)")
plt.show()



for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X_train.iloc[val_idx], y.iloc[val_idx]
    model_cb = cb.CatBoostRegressor(verbose=0, iterations=5000, learning_rate=0.01, depth=8, loss_function='RMSE', random_seed=42)
    model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    oof_cb[val_idx] = model_cb.predict(X_val)
    preds_cb += model_cb.predict(X_test) / folds.n_splits



xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 6,
    'n_estimators': 5000,
    'seed': 42
}

for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X_train.iloc[val_idx], y.iloc[val_idx]
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    preds_xgb += model_xgb.predict(X_test) / folds.n_splits



stacked = pd.DataFrame({
    'lgb': oof_lgb,
    'cb': oof_cb,
    'xgb': oof_xgb
})
stacked_test = pd.DataFrame({
    'lgb': preds_lgb,
    'cb': preds_cb,
    'xgb': preds_xgb
})

meta_model = Ridge(alpha=0.5)
meta_model.fit(stacked, y)
final_preds = meta_model.predict(stacked_test)



plt.figure(figsize=(8, 6))
plt.scatter(meta_model.predict(stacked), y, alpha=0.3, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("OOF Predictions")
plt.ylabel("Actual Listening Time")
plt.title("OOF Predictions vs Actual")
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 5))
sns.histplot(final_preds, bins=50, kde=True, color='purple')
plt.title("Final Prediction Distribution")
plt.xlabel("Predicted Listening Time")
plt.grid(True)
plt.show()


# ✅ SHAP Workaround (No PyTorch)
try:
    import shap

    explainer = shap.Explainer(lgb_models[0])
    shap_values = explainer(X_train.iloc[:200])
    shap.plots.beeswarm(shap_values, max_display=15)
except Exception as e:
    print("SHAP error:", e)




print("Ensemble RMSE:", mean_squared_error(y, meta_model.predict(stacked), squared=False))
sample_submission['Listening_Time_minutes'] = final_preds
sample_submission.to_csv('submission.csv', index=False)

