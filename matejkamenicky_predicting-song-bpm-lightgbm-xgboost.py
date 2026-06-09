import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


data.shape


data.head()


data.tail()


data.describe()


data.info()


missing_table = pd.DataFrame({
    'Missing Values': data.isna().sum(),
    'Percentage (%)': (data.isnull().mean() * 100).round(2)
})

print(missing_table.sort_values(by='Missing Values', ascending=False))


data.nunique()


plt.figure(figsize=(8,5))
sns.histplot(data['BeatsPerMinute'], kde=True, bins=30)
plt.title(f"Distribution of target")
plt.show()


numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('id')

data[numeric_cols].hist(bins=30, figsize=(20, 15), edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=18)
plt.show()


plt.figure(figsize=(16, 6))

data_corr = data.corr(numeric_only=True)

heatmap = sns.heatmap(data_corr.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12})

plt.show()


correlations = data[numeric_cols].corr()['BeatsPerMinute']
correlations = correlations.drop('BeatsPerMinute')

top_features = correlations.abs().sort_values(ascending=False).head(5).index.tolist()

print("Top 5 features correlated with target:")
print(correlations[top_features])


for feature in top_features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=data[feature], y=data['BeatsPerMinute'])
    plt.title(f"{feature} vs target (corr={correlations[feature]:.4f})")
    plt.show()


numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('id')

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot for {col}")
    plt.show()


from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline


X = data.drop(['BeatsPerMinute', 'id'], axis=1)
y = data['BeatsPerMinute']


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


test_id = test_data['id']
test_data = test_data.drop('id', axis=1)


X_clipped = X.copy()

for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X_clipped[col] = X[col].clip(lower, upper)


scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_clipped), columns=X_clipped.columns, index=X_clipped.index)
test_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns, index=test_data.index)


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 20,
    "verbose": -1,
    "random_state": 42
}


xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "gamma": 0,
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 300
}



kf = KFold(n_splits=10, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(X))
test_preds_lgb = np.zeros(len(test_scaled))

oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(test_scaled))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"Training fold {fold+1}...")
    X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # ---------------- LightGBM ----------------
    lgb_model  = lgb.LGBMRegressor(**lgb_params)
    lgb_model .fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[
        lgb.early_stopping(stopping_rounds=300)]
        )

    oof_preds_lgb[valid_idx] = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration_)
    test_preds_lgb += lgb_model.predict(test_scaled, num_iteration=lgb_model.best_iteration_) / kf.n_splits


    # ---------------- XGBoost ----------------
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
        )
    oof_preds_xgb[valid_idx] = xgb_model.predict(X_valid)
    test_preds_xgb += xgb_model.predict(test_scaled) / kf.n_splits



rmse_lgb = root_mean_squared_error(y, oof_preds_lgb)
rmse_xgb = root_mean_squared_error(y, oof_preds_xgb)
print(f"LightGBM OOF RMSE: {rmse_lgb:.5f}")
print(f"XGBoost  OOF RMSE: {rmse_xgb:.5f}")


oof_ensemble = 0.7 * oof_preds_lgb + 0.3 * oof_preds_xgb
test_ensemble = 0.7 * test_preds_lgb + 0.3 * test_preds_xgb

rmse_ensemble = root_mean_squared_error(y, oof_ensemble)
print(f"Ensemble OOF RMSE: {rmse_ensemble:.5f}")


submission = pd.DataFrame({
    'id': test_id,
    'y': test_ensemble
})


submission


submission.to_csv('submission.csv', index=False)




