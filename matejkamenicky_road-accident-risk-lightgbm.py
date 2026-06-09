import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


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
sns.histplot(data['accident_risk'], kde=True, bins=50)
plt.title(f"Distribution of target")
plt.show()


numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('id')

data[numeric_cols].hist(bins=30, figsize=(18, 15), edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=18)
plt.show()


cat_cols = data.select_dtypes(include=['bool', 'object']).columns.tolist()

fig, axes = plt.subplots(4, 2, figsize=(12, 15))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    sns.countplot(data=data, x=col, ax=ax)
    ax.set_title(col, fontsize = 16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize = 14)


plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 6))

data_corr = data.corr(numeric_only=True)

heatmap = sns.heatmap(data_corr.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12})

plt.show()


correlations = data[numeric_cols].corr()['accident_risk']
correlations = correlations.drop('accident_risk')

top_features = correlations.abs().sort_values(ascending=False).head(5).index.tolist()

print("Top 5 features correlated with target:")
print(correlations[top_features])


for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot for {col}")
    plt.show()


from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def add_engineered_features(df):
    data = df.copy()
    
    # 1. Polynomial features
    data['speed_squared'] = data['speed_limit'] ** 2
    df['curvature_squared'] = df['curvature'] ** 2
    
    # 2. Binned features
    data['curvature_bin'] = pd.cut(data['curvature'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])
    data['speed_category'] = pd.cut(data['speed_limit'], bins=[0, 30, 50, 100], labels=[0, 1, 2])
    
    # 3. Combined categorical features
    data["road_weather"] = data["road_type"].astype(str) + "_" + data["weather"].astype(str)
    data["road_lighting"] = data["road_type"].astype(str) + "_" + data["lighting"].astype(str)
    
    # 3. Numerical interaction features
    data['speed_curvature'] = data['speed_limit'] * data['curvature']
    data['lanes_curvature'] = data['num_lanes'] * data['curvature']
    data['speed_lanes'] = data['speed_limit'] * data['num_lanes']

    # 4. Environmental risk conditions
    data["is_high_risk_condition"] = (
        ((data["weather"].isin(["Rain", "Snow", "Fog"])) &
         (data["lighting"].isin(["Low", "Dark"]))) |
        ((data["time_of_day"].isin(["Night", "Late Evening"])) &
         (data["weather"].isin(["Rain", "Fog"])))
    ).astype(int)

    return data


X = data.drop(['accident_risk'], axis=1)
y = data['accident_risk']


categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("scaler", StandardScaler())
        ]), numerical_cols),

        ("cat", Pipeline([
            ("encoder", OrdinalEncoder())
        ]), categorical_cols),
    ]
)


X_processed = add_engineered_features(X)
X_test_processed = add_engineered_features(test_data)


# X_processed = preprocessor.fit_transform(X)
# X_test_processed = preprocessor.transform(test_data)


import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "gpu" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 20,
    "verbose": -1,
    "random_state": 42
}


cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 5000,
    "learning_rate": 0.01,
    "depth": 8,
    "l2_leaf_reg": 3,
    "bagging_temperature": 0.2,
    "random_strength": 1,
    "border_count": 128,
    "boosting_type": "Plain",   
    "task_type": "GPU",         
    "od_type": "Iter",
    "od_wait": 300,            
    "random_seed": 42,
    "verbose": 100
}



def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


categorical_cols = X_processed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
for col in categorical_cols:
    X_processed[col] = X_processed[col].astype('category')
    X_test_processed[col] = X_test_processed[col].astype('category')


kf = KFold(n_splits=10, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(X_processed))
test_preds_lgb = np.zeros(len(X_test_processed))

oof_preds_cat = np.zeros(len(X_processed))
test_preds_cat = np.zeros(len(X_test_processed))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_processed, y)):
    print(f"Training fold {fold+1}...")
    X_train, X_valid = X_processed.iloc[train_idx], X_processed.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # ---------------- LightGBM ----------------
    lgb_model  = lgb.LGBMRegressor(categorical_feature=categorical_cols, **lgb_params)
    lgb_model .fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[
        lgb.early_stopping(stopping_rounds=300)]
        )
    
    oof_preds_lgb[valid_idx] = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration_)
    test_preds_lgb += lgb_model.predict(X_test_processed, num_iteration=lgb_model.best_iteration_) / kf.n_splits

     # ---------------- CatBoost ----------------
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        cat_features=categorical_cols ,
        verbose=False
        )
    oof_preds_cat[valid_idx] = cat_model.predict(X_valid)
    test_preds_cat += cat_model.predict(X_test_processed) / kf.n_splits

rmse_lgb = rmse(y, oof_preds_lgb)
rmse_cat = rmse(y, oof_preds_cat)

print(f"LightGBM OOF RMSE: {rmse_lgb:.5f}")
print(f"CatBoost OOF RMSE: {rmse_cat:.5f}")


oof_ensemble = 0.5 * oof_preds_lgb + 0.5 * oof_preds_cat
test_ensemble = 0.5 * test_preds_lgb + 0.5 * test_preds_cat

rmse_ensemble = rmse(y, oof_ensemble)
print(f"Ensemble OOF RMSE: {rmse_ensemble:.5f}")


submission = pd.DataFrame({
    'id': test_data["id"],
    'y': test_preds_lgb
})


submission


submission.to_csv('submission.csv', index=False)




