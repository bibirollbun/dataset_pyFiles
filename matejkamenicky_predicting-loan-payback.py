import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


data.shape


data.head()


data.tail()


data.info()


data.describe()


missing_table = pd.DataFrame({
    'Missing Values': data.isna().sum(),
    'Percentage (%)': (data.isnull().mean() * 100).round(2)
})

print(missing_table.sort_values(by='Missing Values', ascending=False))


data.nunique()


data['loan_paid_back'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, explode=(0,0.1), colors=['#4c72b0', '#aec7e8'])
plt.title('Target distribution - loan payback')
plt.ylabel('')
plt.show()


numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('id')
numeric_cols.remove('loan_paid_back')

data[numeric_cols].hist(bins=30, figsize=(18, 15), edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=18)
plt.show()


cat_cols = data.select_dtypes(include=['bool', 'object']).columns.tolist()

fig, axes = plt.subplots(3, 2, figsize=(12, 15))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    sns.countplot(data=data, x=col, ax=ax)
    ax.set_title(col, fontsize = 16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center', fontsize = 10)


plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 6))

data_corr = data.corr(numeric_only=True)

heatmap = sns.heatmap(data_corr.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12})

plt.show()


numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('id')

correlations = data[numeric_cols].corr()['loan_paid_back']
correlations = correlations.drop('loan_paid_back')

top_features = correlations.abs().sort_values(ascending=False).head(5).index.tolist()

print("Top 5 features correlated with target:")
print(correlations[top_features])


def new_features(df):
    df = df.copy()

    df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["loan_burden"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["income_per_credit"] = df["annual_income"] / (df["credit_score"] + 1)

    df["loan_amount_log"] = np.log1p(df["loan_amount"])
    df["annual_income_log"] = np.log1p(df["annual_income"])
    df["log_credit"] = np.log1p(df["credit_score"])

    df["annual_debt"] = df["annual_income"] * df["debt_to_income_ratio"]
    df["financial_health"] = (df["credit_score"] / 850) * (1 - df["debt_to_income_ratio"])
    df["income_efficiency"] = df["annual_income"] * (1 - df["debt_to_income_ratio"])
    df["affordability_score"] = (df["annual_income"] / 12) / (
        df["loan_amount"] * df["interest_rate"] / 1200 + 1
    )

    df["grade_letter"] = df["grade_subgrade"].str[0]
    df["grade_number"] = df["grade_subgrade"].str[1].astype(int)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df["grade_rank"] = df["grade_letter"].map(grade_map)
    df["grade_score"] = df["grade_rank"] * 10 + df["grade_number"]

    df["interest_rate_squared"] = df["interest_rate"] ** 2

    df["is_self_employed"] = (df["employment_status"] == "Self-employed").astype(int)

    return df



data = new_features(data)
test_data = new_features(test_data)


import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


X = data.drop(['id', 'loan_paid_back'], axis=1)
y = data['loan_paid_back']
test_id = test_data['id']
test_data = test_data.drop(['id'], axis=1)


cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]


lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 45,
    "max_depth": 10,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    "min_data_in_leaf": 70,
    'min_child_samples': 20,
    'reg_alpha': 0.05,
    'reg_lambda': 0.1,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
    "device": "gpu"
}


xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    "tree_method": "gpu_hist",
    'max_depth': 8,
    'colsample_bytree': 0.5,
    'subsample': 0.55,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'min_child_weight': 20,
    'gamma': 0.7,
    'reg_alpha': 0.2,
    'reg_lambda': 0.3,
    'random_state': 42,
    'n_jobs': -1,
    'enable_categorical': True,
    'device': 'cuda',
}


oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_lgb = np.zeros(len(test_data))
test_xgb = np.zeros(len(test_data))

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for c in cat_cols:
    X[c] = X[c].astype("category")
    test_data[c] = test_data[c].astype("category")


for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"=== FOLD {fold} ===")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # --------------------- LightGBM -------------------------
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
    lgb_valid = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_cols)
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=200)]
    )

    oof_lgb[valid_idx] = lgb_model.predict(X_valid)
    test_lgb += lgb_model.predict(test_data) / kf.n_splits

    # --------------------- XGBoost -------------------------    
    xgb_model = xgb.XGBClassifier(**xgb_params)

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=False
    )

    oof_xgb[valid_idx] = xgb_model.predict_proba(X_valid)[:, 1]
    test_xgb += xgb_model.predict_proba(test_data)[:, 1] / kf.n_splits


oof_blend= (oof_lgb * 0.8 + oof_xgb * 0.2)
oof_auc = roc_auc_score(y, oof_blend)
print("OOF AUC:", oof_auc)


print("LGB OOF:", roc_auc_score(y, oof_lgb))
print("XGB OOF:", roc_auc_score(y, oof_xgb))


test_final = test_lgb * 0.75 + test_xgb * 0.25


submission = pd.DataFrame({
    'id': test_id,
    'y': test_final
})


submission


submission.to_csv('submission.csv', index=False)




