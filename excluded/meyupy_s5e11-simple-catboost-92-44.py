import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
print(df.shape)
df.head()


corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, center=0)


for col in df.select_dtypes(include="object"):
    rates = df.groupby(col)["loan_paid_back"].mean().reset_index()
    plt.figure(figsize=(8,4))
    sns.barplot(x=col, y="loan_paid_back", data=rates)


def add_features(df):
    df = df.copy()

    # count encoding
    categorical_feats = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_feats:
        df[f"CE_{col}"] = df[col].map(df[col].value_counts())

    
    important_feats = ["debt_to_income_ratio", "credit_score", "interest_rate"]
    for feat in important_feats:
        df[f"SQUARED_{feat}"] = df[feat] ** 2

    return df

# df_widen = add_features(df)


df_widen = df.copy()


corr = df_widen.corr(numeric_only=True)
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", square=True, center=0)


def downcast(df):
    df = df.copy()
    mem_before = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before: {mem_before:.2f} MB")
    for col in df.select_dtypes("number"):
        dtype = 'integer' if pd.api.types.is_integer_dtype(df[col]) else 'float'
        df[col] = pd.to_numeric(df[col], downcast=dtype)
    mem_after = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after: {mem_after:.2f} MB ({100 * (mem_before - mem_after) / mem_before:.1f}% decrease)")
    return df
    
df_widen = downcast(df_widen)


columns_to_drop = ["id"]
df_data = df_widen.drop(columns=columns_to_drop)
cat_feats = df_data.select_dtypes(include="object").columns.tolist()
df_data[cat_feats] = df_data[cat_feats].astype("category")
X, y = df_data.drop(columns=["loan_paid_back"]), df_data["loan_paid_back"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_pool = Pool(X_train, label=y_train, cat_features=cat_feats)
test_pool = Pool(X_test, label=y_test, cat_features=cat_feats)


df_ext = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_ext[cat_feats] = df_ext[cat_feats].astype("category")
X_ext = df_ext.drop(columns=columns_to_drop)


model_xgb = XGBClassifier(n_estimators=55, max_depth=7, enable_categorical=True, random_state=42)
model_xgb.fit(X_train, y_train, verbose=2)


y_pred = model_xgb.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred) * 1e4


model_cb = CatBoostClassifier(iterations=100_000, learning_rate=0.01, eval_metric="AUC", early_stopping_rounds=1000, verbose=1000, random_state=42)


model_cb.fit(train_pool, eval_set=test_pool)
model_cb.save_model("/kaggle/working/catboost_v3.cbm")


# model_cb.load_model("/kaggle/input/s5e11-catboost-92-35/catboost_v2.cbm")


y_pred = model_cb.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred) * 1e4


y_pred = model_cb.predict_proba(X_ext)[:,1]
df_ss = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
df_ss["loan_paid_back"] = y_pred
# df_ss.to_csv("/kaggle/working/v1_9243.csv", index=False)

