import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score , StratifiedKFold
from sklearn.preprocessing import LabelEncoder , StandardScaler,OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



train_bank = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_bank = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train_bank.head()


train_bank.columns


test_bank.dtypes


test_bank.head()


plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=train_bank)
plt.title('Distribution of Bank Term Deposit Subscriptions')
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel('Subscribed to Term Deposit?')
plt.ylabel('Count')
plt.show()


categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    sns.countplot(x=col, hue='y', data=train_bank, ax=axes[i], order=train_bank[col].value_counts().index)
    axes[i].set_title(f'Subscription by {col}')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()


numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    sns.histplot(data=train_bank, x=col, hue='y', kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col} by Subscription')

plt.tight_layout()
plt.show()

for col in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='y', y=col, data=train_bank)
    plt.title(f'{col} vs. Subscription')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()


for col in numerical_features:
    plt.figure(figsize=(6,4))
    sns.histplot(train_bank[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()


for col in numerical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="y", y=col, data=train_bank)
    plt.title(f"{col} vs Target (y)")
    plt.show()

for col in categorical_features:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, hue="y", data=train_bank)
    plt.title(f"{col} vs Target (y)")
    plt.xticks(rotation=45)
    plt.show()


for col in ["default", "housing", "loan"]:
    if train_bank[col].dtype == "object":
        train_bank[col] = train_bank[col].map({"yes": 1, "no": 0})

df_corr = train_bank.copy()
df_corr = pd.get_dummies(df_corr, drop_first=True)

plt.figure(figsize=(12,8))
sns.heatmap(df_corr.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()


train_ids = train_bank["id"]
test_ids = test_bank["id"]


y = train_bank["y"]
X = train_bank.drop(["id", "y"], axis=1)
X_test = test_bank.drop(["id"], axis=1)


for col in ["default", "housing", "loan"]:
    for train_bank in [X, X_test]:
        train_bank[col] = train_bank[col].map({"yes": 1, "no": 0})



cat_cols = ["job", "marital", "education", "contact", "month", "poutcome"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)


X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)



scaler = StandardScaler()
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


scale_pos_weight = len(y[y==0]) / len(y[y==1])

lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    objective="binary"
)


lgbm.fit(X, y)

y_pred = lgbm.predict(X_test)

submission = pd.DataFrame({
    "id": test_ids,
    "y": y_pred
})
submission.to_csv("submission_lgbm.csv", index=False)




