# %%capture

!pip install --upgrade scikit-learn imbalanced-learn

# After this, restart (not reset) the notebook.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# from pprint import pprint as print

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score as ROC, 
    roc_curve as ROCCurve,
    accuracy_score as ACC, 
    classification_report as REPORT,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

random_state = 42

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df.head()


# df.isna().sum().sum() # clean data (w/o any null values)

num_df = df.select_dtypes(include="number")
num_df.head()


# Count plot to deduce negative to positive ratio (FYI ~1:4)

ax = sns.countplot(data=df, x="loan_paid_back")

# Add count labels
for container in ax.containers:
    ax.bar_label(container)


# cols = [
#     "annual_income",
#     "debt_to_income_ratio",
#     "credit_score",
#     "loan_amount",
#     "interest_rate"
# ]

# # Create a 2x3 grid of subplots
# fig, axes = plt.subplots(2, 3, figsize=(18, 8))
# axes = axes.flatten()  # flatten 2D array of axes → 1D list for easy looping

# for i, col in enumerate(cols):
#     # Group data by column and loan_paid_back
#     s = (
#         df.groupby([col, 'loan_paid_back'])['id']
#           .count()
#           .unstack()
#           .fillna(0)
#           .reset_index()
#     )

#     # Optional: smooth with rolling mean (uncomment if needed)
#     # s[1.0] = s[1.0].rolling(window=5, min_periods=1).mean()
#     # s[0.0] = s[0.0].rolling(window=5, min_periods=1).mean()

#     # Plot both lines
#     sns.lineplot(data=s, x=col, y=1.0, ax=axes[i], label="Paid Back")
#     sns.lineplot(data=s, x=col, y=0.0, ax=axes[i], label="Not Paid", color='orange')

#     # Formatting
#     axes[i].set_title(col.replace('_', ' ').title())
#     axes[i].tick_params(axis='x', rotation=30)
#     axes[i].legend()

# # Hide the extra (6th) subplot if there are fewer than 6 columns
# for j in range(len(cols), len(axes)):
#     fig.delaxes(axes[j])

# plt.tight_layout()
# plt.show()


cat_df = df.select_dtypes(exclude="number")
cat_df.head()


for col in cat_df.columns:
    s = df.groupby([col, 'loan_paid_back'])['id'].count().unstack()
    s["times"] = s[1.0] / s[0.0]
    s = s.sort_values("times", ascending=False)
    print(s, end="\n\n\n")


num_df["better_grade"] = (cat_df.grade_subgrade.str[0] == "A") | (cat_df.grade_subgrade.str[0] == "B")
num_df["better_grade"] = num_df["better_grade"].astype(int)

num_df["is_retired"] = cat_df["employment_status"] == "Retired"
num_df["is_retired"] = num_df["is_retired"] + 0

num_df.head()


# cols = [
#     "gender",
#     "marital_status",
#     "education_level",
#     "employment_status",
#     "loan_purpose",
#     "grade_subgrade"
# ]

# # Create subplots (2 rows × 3 columns)
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# # Flatten axes array for easy looping
# axes = axes.flatten()

# for i, col in enumerate(cols):
#     sns.countplot(data=df, x=col, hue="loan_paid_back", ax=axes[i])
#     axes[i].set_title(f"{col.replace('_', ' ').title()}")
#     axes[i].tick_params(axis='x', rotation=45)  # rotate x labels for readability

# plt.tight_layout()
# plt.show()


# Legend for mapping categorical values to numbers

legend = {}

for col in cat_df.columns:
    legend[col] = {}
    values = cat_df[col].unique()
    total = len(values)
    for index, value in enumerate(values):
        swap = index + 1
        legend[col][value] = swap
        legend[col][swap] = value


# Encode the cat variables

cat_df_encoded = pd.DataFrame()

for col in cat_df.columns:
    cat_df_encoded[col] = cat_df[col].apply(lambda x: legend[col][x])

cat_df_encoded.head()


df_final = pd.concat([num_df, cat_df_encoded], axis=1)

X = df_final.drop(['id', 'loan_paid_back'], axis=1)
y = df_final.loan_paid_back

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.2, random_state=random_state,
    stratify=y,
)

smote = SMOTE(random_state=random_state)

X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)


params = {
    "max_depth": 10,
    "colsample_bytree": 0.7,
    "subsample": 0.9,
    "n_estimators": 800,
    "learning_rate": 0.08,
    "gamma": 0.01, 
    "max_delta_step": 2,
    "eval_metric": "rmsle",
    "enable_categorical": True,
    "random_state": random_state,
}

xt = XGBClassifier(**params)

xt.fit(X_train_bal, y_train_bal)

y_preds = xt.predict(X_test)
y_preds_proba = xt.predict_proba(X_test)[:, 1]

# ROC AUC score and generic report

print(
    f'\nROC AUC Score: {ROC(y_test, y_preds)}\n',
)
print(REPORT(y_test.round(), y_preds.round()))

# Display importance of each feature

features = zip(X.columns, xt.feature_importances_)
features = sorted(dict(features).items(), key=lambda x: abs(x[1]), reverse=True)

print(f"{'Feature':<30} {'Importance':>12} {'Is important?':>16}", end="\n\n")
for col_name, importance in features:
    print(f"{col_name:<30} {importance*100:>10.4f} {'N' if not importance else 'Y':>12}")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

num_df = test_df.select_dtypes(include="number")
cat_df = test_df.select_dtypes(exclude="number")

num_df["better_grade"] = (cat_df.grade_subgrade.str[0] == "A") | (cat_df.grade_subgrade.str[0] == "B")
num_df["better_grade"] = num_df["better_grade"].astype(int)

num_df["is_retired"] = cat_df["employment_status"] == "Retired"
num_df["is_retired"] = num_df["is_retired"] + 0

cat_df_encoded = pd.DataFrame()
for col in cat_df.columns:
    cat_df_encoded[col] = cat_df[col].apply(lambda x: legend[col][x])

df_final = pd.concat([num_df, cat_df_encoded], axis=1).drop(['id'], axis=1)

y_preds = xt.predict(df_final)

submission = pd.DataFrame({
    "id": test_df.id,
    "loan_paid_back": y_preds,
})
submission.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()

