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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance


import warnings


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


def highlight_rows(row):
    return ['background-color: lightpink; color: white' if i % 2 == 0 else 'background-color: lightgrey; color: black' for i in range(len(row))]

train.head().style.apply(lambda x: highlight_rows(x), axis=1)


def highlight_rows(row):
    return ['background-color: lightpink; color: white' if i % 2 == 0 else 'background-color: lightgrey; color: black' for i in range(len(row))]

test.head().style.apply(lambda x: highlight_rows(x), axis=1)


train.shape



test.shape


print("\nMissing values in train:\n", train.isnull().sum())
print("\nMissing values in test:\n", test.isnull().sum())


train.info()


test.info()


cols = train.columns
num_cols_train = [x for x in cols if train[x].dtype != 'O' and x not in ['id', 'Personality']]
cat_cols_train = [y for y in cols if y not in num_cols_train and y not in ['id', 'Personality']]


for col in num_cols_train:
    ext_mean = train[train.Personality == 'Extrovert'][col].mean()
    intro_mean = train[train.Personality == 'Introvert'][col].mean()
    train.loc[train.Personality == 'Extrovert', col] = train[train.Personality == 'Extrovert'][col].fillna(ext_mean)
    train.loc[train.Personality == 'Introvert', col] = train[train.Personality == 'Introvert'][col].fillna(intro_mean)


for col in cat_cols_train:
    ext_mode = train[train.Personality == 'Extrovert'][col].mode()[0]
    intro_mode = train[train.Personality == 'Introvert'][col].mode()[0]
    train.loc[train.Personality == 'Extrovert', col] = train[train.Personality == 'Extrovert'][col].fillna(ext_mode)
    train.loc[train.Personality == 'Introvert', col] = train[train.Personality == 'Introvert'][col].fillna(intro_mode)


train.isnull().sum()


plt.figure(figsize=(8, 5))
sns.countplot(x="Personality", data=train)
plt.title("Class distribution: Introvert vs Extrovert")
plt.show()


cat_cols = train.select_dtypes(include="object").drop(columns=["Personality"], errors="ignore").columns.tolist()
for col in cat_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=train, x=col, hue="Personality")
    plt.title(f'Count of {col} by Personality')
    plt.xticks(rotation=45)
    plt.legend(title='Personality')
    plt.tight_layout()
    plt.show()


for col in cat_cols:
    grouped = train.groupby([col, 'Personality']).size().unstack(fill_value=0)
    grouped = grouped.div(grouped.sum(axis=1), axis=0)  
    
    grouped.plot(kind='bar', stacked=True, figsize=(10, 4))
    plt.title(f'{col} vs Personality Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Proportion')
    plt.legend(title='Personality')
    plt.tight_layout()
    plt.show()


warnings.simplefilter(action='ignore', category=FutureWarning)

num_cols = train.select_dtypes(include='number').drop(columns=['id'], errors='ignore').columns.tolist()

for col in num_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(data=train, x=col, hue='Personality', bins=20, kde=True)
    plt.title(f'Distribution of {col} by Personality')
    plt.show()


for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Personality', y=col, data=train)
    plt.title(f'{col} vs Personality')
    plt.show()


le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])

# Creating features
X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])

# Combining for processing categorical features
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)

# Encode categorical features
cat_cols_combined = combined.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols_combined] = encoder.fit_transform(combined[cat_cols_combined])

X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=10,
        learning_rate=0.015,
        n_estimators=3000,
        subsample=0.79,
        colsample_bytree=0.95,
        min_child_weight=5,
        gamma=1.1,
        reg_alpha=2.0,
        reg_lambda=1.0,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42,
        use_label_encoder=False,
        enable_categorical=True
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

cv_acc = accuracy_score(y, oof_preds)
print(f"CV Accuracy: {cv_acc:.4f}")


print(classification_report(y, oof_preds, target_names=le.classes_))


plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight', max_num_features=10, height=0.5)
plt.title("Feature Importance")
plt.show()


submission["Personality"] = le.inverse_transform((test_preds > 0.5).astype(int))
submission.to_csv("submission.csv", index=False)
print(submission.head())

