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


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


train_df


test_df


submission_df


submission_df["diagnosed_diabetes"].sum()


train_df.isna().sum()


test_df.isna().sum()


train_df.info()


train_df.hist(bins=20, figsize=(20,16))
plt.show()


correlation_matrix = train_df.corr(numeric_only=True)
plt.figure(figsize=(20,16))
sns.heatmap(correlation_matrix, annot=True,  cmap='coolwarm', fmt=".3f")



TARGET = "diagnosed_diabetes"
ID_COL = "id"

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in train_df.columns if c not in cat_cols + [TARGET, ID_COL]]

FEATURES = [c for c in train_df.columns if c not in [ID_COL, TARGET]]



cat_cols


for col in cat_cols:
    plt.figure(figsize=(4,3))
    sns.countplot(x=col, data=train_df)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()


num_cols


FEATURES


from sklearn.preprocessing import OrdinalEncoder

cat_cols = train_df.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])
test_df[cat_cols]  = encoder.transform(test_df[cat_cols])


train_df


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

X = train_df[FEATURES]
y = train_df[TARGET]
X_test = test_df[FEATURES]


SEEDS = [1, 5, 37, 1234, 2025]
N_SPLITS = 5

final_test_preds = np.zeros(len(X_test))
oof_preds_all = []

for seed in SEEDS:
    print(f"\n========== SEED {seed} ==========")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}")

        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.015,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            early_stopping_rounds=300,
            random_state=seed
        )

        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[val], y.iloc[val])],
            verbose=False
        )

        oof_preds[val] = model.predict_proba(X.iloc[val])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    seed_auc = roc_auc_score(y, oof_preds)
    print(f"Seed {seed} OOF AUC: {seed_auc:.5f}")

    oof_preds_all.append(oof_preds)
    final_test_preds += test_preds / len(SEEDS)


final_oof = np.mean(oof_preds_all, axis=0)
final_auc = roc_auc_score(y, final_oof)

print("\n==============================")
print(f"FINAL ENSEMBLE OOF AUC: {final_auc:.5f}")
print("==============================")



submission_df[TARGET] = final_test_preds
submission_df.to_csv("submission.csv", index=False)

print("submission.csv saved")





