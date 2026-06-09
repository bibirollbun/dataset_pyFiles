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


# ğŸ“¦ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ğŸ“¥ Load Data
df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
dt = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
org1 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv")
org2 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")
org3 = pd.read_csv("/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv")

# ğŸ§¹ Preprocess org3 and Merge
org3[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]] = \
    org3[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]].astype(float)
org_full = pd.concat([org1, org2, org3], ignore_index=True)
org_full = org_full.rename(columns={"Personality": "P2"})
org_full = org_full.drop_duplicates(subset=[
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
    'Post_frequency'
])

df = df.drop(columns=['id'])
dt = dt.drop(columns=['id'])
df = df.merge(org_full, how='left')
dt = dt.merge(org_full, how='left')

# ğŸ”¢ Encode target
le = LabelEncoder()
df["Personality"] = le.fit_transform(df["Personality"])

# ğŸ�¯ Prepare features and target
X = df.drop(columns=["Personality"])
y = df["Personality"]
X_test = dt.copy()

# ğŸ”¤ Ordinal encode categoricals
cat_cols = ['Stage_fear', 'Drained_after_socializing', 'P2']
combined = pd.concat([X, X_test], axis=0)
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])
X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)

# â�• Add feature
X['Social_ratio'] = X['Social_event_attendance'] / (X['Friends_circle_size'] + 1)
X_test['Social_ratio'] = X_test['Social_event_attendance'] / (X_test['Friends_circle_size'] + 1)

# ğŸ”� Cross-validation
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
oof_xgb = np.zeros(len(X)); test_xgb = np.zeros(len(X_test))
oof_cat = np.zeros(len(X)); test_cat = np.zeros(len(X_test))

# ğŸš€ XGBoost Training
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    model = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                      evals=[(dval, "valid")], early_stopping_rounds=50, verbose_eval=False)
    oof_xgb[val_idx] += model.predict(dval)
    test_xgb += model.predict(dtest) / 5

# ğŸ�± CatBoost Training
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        eval_metric='Logloss', random_seed=42, verbose=0,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    oof_cat[val_idx] += model.predict_proba(X_val)[:, 1]
    test_cat += model.predict_proba(X_test)[:, 1] / 5

# ğŸ”� Stacking
stacked_train = np.vstack([oof_xgb, oof_cat]).T
stacked_test = np.vstack([test_xgb, test_cat]).T
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y)
final_test_preds = meta_model.predict_proba(stacked_test)[:, 1]

# âœ… Evaluate
oof_meta = meta_model.predict_proba(stacked_train)[:, 1]
oof_binary = (oof_meta >= 0.5).astype(int)
print("\nğŸ“Š Evaluation Metrics:")
print("Log Loss:", log_loss(y, oof_meta))
print("ROC AUC:", roc_auc_score(y, oof_meta))
print("Accuracy:", accuracy_score(y, oof_binary))
print("F1 Score:", f1_score(y, oof_binary))
print("Precision:", precision_score(y, oof_binary))
print("Recall:", recall_score(y, oof_binary))

# ğŸ“¤ Submission
submission["Personality"] = le.inverse_transform((final_test_preds >= 0.5).astype(int))
submission.to_csv("submission.csv", index=False)
print("\nğŸ“„ Submission Head:")
print(submission.head())

# ğŸ“ˆ Plot Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=submission["Personality"])
plt.title("Predicted Personality Distribution")
plt.xlabel("Personality")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ğŸ§® Value Counts
print("\nğŸ”¢ Personality Prediction Counts:")
print(submission["Personality"].value_counts())


