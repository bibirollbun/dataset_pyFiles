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


!pip install autogluon.tabular[all] --no-cache-dir --quiet



!pip install --no-cache-dir scikit-learn==1.2.2 --force-reinstall --quiet



import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ✅ AutoGluon
from autogluon.tabular import TabularPredictor


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna

# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_orig = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")
df_orig = df_orig.rename(columns={'Personality': 'match_p'}).drop_duplicates()
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Merge External Features
train = train.merge(df_orig, how='left')
test = test.merge(df_orig, how='left')

# Encode Target
le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])
y = train["Personality_encoded"]

# Preprocessing
X = train.drop(columns=["id", "Personality", "Personality_encoded"])
X_test = test.drop(columns=["id"])
combined = pd.concat([X, X_test], axis=0)
cat_cols = combined.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)

# CV Setup
N_SPLITS = 5
SEED = 42
skf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=SEED)

# Placeholders
oof_preds = np.zeros(len(X))
meta_features = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.01, 
                                  subsample=0.8, colsample_bytree=0.8, random_state=SEED, 
                                  eval_metric='logloss', use_label_encoder=False)
    xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_val, y_val)], verbose=False)
    meta_features[val_idx, 0] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds[:, 0] += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=1000, max_depth=4, learning_rate=0.01, 
                                   subsample=0.8, colsample_bytree=0.8, random_state=SEED)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='logloss', callbacks=[lgb.early_stopping(50)])
    meta_features[val_idx, 1] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds[:, 1] += lgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

    # CatBoost
    cat_model = cb.CatBoostClassifier(n_estimators=1000, max_depth=4, learning_rate=0.01, verbose=0, random_state=SEED)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    meta_features[val_idx, 2] = cat_model.predict_proba(X_val)[:, 1]
    test_preds[:, 2] += cat_model.predict_proba(X_test)[:, 1] / N_SPLITS

# Logistic Regression Meta Model
meta_model = LogisticRegression()
meta_model.fit(meta_features, y)
oof_meta_preds = meta_model.predict_proba(meta_features)[:, 1]
test_meta_preds = meta_model.predict_proba(test_preds)[:, 1]

# Optimize Threshold
fpr, tpr, thresholds = roc_curve(y, oof_meta_preds)
opt_thresh = thresholds[np.argmax(tpr - fpr)]

# Evaluate
print(f"OOF LogLoss: {log_loss(y, oof_meta_preds):.6f}")
print(f"OOF Accuracy: {accuracy_score(y, oof_meta_preds > opt_thresh):.6f}")
print(f"Best Threshold: {opt_thresh:.4f}")

# Submission
final_preds = (test_meta_preds > opt_thresh).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
submission.head()





