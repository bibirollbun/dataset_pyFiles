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


# Set display options to show full content of columns
pd.set_option('display.max_colwidth', None) 


df_train=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_test=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
sample_submission=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")


df_train.head(1)


df_test.head(1)


sample_submission.head(5)


print("Target distribution:\n", df_train["rule_violation"].value_counts())


# Target Distribution Check
print("\n--- Distribution of Target Variable for Class Balance Check ---\n")
df_train["rule_violation"].value_counts(normalize=True).plot(kind='barh')


print("\n--- Missing Values in Train Data ---")
print(df_train.isnull().sum())


print("\n--- Missing Values in Test Data ---")
print(df_train.isnull().sum())


print("\n--- Duplicate Rows in Train Data ---")
print(df_train.duplicated().sum())

print("\n--- Duplicate Rows in Test Data ---")
print(df_test.duplicated().sum())


def build_text(row):
    return f"{row['rule']} [SEP] {row['body']} [SEP] " \
           f"POS1: {row['positive_example_1']} [SEP] POS2: {row['positive_example_2']} [SEP] " \
           f"NEG1: {row['negative_example_1']} [SEP] NEG2: {row['negative_example_2']}"

# Apply to train and test
df_train['text'] = df_train.apply(build_text, axis=1)
df_test['text'] = df_test.apply(build_text, axis=1)



df_train.head(1)


df_test.head(1)


X=df_train["text"]
y=df_train["rule_violation"]


from sklearn.model_selection import train_test_split
# Train/val split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline


# Vectorizer + classifier
pipeline = make_pipeline(
    TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words='english'),
    LogisticRegression(max_iter=1000)
)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
val_preds = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, val_preds)
print(f"Validation AUC LogisticRegression: {auc:.4f}")



#  Vectorize with TF-IDF
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_test)
df_text_vec = vectorizer.transform(df_test['text'])


# Train XGBoost classifier
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=1000)

xgb_model.fit(X_train_vec, y_train)

# 5. Validation
val_preds = xgb_model.predict_proba(X_val_vec)[:, 1]
val_auc = roc_auc_score(y_test, val_preds)
print(f"Validation AUC with XGBoost: {val_auc:.4f}")


from lightgbm import LGBMClassifier
# Train LGBM
lgbm_model = LGBMClassifier(n_estimators=1000, random_state=42,verbosity=-1)
lgbm_model.fit(X_train_vec, y_train)

# Validate
val_preds_lgbm = lgbm_model.predict_proba(X_val_vec)[:, 1]
val_auc_lgbm = roc_auc_score(y_test, val_preds_lgbm)
print(f"Validation AUC with LightGBM: {val_auc_lgbm:.4f}")


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_vec, y_train)

# Validate
val_preds_rf = rf_model.predict_proba(X_val_vec)[:, 1]
val_auc_rf = roc_auc_score(y_test, val_preds_rf)
print(f"Validation AUC with Random Forest: {val_auc_rf:.4f}")



from catboost import CatBoostClassifier
# Train CatBoost
cat_model = CatBoostClassifier(
    iterations=27,
    learning_rate=0.1,
    depth=6,
    eval_metric='AUC',
    verbose=100,
    random_seed=42
)

# CatBoost can work directly with sparse matrices like TF-IDF
cat_model.fit(X_train_vec, y_train, eval_set=(X_val_vec, y_test), use_best_model=True)

# Validate
val_preds_cat = cat_model.predict_proba(X_val_vec)[:, 1]
val_auc_cat = roc_auc_score(y_test, val_preds_cat)
print(f"Validation AUC with CatBoost: {val_auc_cat:.4f}")



# Predict on test set
test_preds = cat_model.predict_proba(df_text_vec)[:, 1]

# Submission
sample_submission['rule_violation'] = test_preds
sample_submission.to_csv('submission.csv', index=False)



sample_submission.head(5)




