# Author: Aaron Isom.
# Kaggle Playground-Series-S5e8 - Binary Classification with a Bank Dataset.
# Voting Classifier using CatBoost, LGBM, XGBoost, and HGBT. Drop the lowest scoring model.

from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=";")
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

original['y'] = original['y'].replace({'yes': 1, 'no': 0})

train = pd.concat([train, original], axis=0, ignore_index=True)

# Features for training (drop id and target)
X = train.drop(['id', 'y'], axis=1)
y = train['y']

# Features for test set (drop only id)
X_test = test.drop(['id'], axis=1)

# Encode object and category columns to ensure unique values are mapped
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    le.fit(list(X[col].astype(str)) + list(X_test[col].astype(str)))
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


# Create three voting classifiers - CatBoost, LGBM, XGBoost, and HGBT. Drop the lowest scoring model.
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

clf1 = LGBMClassifier(n_estimators=2000, objective='binary', random_state=42, metric='binary_logloss', verbose=0, scale_pos_weight=scale_pos_weight)
clf2 = XGBClassifier(n_estimators=2000, objective='binary:logistic', eval_metric='auc', random_state=42, n_jobs=-1, 
                          enable_categorical=True, tree_method='hist', verbose=0, scale_pos_weight=scale_pos_weight)
clf3 = CatBoostClassifier(n_estimators=2000, eval_metric='AUC', random_state=42, verbose=0, class_weights=[1, scale_pos_weight])
clf4 = HistGradientBoostingClassifier(max_iter=2000, random_state=42, verbose=0, class_weight='balanced')

models = [
    ('lgbm', clf1),
    ('xgb', clf2),
    ('cat', clf3),
    ('hgbt', clf4)
]

scores = {}
for name, model in models:
    score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    scores[name] = score
    print(f"{name} ROC-AUC: {score:.5f}")


# Sort by score and drop the lowest
sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
top3_names = [name for name, _ in sorted_models[:3]]

# Filter the original models to keep only the top 3
final_estimators = [item for item in models if item[0] in top3_names]
print("Top 3 models:", top3_names)

voting_clf = VotingClassifier(estimators=final_estimators, voting='soft')
voting_clf.fit(X, y)

preds = voting_clf.predict_proba(X_test)[:, 1]


# Final submission
submission['y'] = preds
submission.to_csv('submission.csv', index=False)
display(submission)
print('Submission file saved.')

