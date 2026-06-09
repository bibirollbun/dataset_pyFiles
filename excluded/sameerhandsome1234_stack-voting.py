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
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns  
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb



train_df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_df=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


crop_counts = train_df['Crop Type'].value_counts()


plt.figure(figsize=(6,6))
plt.pie(crop_counts, labels=crop_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Crop Types')
plt.axis('equal')
plt.show()



soil_counts = train_df['Soil Type'].value_counts()


plt.figure(figsize=(6,6))
plt.pie(soil_counts, labels=soil_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Soil Types')
plt.axis('equal')
plt.show()



train_df[['Temparature', 'Humidity', 'Moisture']].hist(bins=20, figsize=(12, 5))
plt.suptitle("Histograms of Temperature, Humidity, and Moisture")
plt.show()


cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    train_df[col] = LabelEncoder().fit_transform(train_df[col])
    test_df[col] = LabelEncoder().fit_transform(test_df[col])


le = LabelEncoder()
train_df["Fertilizer Name"] = le.fit_transform(train_df["Fertilizer Name"])
target_classes = le.classes_


X=train_df.drop(columns=['id','Fertilizer Name'])
y=train_df['Fertilizer Name']
X_test=test_df.drop(columns='id')


numeric_cols=X.select_dtypes(include=np.number).columns.tolist()


scaler=StandardScaler()
scaler.fit(X[numeric_cols])
X[numeric_cols]=scaler.transform(X[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])


base_learners = [
    ('xgb', xgb.XGBClassifier(
        objective='multi:softprob', num_class=7,
        tree_method='hist', use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    
    ('lgb', lgb.LGBMClassifier(
        objective='multiclass', num_class=7,
        verbose=0,
        device='gpu', random_state=42)),
    
    ('cat', cb.CatBoostClassifier(
        loss_function='MultiClass',task_type='GPU',
        verbose=0, random_state=42)),
    
    ('hist', HistGradientBoostingClassifier(
        loss='log_loss', max_iter=300, random_state=42))  
]

# Meta learner 
meta_model = LogisticRegression(multi_class='multinomial', max_iter=1000)

# Create StackingClassifier
stack_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,
    passthrough=False,
    n_jobs=1
)

# MAP@3 scorer
def mapk(y_true, y_pred_topk, k=3):
    def apk(actual, predicted):
        return 1.0 / (predicted.index(actual) + 1) if actual in predicted else 0.0
    return np.mean([apk(a, p[:k]) for a, p in zip(y_true, y_pred_topk)])

def map3_score(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
    return mapk(y_val.tolist(), top_3_preds.tolist(), k=3)

# Cross-validation with  MAP@3
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    stack_clf.fit(X_train, y_train)
    score = map3_score(stack_clf, X_val, y_val)
    scores.append(score)
    print(f"Fold {fold+1} MAP@3: {score:.4f}")

print(f"\nMean MAP@3: {np.mean(scores):.4f}")



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
global2 = None  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold != 2:
        continue

    print(f"\nRunning Fold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    stack_clf.fit(X_train, y_train)

    score = map3_score(stack_clf, X_val, y_val)
    scores.append(score)
    print(f"Fold {fold+1} MAP@3: {score:.4f}")

    global2 = stack_clf.predict_proba(X_test)
    print(f"Stored predictions for X_test in global2")

print(f"\nFold 3 MAP@3: {score:.4f}")


from IPython.display import FileLink

top_3_preds = np.argsort(global2, axis=1)[:, -3:][:, ::-1] 

flat_preds = top_3_preds.ravel()
top_3_labels = le.inverse_transform(flat_preds).reshape(top_3_preds.shape)

submission_df['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]


submission_df.to_csv('fsubmission10.csv', index=False)
FileLink("fsubmission10.csv")


from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier


# Define base models
base_models = [
    ('xgb', xgb.XGBClassifier(
        objective='multi:softprob', num_class=7,
        tree_method='hist', use_label_encoder=False,
        eval_metric='mlogloss', random_state=42
    )),
    
    ('lgb', lgb.LGBMClassifier(
        objective='multiclass', num_class=7,
        device='gpu', verbose=0, random_state=42
    )),
    
    ('cat', CatBoostClassifier(
        loss_function='MultiClass', task_type='GPU',
        devices='0', verbose=0, random_state=42
    )),
    
    ('hist', HistGradientBoostingClassifier(
        loss='log_loss', max_iter=500, random_state=42
    ))
]

# VotingClassifier with soft voting
voting_clf = VotingClassifier(
    estimators=base_models,
    voting='soft',
    n_jobs=-1 
)

# MAP@3 scorer
def mapk(y_true, y_pred_topk, k=3):
    def apk(actual, predicted):
        return 1.0 / (predicted.index(actual) + 1) if actual in predicted else 0.0
    return np.mean([apk(a, p[:k]) for a, p in zip(y_true, y_pred_topk)])

def map3_score(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
    return mapk(y_val.tolist(), top_3_preds.tolist(), k=3)

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    voting_clf.fit(X_train, y_train)
    score = map3_score(voting_clf, X_val, y_val)
    scores.append(score)
    print(f"Fold {fold+1} MAP@3: {score:.4f}")

print(f"\nMean MAP@3: {np.mean(scores):.4f}")



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
global2 = None  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold != 1:
        continue

    print(f"\nRunning Fold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    voting_clf.fit(X_train, y_train)

    score = map3_score(voting_clf, X_val, y_val)
    scores.append(score)
    print(f"Fold {fold+1} MAP@3: {score:.4f}")

    global2 = voting_clf.predict_proba(X_test)
    print(f"Stored predictions for X_test in global2")

print(f"\nFold 2 MAP@3: {score:.4f}")


from IPython.display import FileLink

top_3_preds = np.argsort(global2, axis=1)[:, -3:][:, ::-1] 

flat_preds = top_3_preds.ravel()
top_3_labels = le.inverse_transform(flat_preds).reshape(top_3_preds.shape)

submission_df['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]


submission_df.to_csv('fsubmission11.csv', index=False)
FileLink("fsubmission11.csv")


file1=pd.read_csv('/kaggle/input/submissionfile/fsubmission10.csv')
file2=submission_df.copy()


# from collections import Counter
# w1, w2 = 0.4, 0.6

# classes = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP','Urea']
# le = LabelEncoder()
# le.fit(classes)

# # Function to encode fertilizer predictions
# def encode_preds(df):
#     return np.array([le.transform(row.split()) for row in df['Fertilizer Name']])

# pred1 = encode_preds(file1)
# pred2 = encode_preds(file2)

# # Blending function
# def weighted_blend_top3(p1, p2, w1, w2):
#     blended_preds = []
#     for row1, row2 in zip(p1, p2):
#         combined = list(row1) * int(w1 * 10) + list(row2) * int(w2 * 10)
#         top3 = [item for item, _ in Counter(combined).most_common(3)]
#         blended_preds.append(top3)
#     return np.array(blended_preds)

# # Get blended predictions
# blended = weighted_blend_top3(pred1, pred2, w1, w2)

# # Decode to class names
# decoded_preds = [' '.join(le.inverse_transform(row)) for row in blended]

# # Prepare final submission
# submission_df = file1[['id']].copy()
# submission_df['Fertilizer Name'] = decoded_preds
# submission_df.to_csv('fsubmission12.csv', index=False)
# FileLink("fsubmission12.csv")

