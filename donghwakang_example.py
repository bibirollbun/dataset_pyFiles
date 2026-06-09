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


## import packages
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

print(f"import packages")


## Data Load
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.info()


object_cols = [
    'gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status'
] 

for col in object_cols:
    print(f"### {col} ê³ ìœ ê°’ í™•ì�¸ ### ")
    print(train[col].value_counts())
    print("-" * 30)


## IDì™€ Target ë¶„ë¦¬
train_ids = train['id']
test_ids = test['id']

y = train['diagnosed_diabetes']
train = train.drop(['id', 'diagnosed_diabetes'], axis=1)
test = test.drop(['id'], axis=1)


## object data to enumerical data
education_level_map = {
    'No formal':0,
    'Highschool':1, 
    'Graduate':2,
    'Postgraduate':3    
}

income_level_map = {
    'Low':0,
    'Lower-Middle':1,
    'Middle':2,
    'Upper-Middle':3,
    'High':4
}

train['education_level'] = train['education_level'].map(education_level_map)
test['education_level'] = test['education_level'].map(education_level_map)

train['income_level'] = train['income_level'].map(income_level_map)
test['income_level'] = test['income_level'].map(income_level_map)


## Object data to enumerical data (One-Hot Encoding)
train = pd.get_dummies(train, columns=['gender', 'ethnicity', 'smoking_status', 'employment_status'], drop_first=True)
test = pd.get_dummies(test, columns=['gender', 'ethnicity', 'smoking_status', 'employment_status'], drop_first=True)


test = test.reindex(columns=train.columns, fill_value=0)


display(train.head(3).T)


bool_cols = train.select_dtypes(include=['bool']).columns 
train[bool_cols] = train[bool_cols].astype(int)
test[bool_cols] = test[bool_cols].astype(int)


import xgboost as xgb
X = train 
X_test = test 


model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    device='cuda',
    early_stopping_rounds=100
)


## Cross validation and training
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
cv_scores = []


print("Start learning...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # í•™ìŠµ
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False # ë¡œê·¸ ë„ˆë¬´ ë§�ì�´ ëœ¨ë©´ ì •ì‹  ì‚¬ë‚˜ìš°ë‹ˆê¹Œ ë�”
    )
    
    # ê²€ì¦� ì �ìˆ˜ í™•ì�¸
    val_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_pred)
    cv_scores.append(score)
    print(f"Fold {fold+1} ROC-AUC: {score:.5f}")
    
    # í…ŒìŠ¤íŠ¸ ë�°ì�´í„° ì˜ˆì¸¡ ì �ë¦½
    test_preds += model.predict_proba(X_test)[:, 1] / 5

print("-" * 30)
print(f"ğŸ�† í�‰ê·  ê²€ì¦� ì �ìˆ˜: {np.mean(cv_scores):.5f}")


# 5. ì œì¶œ íŒŒì�¼ ë§Œë“¤ê¸°
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission['Diagnosed_Diabetes'] = test_preds
submission.to_csv('submission_final.csv', index=False)
print(f"done!")


# 1. íŒŒì�¼ ë¶ˆëŸ¬ì˜¤ê¸°
df = pd.read_csv('submission_final.csv')

# 2. ëŒ€ë¬¸ì�� 'D'ë¡œ ì‹œì�‘í•˜ëŠ”(ì†Œìˆ˜ì � ë“¤ì–´ì�ˆëŠ”) ì§„ì§œ ì»¬ëŸ¼ë§Œ ë‚¨ê¸°ê¸°
# ì ˆëŒ€ int(0/1)ë¡œ ë°”ê¾¸ì§€ ë§ˆì„¸ìš”! float(0.XXXX)ê°€ ì •ë‹µì�…ë‹ˆë‹¤.
df = df[['id', 'Diagnosed_Diabetes']]

# 3. ì €ì�¥
df.to_csv('submission_clean.csv', index=False)

print("âœ… ì™„ë£Œ! submission_clean.csv íŒŒì�¼ì�´ ìƒ�ì„±ë�˜ì—ˆìŠµë‹ˆë‹¤.")
print("â–¼ íŒŒì�¼ ë‚´ìš© í™•ì�¸ (ì†Œìˆ˜ì �ì�´ ë³´ì—¬ì•¼ ì •ìƒ�ì�…ë‹ˆë‹¤!) â–¼")
print(df.head())




