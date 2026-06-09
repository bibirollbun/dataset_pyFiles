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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


df_train.info()


df_train['diagnosed_diabetes'].value_counts()


df_test.info()


features=df_train.drop(['diagnosed_diabetes'],axis=1).columns
y = df_train['diagnosed_diabetes']
X = df_train[features].copy()
X_test = df_test[features].copy()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in num_cols]

for c in num_cols:
    med = X[c].median()
    X[c]=X[c].fillna(med)
    X_test[c]=X_test[c].fillna(med)

for c in cat_cols:
    X[c] = X[c].fillna('Missing').astype(str)
    X_test[c] = X_test[c].fillna('Missing').astype(str)

if y.dtype == object or y.dtype == 'bool':
    le = LabelEncoder()
    y = le.fit_transform(y)


df_train.drop(cat_cols,axis=1).corr()['diagnosed_diabetes'].sort_values()


# from sklearn.preprocessing import OrdinalEncoder
# enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value =-1)
# enc.fit(pd.concat([X[cat_cols], X_test[cat_cols]], axis=0))
X_enc = X.copy()
X_test_enc = X_test.copy()
# X_enc[cat_cols] = enc.transform(X[cat_cols])
# X_test_enc[cat_cols] = enc.transform(X_test[cat_cols])


from catboost import CatBoostClassifier

scoring_model = CatBoostClassifier(
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    task_type='GPU'
)

scoring_model.fit(X_enc, y, cat_features=cat_cols)
proba = scoring_model.predict_proba(X_enc)[:, 1]
sample_score = 1.0 - np.abs(y - proba)
sample_weight = np.clip(sample_score, 0.1, 1.0)


from catboost import CatBoostClassifier
from xgboost import XGBClassifier
model = CatBoostClassifier(verbose=False,
                           random_seed=42,
                          task_type='GPU',
                          eval_fraction=0.2,
                          eval_metric='AUC')#auto_class_weights='Balanced'

model.fit(X_enc,y,cat_features=cat_cols,sample_weight=sample_weight)


model.best_score_


from sklearn.metrics import accuracy_score,roc_auc_score
y_predict=model.predict(X_enc)
y_proba = model.predict_proba(X_enc)
auc_p = roc_auc_score(y, y_proba[:, 1])
acc=accuracy_score(y,y_predict)
print(f"AUC_p 值: {auc_p:.4f}") 
print(f"ACC 值: {acc:.4f}")


y_pred_test=model.predict_proba(X_test_enc)
df_submit = pd.DataFrame({
    'id': df_test['id'].values,
    'diagnosed_diabetes': y_pred_test[:, 1]
})
df_submit=df_submit.set_index('id')


df_submit.to_csv("/kaggle/working/submission.csv")


df_submit

