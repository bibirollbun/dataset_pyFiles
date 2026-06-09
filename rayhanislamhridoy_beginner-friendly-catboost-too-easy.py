import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from catboost import CatBoostClassifier


train= pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test= pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample= pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
train.drop(columns="id",inplace= True)
test.drop(columns="id",inplace= True)


x= train.drop(columns='diagnosed_diabetes')
y= train['diagnosed_diabetes']
cat_cols= x.select_dtypes(include='object').columns.to_list()
num_cols= x.select_dtypes(include=np.number).columns.to_list()


'''
X_train, X_valid, y_train, y_valid= train_test_split(x,y,test_size=0.2, random_state=42)

cat= CatBoostClassifier(random_seed=42, eval_metric='AUC', verbose=100)
cat.fit(X_train, y_train, cat_features= cat_cols, eval_set=(X_valid, y_valid), use_best_model=True)
preds= cat.predict_proba(X_valid)[:,1]
auc= roc_auc_score(y_valid, preds)
print(f"AUC on validation set: {auc}")
'''


# FInal model

cat2=CatBoostClassifier(random_seed= 42, eval_metric="AUC", verbose= 100)
cat2.fit(x,y, cat_features= cat_cols)
y_pred=cat2.predict_proba(test)[:,1]


# creat csv file for submission
sample['diagnosed_diabetes']= y_pred
sample.to_csv("submission.csv",index=False)


sample




