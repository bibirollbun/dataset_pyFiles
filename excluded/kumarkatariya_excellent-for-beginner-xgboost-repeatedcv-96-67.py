# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Your Goal: Your goal is to predict whether a client will subscribe to a bank term deposit.

#Will the client say "Yes" to opening a fixed-term deposit account or not?


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')



train_df.head()


train_df['y'].value_counts()


test_ids = train_df['id']



train_df.drop('id',axis=1,inplace=True)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_df.head()


# external_df = pd.read_csv('/kaggle/input/bank-full-csv/bank-full.csv',sep=';')
# external_df.head()
# external_df.shape


# external_df = external_df.rename(columns = {'y':'match_y'}).drop_duplicates(['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])

# external_df.head()


# now lets merge actual training data and external data

# updated_train_df = pd.concat([train_df,external_df],ignore_index=True)
# updated_train_df.head()


# updated_train_df.shape


# updated_train_df.isna().sum()





num_cols = train_df.select_dtypes(include = ['int','float']).columns
cat_cols = train_df.select_dtypes(include = ['object']).columns

print(num_cols)
print(cat_cols)


train_df.isna().sum()


train_df.head()


train_df['job'].value_counts()


from sklearn.preprocessing import LabelEncoder


for col in cat_cols:
    LE = LabelEncoder()
    train_df[col] = LE.fit_transform(train_df[col])
    test_df[col] = LE.transform(test_df[col])



train_df.head()



test_ids = test_df['id']
test_df.head()


test_df.drop('id',axis=1,inplace=True)


test_df.head()


from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

xg = XGBClassifier(random_state=42)




y = train_df['y'] 



train_df.head()


X = train_df.drop('y',axis=1)
X.head()





fold = RepeatedStratifiedKFold(n_repeats = 10,n_splits = 5, random_state=42)   
for train_idx,val_idx in fold.split(X,y):
    X_train,X_valid = X.iloc[train_idx],X.iloc[val_idx]
    y_train,y_valid = y.iloc[train_idx],y.iloc[val_idx]
    xg.fit(X_train,y_train)
    y_preds_proba = xg.predict_proba(X_valid)[:,1]
    auc = roc_auc_score(y_valid, y_preds_proba)
    print(f"ROC AUC: {auc:.4f}")
 


sample = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
sample.head()


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_valid, y_preds_proba)



import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label="Random Classifier (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(True)
plt.show()


y_test_preds = xg.predict_proba(test_df)[:, 1] 


y_test_preds


submission = pd.DataFrame({'id':test_ids,
                           'y':y_test_preds 
})


submission.to_csv('submission.csv',index=False)




