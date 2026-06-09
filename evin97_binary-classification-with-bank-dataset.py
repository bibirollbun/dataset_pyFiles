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


tr = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
tt = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier



def eda(df):
  print('head of dataset\n')
  print(df.head(10))
  print('\n description about dataset\n')
  print(df.describe())
  print('\n info about dataset\n')
  print(df.info())
  print('\n shape of dataset\n')
  print(df.shape)
  print('\n columns of dataset\n')
  print(df.columns)
  print('\n null values \n')
  print(df.isnull().sum())
  print('')

eda(tr)


tr['job'].unique()


tr.columns


numerical_cols =['age','balance','day','duration','campaign','pdays','previous']
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']

from sklearn.preprocessing import LabelEncoder
def preprocessing (df,cat_cols):
    #encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    encoder= LabelEncoder()
    for cols in cat_cols:
        df[cols] = encoder.fit_transform(df[cols])

preprocessing(tr,cat_cols)        
preprocessing(tt,cat_cols)


print(tr.head(20))


tt.columns


tr.drop(['default','contact'],axis=1, inplace=True)
tt.drop(['default','contact'],axis=1,inplace=True)


# Split features and target
X = tr.drop(['y','id'], axis=1)
y = tr['y']
test_data = tt.drop('id',axis=1)

#model evaluation technique
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
y_probs = np.zeros(len(test_data))

for fold ,(train_idx,val_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}/5 >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=20000,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=10,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.79,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(500),
            lgb.log_evaluation(period=500)
        ])
    
    
    y_probs += model.predict_proba(test_data)[:, 1]/5  # Ensemble test predictions



sub = pd.DataFrame({'id':tt['id'],'y':y_probs})
print(sub)
sub.to_csv("/kaggle/working/submission.csv",index=False)
print("file saved successfully")


import matplotlib.pyplot as plt
lgb.plot_importance(model, max_num_features=20)
plt.show()

