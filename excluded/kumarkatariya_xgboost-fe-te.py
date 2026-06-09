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


# %load_ext cudf.pandas


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col = 'id')
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test.head()


y = train['accident_risk'] 


train.drop('accident_risk',axis=1,inplace=True)


train.head()


test_ids = test['id']


test.drop('id',axis=1,inplace=True)


orig1 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
orig2 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
orig3 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')


orig = pd.concat([orig1,orig2,orig3],axis=0)


orig.head()


# now will be doing feature interactions num*num

num = train.select_dtypes(include=['int','float']).columns.tolist()
cat = train.select_dtypes(include=['object','bool']).columns.tolist()



from itertools import combinations
from tqdm import tqdm


num


columns = num  # or CATS, depending on what you're doing

for r in [2]:
    for pair in tqdm(list(combinations(columns, r))):
        name = '_x_'.join(pair)
        
        train[name] = train[pair[0]]
        for col in pair[1:]:
            train[name] = train[name] * train[col]
        
        test[name] = test[pair[0]]
        for col in pair[1:]:
            test[name] = test[name] * test[col]
        
        # orig[name] = orig[pair[0]]
        # for col in pair[1:]:
        #     orig[name] = orig[name] * orig[col]
 
    
    


columns = cat + num 

TE_columns = []

for r in [2]:
    for cols in tqdm(list(combinations(columns,r))):
        name = '_+_'.join(cols)

        train[name] = train[cols[0]].astype('str')
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype('str')

        test[name] = test[cols[0]].astype('str')
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype('str')

        # orig[name] = orig[cols[0]].astype('str')
        # for col in cols[1:]:
        #     orig[name] = orig[name] + '_' + orig[col].astype('str')

        # combined = pd.concat([train[name],test[name],orig[name]],ignore_index=True)
        combined = pd.concat([train[name],test[name]],ignore_index=True)
        combined,_ = combined.factorize()
        
        train[name] = combined[:len(train)]
        test[name] =  combined[len(train):]
        # orig[name] = combined[len(train)+len(test):]
        TE_columns.append(name)

# columns_cc = num + cat + TE_columns

# for c in columns_cc: 
#     TE = orig.groupby(c)['accident_risk'].mean()
#     TE.name = f'TE_{c}'

#     train = train.merge(TE, on=c, how='left')
#     test = test.merge(TE, on=c, how='left')

#     train[TE.name] = train[TE.name].fillna(train[TE.name].mean())
#     test[TE.name] = test[TE.name].fillna(train[TE.name].mean()) 


train.info()


test.info()


train.select_dtypes('object')


train.info()


for col in train.select_dtypes(include=['object','bool']).columns:
    train[col] = train[col].astype('category')
    test[col]=pd.Categorical(test[col],categories=train[col].cat.categories)
    test[col] = test[col].astype('category')


test.info()


train.info()


train.head()


X = train.copy()


y


import warnings
warnings.filterwarnings('ignore')


columns_cc = num + cat + TE_columns
len(columns_cc)


len(num+cat)
    


test


# import cudf
# from cuml.preprocessing import TargetEncoder 
from category_encoders import TargetEncoder as CETargetEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=7,random_state=42, shuffle=True) 
oof_xgb = np.zeros(len(X))
test_preds = np.zeros(len(test))  
columns_cc = TE_columns 

for train_idx,val_idx in kf.split(X,y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train,y_val = y.iloc[train_idx],y.iloc[val_idx] 

    X_test = test.copy()

    for col in columns_cc:
        # use category_encoders TargetEncoder (CPU)
        TE = CETargetEncoder(cols=[col])   # minimal config; you can pass smoothing/min_samples_leaf if wanted

        # fit_transform expects a DataFrame for X
        X_train[f'TE_{col}'] = TE.fit_transform(X_train[[col]], y_train)[col].astype(float)

        # transform validation/test
        X_val[f'TE_{col}'] = TE.transform(X_val[[col]])[col].astype(float).fillna(y_train.mean())
        X_test[f'TE_{col}'] = TE.transform(X_test[[col]])[col].astype(float).fillna(y_train.mean())

    # for c in columns_cc:
    #     means = y_train.groupby(X_train[c]).mean()
    #     X_train[f'TE_{c}'] = X_train[c].map(means).astype(float)
    #     X_val[f'TE_{c}'] = X_val[c].map(means).astype(float) 
    #     X_val[f'TE_{c}'] = X_val[f'TE_{c}'].fillna(means.mean())

    #     # Test encoding averaged across folds
    #     if f'TE_{c}' not in test:
    #         test[f'TE_{c}'] = 0
    #     test[f'TE_{c}'] += test[c].map(means).astype(float).fillna(means.mean()) / kf.n_splits

    model = XGBRegressor(n_estimators=100000,
                    learning_rate=0.03,
                    random_state=42,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    enable_categorical=True,
                    #device = 'cuda'
                        )

    model.fit(X_train,y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=200,
              verbose=False)
    
    oof_xgb[val_idx] = model.predict(X_val) 
    test_preds += model.predict(X_test)/kf.n_splits

print(np.sqrt(mean_squared_error(y,oof_xgb)))  



pd.DataFrame({'xgb_oof':oof_xgb,'target':y}).to_csv('xgb_fe.csv',index=False)


submission = pd.DataFrame({'id': test_ids, 'accident_risk': test_preds})
submission.to_csv('submission.csv', index=False)







