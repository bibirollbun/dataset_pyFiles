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


%load_ext cudf.pandas


import pandas as pd       
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from cuml.preprocessing import TargetEncoder
from cuml.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

seed = 42


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

train = pd.concat([train, train_extra], axis=0, ignore_index=True)

train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)



def feature_engineering(df):
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Num'] = df['Size'].map(size_mapping)
    df['Compartments_per_Size'] = df['Compartments'] / df['Size_Num']    
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments'] 
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof_Laptop'] = df['Waterproof'] * df['Laptop Compartment']
    df['Is_Durable_Material'] = df['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df['Is_Lightweight_Material'] = df['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df['Luxury_Material'] = df['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    df['Professional_Style'] = df['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df['Casual_Style'] = df['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df['Is_Budget_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    df['Is_Small'] = df['Size'].apply(lambda x: 1 if x == 'Small' else 0)
    df['Is_Medium'] = df['Size'].apply(lambda x: 1 if x == 'Medium' else 0)
    df['Is_Large'] = df['Size'].apply(lambda x: 1 if x == 'Large' else 0)

    return df

train = feature_engineering(train)
test = feature_engineering(test)


target = "Price"
features = [col for col in train.columns if col != target]
CATS = [col for col in train.columns if col not in ["Price", "Weight Capacity (kg)"]]

for col in CATS:
    train[col] = train[col].fillna('Missing').astype(str)
    test[col] = test[col].fillna('Missing').astype(str)
    


TE = TargetEncoder(n_folds=5, smooth=20, split_method='random', stat='mean')

for col in CATS:
    train[f"TE_{col}"] = TE.fit_transform(train[col], train["Price"])
    test[f"TE_{col}"] = TE.transform(test[col])

all_features = features + [f"TE_{col}" for col in CATS]

train = train.drop(columns=CATS)
test = test.drop(columns=CATS)


X_train = train.drop(target, axis = 1)
Y_train = train[target].copy()


rf_model = RandomForestRegressor(n_estimators = 100, random_state = seed, n_streams = 1, bootstrap = False
                                 , max_depth = 10, max_features=7, min_samples_split = 10, min_samples_leaf = 5
                                 , accuracy_metric = 'mse')


%%time

def cv_function (X_train, Y_train, model, splits = 10, seed = seed):
    
    print('seed:', seed)

    kfold = KFold(n_splits = splits, shuffle=True, random_state = seed)
    rmse = []
   
    cv_pred = np.zeros(len(X_train))
    
    for idx in kfold.split(X=X_train, y=Y_train):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = X_train.iloc[train_idx]
        ytrain = Y_train.iloc[train_idx]
        xtest = X_train.iloc[test_idx]
        ytest = Y_train.iloc[test_idx]
        
        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)
        cv_pred[test_idx] = preds
                              
        fold_rmse = mean_squared_error(ytest,preds, squared=False)
        print("RMSE: {0:0.5f}". format(fold_rmse))
        rmse.append(fold_rmse)
        
    print (np.mean(rmse))
    return cv_pred

rf_cvpred = cv_function(X_train, Y_train, rf_model)


%%time


def prediction (X_train, Y_train, model, test, seed = seed):
    
    print('seed:', seed)
        
    kfold = KFold(n_splits = 10, shuffle=True, random_state = seed)
    
    y_pred = np.zeros(len(test))
    train_oof = np.zeros(len(X_train))
    
    for idx in kfold.split(X=X_train, y=Y_train):
        train_idx, val_idx = idx[0], idx[1]
        xtrain = X_train.iloc[train_idx]
        ytrain = Y_train.iloc[train_idx]
        xval = X_train.iloc[val_idx]
        yval = Y_train.iloc[val_idx]
        
        model.fit(xtrain, ytrain)

        y_pred += model.predict(test)/kfold.n_splits
        print(y_pred)
               
        val_pred = model.predict(xval)
        train_oof[val_idx] = val_pred

        rmse = mean_squared_error(yval,val_pred, squared=False)
        print('RMSE : {}'.format(rmse))
  
    return y_pred, train_oof

pred, train_oof = prediction (X_train, Y_train, rf_model, test)


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

output = pd.DataFrame({"id": sub.id, "Price": pred})
output.to_csv('submission_ensemble.csv', index=False)

output.head()

