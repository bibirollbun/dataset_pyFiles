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
import seaborn as sns
from sklearn.linear_model import LinearRegression,RidgeCV
import xgboost as xgb
from sklearn.model_selection import train_test_split,StratifiedKFold ,KFold
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,mean_absolute_error

import random
import warnings 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost.callback import EarlyStopping


warnings.simplefilter(action = 'ignore', category = FutureWarning)


train_d = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_d = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_d.head(5)


train_d.describe()


train_d.shape,test_d.shape


#let's check wheather the target variable is balanced or not 
train_d.value_counts('accident_risk')


train_d.info()


print("printing  unique values of all columns")

for col in train_d:
    print("-"*60)
    unq = train_d[col].unique()
    if len(unq)>15 :
        print(col,":", unq[:5],"....", "so much unique values")
    else:
        print(col,":", unq)


# checking missing values 
train_d.isnull().sum()


num_col = train_d.select_dtypes(include = ['int64','float64']).columns
random_search = ['royalblue', 'seagreen', 'orange', 'crimson', 'purple', 'gold', 'teal', 'tomato']

plt.figure(figsize = (14,12))
for i,col in enumerate(num_col,start = 1):
    choose_color = random.choice(random_search)
    plt.subplot(3,3,i)
    sns.histplot(x = col,data = train_d,kde = True,color = choose_color,bins = 20 )

plt.tight_layout()
plt.show()


train_d.head(3)


# countplot of target variable 
plt.figure(figsize = (14,12))
object_col = train_d.select_dtypes('object').columns
colors = ('Set1','Set2','Set3')
for i,col in enumerate(object_col,1):
    random_color = random.choice(colors)
    plt.subplot(3,3,i)
    sns.countplot(x = col ,data= train_d,palette =random_color )
    plt.xticks(rotation = 90)
    plt.title(f"Countplot of {col}")

plt.tight_layout()
plt.show()


train_d = train_d.drop('id',axis = 1)
test_d = test_d.drop('id',axis = 1)


# applying label encoder 
ln = LabelEncoder()

for col in object_col:
    train_d[col] = ln.fit_transform(train_d[col])
    test_d[col] = ln.transform(test_d[col])


train_d.head(2)


bool_col = train_d.select_dtypes('bool').columns
for col in bool_col:
    train_d[col] = train_d[col].astype(int)
    test_d[col] = test_d[col].astype(int)


scaler = StandardScaler()
num_float = test_d.select_dtypes(include = ['int64','float64']).columns
train_d[num_float] = scaler.fit_transform(train_d[num_float])
test_d[num_float] = scaler.transform(test_d[num_float])


x = train_d.drop('accident_risk',axis = 1)
y = train_d['accident_risk']
test = test_d.copy()


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


# setting the hyper parameter for the xgboost
params = {
    "n_estimators" : 1000,
    'objective': 'reg:squarederror',
    'n_jobs' : -1,
    'eval_metric': 'rmse',
    'max_depth' : 7,
    'learning_rate': 0.005,
    'enable_categorical': True,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'lambda':2.0,
    'alpha':1.0,

    'tree_method': 'hist',

    
}


seds = [42,128,265,510,3000]
cols = [f"seed {seed}" for seed in seds]


oof_xgb_full = []
xgb_preds_full = []
for seed in seds:
    print(f" SEED IS : {seed}\n")


    oof_xgb = np.zeros(x.shape[0])
    xgb_preds = np.zeros(test.shape[0])

    n_splits = 5
    kf = KFold(n_splits = n_splits,shuffle = True,random_state = seed)
    
    for fold,(train_idx,val_idx) in enumerate(kf.split(x,y)):
        x_train,y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_val,y_val = x.iloc[val_idx],y.iloc[val_idx]
        early_stop = EarlyStopping(rounds = 300,metric_name = 'rmse',data_name = 'validation_0')

        local_params = params.copy()
        local_params['seed'] = seed
        local_params['callbacks'] = [early_stop]
        
        model = xgb.XGBRegressor(**local_params)
        
        model.fit(
            x_train,
            y_train,
            eval_set = [(x_val,y_val)],
            verbose = 500
        )
        oof_xgb[val_idx] = model.predict(x_val)
        test_preds = model.predict(test)

        xgb_preds += test_preds/n_splits

        fold_rmse = mean_squared_error(y_val,oof_xgb[val_idx],squared = False)
        print("\n")


        print(f"FOLD {fold+1} -> validation MSE {fold_rmse:.4f}")
    oof_xgb_full.append(oof_xgb)
    xgb_preds_full.append(xgb_preds)




cols = [f"seed{seeds}"for seeds in seds]
oof_xgb_full_array = np.array(oof_xgb_full)
xgb_preds_full_array = np.array(xgb_preds_full)

oof_xgb_full_df = pd.DataFrame(oof_xgb_full_array.T,columns = cols,index = x.index)
xgb_preds_full_df = pd.DataFrame(xgb_preds_full_array.T,columns = cols,index = test.index)

preds_rmse = xgb_preds_full_df.mean(axis = 1)

print("Printing performance of our seed ")
for col in oof_xgb_full_df.columns.tolist():
    rmse = mean_squared_error(y,oof_xgb_full_df[col],squared = False)
    mae = mean_absolute_error(y,oof_xgb_full_df[col])
    mse = mean_squared_error(y,oof_xgb_full_df[col])

    print(f"Performance of {col}")
    print(f"RMSE : {rmse}")
    print(f"MAE : {mae}")
    print(f"MSE : {mse}")
    print("\n")


alphas = [1e-3,1e-2,0.05,0.1,0.3,1.0,3.0,10.0]
meta_model = RidgeCV(alphas = alphas,scoring = 'neg_mean_squared_error',cv = 5)
meta_model.fit(oof_xgb_full_df,y)


# now predictions 
oof_df_preds = meta_model.predict(oof_xgb_full_df)
xgb_df_preds = meta_model.predict(xgb_preds_full_df)

oof_rmse = mean_squared_error(y,oof_df_preds,squared = False)
print(f"RMSE : {oof_rmse}")
print(f"Best Alpha : {meta_model.alpha_}")


# submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
submission['accident_risk'] = preds_mean if rmse < oof_rmse else xgb_df_preds
submission.to_csv('submission.csv',index = False)
print("submission dataset is made successfullâœ…")


submission.head()




