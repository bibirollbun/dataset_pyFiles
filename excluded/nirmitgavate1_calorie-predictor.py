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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import os,sys
import warnings

from sklearn.model_selection import train_test_split,KFold,GroupKFold,RepeatedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from lightgbm import LGBMClassifier
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
import shap


class Config:
    train_data_path="/kaggle/input/playground-series-s5e5/train.csv"
    test_data_path="/kaggle/input/playground-series-s5e5/test.csv"
    target='Calories'
    warnings_ignore=warnings.filterwarnings('ignore')
    seed=42


Config.warnings_ignore


train_df=pd.read_csv(Config.train_data_path)
test_df=pd.read_csv(Config.test_data_path)


train_df.head()


train_df.info()


train_df.describe()


test_df.head()


test_df.info()


test_df.describe()


numeric_cols = train_df.select_dtypes(include='number').columns
object_cols=train_df.select_dtypes(include='object').columns.tolist()


# def remove_outliers(df, cols):
#     outlier_indices = set()
#     for col in cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         upper_bound = Q3 + 1.5 * IQR
#         lower_bound = Q1 - 1.5 * IQR
#         outlier_idx = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
#         outlier_indices.update(outlier_idx)
#     return df.drop(index=outlier_indices)
    


# train_df=remove_outliers(train_df,numeric_cols)


train_df['Sex']=train_df['Sex'].map({"male":1,"female":0})
test_df['Sex']=test_df['Sex'].map({"male":1,"female":0})


train_df.head()


def FE(train, test):
    def apply_features(df):
        
        df['BMI']=df['Weight']/((df['Height']/100)**2)
        
        for f1 in ['Body_Temp','Age', 'Height']:
            for f2 in ['Weight', 'Duration', 'Heart_Rate']:
                df[f'{f1}_x_{f2}']=df[f1]*df[f2]
                df[f'{f1}_+_{f2}']=df[f1]+df[f2]
                df[f'{f1}_-_{f2}']=df[f1]-df[f2]
                df[f'{f1}_by_{f2}']=df[f1]/(df[f2]+1e-5)
                
        for f1 in ['Heart_Rate', 'Body_Temp']:
            df[f'sin_{f1}'] = np.sin(df[f1])
            df[f'sin_{f1}'] = np.sin(df[f1])

        df['BMI_range'] =pd.cut(df['BMI'],bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, 60],labels=False)
        df['age_range']=pd.cut(df['Age'],bins=[0, 12, 18, 30, 45, 60, 80],labels=False)

        for i,f1 in enumerate(['Duration', 'Heart_Rate','Body_Temp','Weight','Height']):
            max_val=df[f1].max()
            min_val=df[f1].min()
            df[f'{f1}_maxdiff']=max_val-df[f1]
            df[f'{f1}_mindiff']=df[f1]-min_val

        for f1 in (['Age', 'Height','Sex']):
            for f2 in (['Duration', 'Heart_Rate','Body_Temp','Weight']):
                temp_df = df.groupby(f1)[f2].mean().reset_index().rename(columns={f2: f'{f2}_{f1}_mean'})
                df = df.merge(temp_df, on=f1, how='left')    
                df[f'diff{f1}mean_grp{f2}']=df[f2]-df[f'{f2}_{f1}_mean']
                df[f'add{f2}mean_grp{f1}']=df[f2]+df[f'{f2}_{f1}_mean']
                # df=df.drop(f'{f2}_{f1}_mean',axis=1)
        
        for f1 in ['Age','Body_Temp']:
            df[f'{f1}_log']=np.log1p(df[f1])
                

        for f1 in ['Duration', 'Heart_Rate','Body_Temp','Weight','Age', 'Height']:
            df[f'{f1}_squared']=df[f1]**2

        for f1 in ['Heart_Rate','Body_Temp']:
            for f2 in ['Weight','Age', 'Height']:
                df[f'dur_{f1}_x_{f2}']=df['Duration']*df[f1]*df[f2]
                df[f'dur_by_{f1}_{f2}']=df['Duration']/((df[f1]*df[f2])+1e-5)
        
            
        return df

    train = apply_features(train)
    test = apply_features(test)

    return train, test

train_df, test_df = FE(train_df, test_df)


X = train_df.drop(columns=['Calories'])
y = np.log1p(train_df['Calories'])


xgb_params_best={
'n_estimators': 835,
 'max_depth': 11,
 'learning_rate': 0.013,
 'subsample': 0.847,
 'colsample_bytree': 0.55,
 'gamma': 0.84,
 'reg_alpha': 3.22,
 'reg_lambda': 0.730,
 'min_child_weight': 8,
 'device':'gpu'
}


kf=KFold(n_splits=5)
xgb_oof=np.zeros(len(X))
xgb_test_preds=np.zeros(len(test_df))
for i, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train,y_train=X.iloc[train_index],y.iloc[train_index]
    X_valid,y_valid=X.iloc[valid_index],y.iloc[valid_index]
    
    model=Pipeline([
    ("scaler",StandardScaler()),
    ("xgb",XGBRegressor(**xgb_params_best))
    ])
    
    model.fit(X_train,y_train)
    
    valid_preds=model.predict(X_valid)
    xgb_oof[valid_index]=valid_preds
    
    rmse_score=np.sqrt(mean_squared_error(y_valid,valid_preds))
    xgb_test_preds+=model.predict(test_df)
    print(f"#######Fold {i+1}#############")
    print(f'RMSE score for Fold {i+1}:{rmse_score:.4f}')
    
final_rmse = np.sqrt(mean_squared_error(y, xgb_oof))
print(f"\nFinal  XGBOOST OOF RMSE: {final_rmse:.5f}")
xgb_preds=np.expm1(xgb_test_preds/kf.get_n_splits())


cat_params = {
    'iterations': 2500,
    'learning_rate': 0.02,
    'depth': 10,
    'loss_function': 'RMSE',
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 200,
    'verbose': 0,
    'task_type': 'GPU'
}


cb_oof=np.zeros(len(X))
cb_test_preds=np.zeros(len(test_df))
for i, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train,y_train=X.iloc[train_index],y.iloc[train_index]
    X_valid,y_valid=X.iloc[valid_index],y.iloc[valid_index]
    
    model=Pipeline([
    ("scaler",StandardScaler()),
    ("catoost",CatBoostRegressor(**cat_params))
    ])
    
    model.fit(X_train,y_train)
    
    valid_preds=model.predict(X_valid)
    cb_oof[valid_index]=valid_preds
    
    rmse_score=np.sqrt(mean_squared_error(y_valid,valid_preds))
    cb_test_preds+=model.predict(test_df)
    print(f"#######Fold {i+1}#############")
    print(f'RMSE score for Fold {i+1}:{rmse_score:.4f}')
    
final_rmse = np.sqrt(mean_squared_error(y, cb_oof))
print(f"\nFinal Catboost OOF RMSE: {final_rmse:.5f}")
cb_preds=np.expm1(cb_test_preds/kf.get_n_splits())


weight_cb=1/cb_preds
weight_xgb=1/xgb_preds
weight_sum=weight_cb+weight_xgb
weight_xgb/=weight_sum
weight_cb/=weight_sum


preds=weight_xgb*xgb_preds+cb_preds*weight_cb


sub_df=pd.DataFrame({"id":test_df['id'],"Calories":preds})


sub_df.to_csv("submission.csv",index=False)
print("File saved successfully!")


sub_df.head()


sns.histplot(data=sub_df,x='Calories',bins=50,kde=True)
plt.title("Predicted Calories",color='red',fontsize=12)
plt.show()

