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


pip install  -q category_encoders


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder,StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

import optuna


train_df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_df.rename(columns={"Listening_Time_minutes": "target"}, inplace=True)
test_df.rename(columns={"Listening_Time_minutes": "target"}, inplace=True)


test_df.head()


train_df.info()


test_df.info()


numeric_cols=list(train_df.select_dtypes(exclude="object").columns)
object_cols=list(train_df.select_dtypes(include="object").columns)


train_df[numeric_cols].corr()['target']


train_null_percent=100*(train_df.isnull().sum()/len(train_df))
train_null_percent[train_null_percent>0].plot(kind='bar')


test_null_percent=100*(train_df.isnull().sum()/len(test_df))
test_null_percent[test_null_percent>0].plot(kind='bar')


train_df.describe().transpose()


train_df=train_df.drop_duplicates()
test_df=test_df.drop_duplicates()


minutes_median=train_df['Episode_Length_minutes'].median()
popularity_median=train_df['Guest_Popularity_percentage'].median()
ads_median=train_df['Number_of_Ads'].median()

train_df['Episode_Length_minutes']=train_df['Episode_Length_minutes'].fillna(minutes_median)
train_df['Guest_Popularity_percentage']=train_df['Guest_Popularity_percentage'].fillna(popularity_median)
train_df['Number_of_Ads']=train_df['Number_of_Ads'].fillna(ads_median)

test_df['Episode_Length_minutes']=test_df['Episode_Length_minutes'].fillna(minutes_median)
test_df['Guest_Popularity_percentage']=test_df['Guest_Popularity_percentage'].fillna(popularity_median)
test_df['Number_of_Ads']=test_df['Number_of_Ads'].fillna(ads_median)


fig,axes=plt.subplots(2,3,figsize=(12,6))
ax=axes.flatten()
for i,col in enumerate(numeric_cols[:5]):
    sns.boxplot(data=train_df,x=col,ax=ax[i])
plt.tight_layout()
plt.show()


fig,axes=plt.subplots(2,3,figsize=(12,6))
ax=axes.flatten()
for i,col in enumerate(numeric_cols[:5]):
    sns.scatterplot(data=train_df.sample(1000),x=col,y='target',ax=ax[i])
plt.tight_layout()
plt.show()


# Q1=train_df['Number_of_Ads'].quantile(0.25)
# Q3=train_df['Number_of_Ads'].quantile(0.75)
# IQR=Q3-Q1
# lower_bound=Q1-Q1*1.5
# upper_bound=Q3+Q3*1.5
# train_df=train_df[~((train_df['Number_of_Ads']>upper_bound) | (train_df['Number_of_Ads']<lower_bound))]


# Q1=train_df['Episode_Length_minutes'].quantile(0.25)
# Q3=train_df['Episode_Length_minutes'].quantile(0.75)
# IQR=Q3-Q1
# lower_bound=Q1-Q1*1.5
# upper_bound=Q3+Q3*1.5
# train_df=train_df[~((train_df['Episode_Length_minutes']>upper_bound) | (train_df['Episode_Length_minutes']<lower_bound))]


def FE(train,test):
    train['host_guest_pop']=train['Host_Popularity_percentage']+train['Guest_Popularity_percentage']
    test['host_guest_pop']=test['Host_Popularity_percentage']+test['Guest_Popularity_percentage']
    
    train['ep_len_pop']=train['Episode_Length_minutes']+train['host_guest_pop']
    test['ep_len_pop']=test['Episode_Length_minutes']+test['host_guest_pop']
    
    train['rel_ep_len']=train['Episode_Length_minutes'].max()-train['Episode_Length_minutes']
    test['rel_ep_len']=train['Episode_Length_minutes'].max()-test['Episode_Length_minutes']
    
    train['ads_ep_len']=train['Number_of_Ads']*train['Episode_Length_minutes']
    test['ads_ep_len']=test['Number_of_Ads']*test['Episode_Length_minutes']

    # train['host_pop_binary']=(train['Host_Popularity_percentage']>70).astype(int)
    # test['host_pop_binary']=(test['Host_Popularity_percentage']>70).astype(int)
    
    # train['ep_len_binary']=(train['Episode_Length_minutes']>70).astype(int)
    # test['ep_len_binary']=(test['Episode_Length_minutes']>70).astype(int)
    
    # train['guest_pop_binary']=(train['Guest_Popularity_percentage']>70).astype(int)
    # test['guest_pop_binary']=(test['Guest_Popularity_percentage']>70).astype(int)

    return train,test
train_df,test_df=FE(train_df,test_df)


train_df.head()


X=train_df.drop("target",axis=1)
y=train_df['target']


# kf = KFold(n_splits=10, shuffle=True, random_state=42)
# errors = []
# all_idx = []
# for i, (train_index, test_index) in enumerate(kf.split(X)):
#     X_train, y_train = X.iloc[train_index].copy(), y.iloc[train_index]
#     X_test, y_test = X.iloc[test_index].copy(), y.iloc[test_index]
#     test_df_copy = test_df.copy()

#     # 1. Target encode only object columns
#     te = TargetEncoder(cols=object_cols)
#     X_train = te.fit_transform(X_train, y_train)
#     X_test = te.transform(X_test)
#     test_df_encoded = te.transform(test_df_copy)

#     # 2. Scale all features (after encoding)
#     scaler = StandardScaler()
#     scaled_X_train = scaler.fit_transform(X_train)
#     scaled_X_test = scaler.transform(X_test)
#     scaled_test_df = scaler.transform(test_df_encoded)

#     # 3. Train XGBRegressor with GPU
#     lgbm = LGBMRegressor(device="gpu")
#     lgbm.fit(scaled_X_train, y_train)

#     # 4. Evaluate
#     test_preds = lgbm.predict(scaled_X_test)
#     RMSE_test = np.sqrt(mean_squared_error(y_test, test_preds))
#     errors.append(RMSE_test)
#     all_idx.append((train_index, test_index))



# # Get best fold index
# best_idx = np.argmin(errors)
# best_train_index, best_test_index = all_idx[best_idx]

# # Split data
# X_train_best = X.iloc[best_train_index]
# y_train_best = y.iloc[best_train_index]
# X_test_best = X.iloc[best_test_index]
# y_test_best = y.iloc[best_test_index]

# te = TargetEncoder(cols=object_cols)
# X_train_best = te.fit_transform(X_train_best, y_train_best)
# X_test_best = te.transform(X_test_best) 
# test_df_encoded = te.transform(test_df_copy)

# # Scale
# scaler = StandardScaler()
# scaled_X_train_best = scaler.fit_transform(X_train_best)
# scaled_X_test_best = scaler.transform(X_test_best)
# scaled_test_df = scaler.transform(test_df_encoded)


# final_lgbm = LGBMRegressor(
#        device='gpu'
#     )
# final_lgbm.fit(scaled_X_train_best, y_train_best)

# final_preds = final_lgbm.predict(scaled_X_test_best)
# final_rmse = np.sqrt(mean_squared_error(y_test_best, final_preds))

# print(f"Best Fold RMSE: {final_rmse:.4f}")


# importances = final_lgbm.feature_importances_ 

# importances_pct = 100 * importances / importances.sum()


# feature_names = X_train.columns  
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances_pct
# }).sort_values(by='Importance', ascending=False)


# sns.barplot(y='Feature',x='Importance',data=importance_df,)


train_df.info()


cat_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype('category')



def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'seed': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }

    errors = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, valid_index in kf.split(X):
        X_train, y_train = X.iloc[train_index].copy(), y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index].copy(), y.iloc[valid_index]

        # Scale numeric columns only
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_valid[numeric_cols] = scaler.transform(X_valid[numeric_cols])

        model = LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[early_stopping(25), log_evaluation(0)],
            categorical_feature=cat_cols
        )

        preds = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds, squared=False)
        errors.append(rmse)

    return np.mean(errors)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25, show_progress_bar=True)


lgbm_best_params=study.best_params

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

te=TargetEncoder(cols=object_cols,smoothing=0.3)
X_train=te.fit_transform(X_train,y_train)
X_valid=te.transform(X_valid)
test_df=te.transform(test_df)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_valid = scaler.transform(X_valid)
scaled_test_df=scaler.transform(test_df)

final_lgbm=LGBMRegressor(**lgbm_best_params,device="gpu")
final_lgbm.fit(scaled_X_train,y_train)

test_preds=final_lgbm.predict(scaled_test_df)


def ft_imp(model):
    cols=X_train.columns
    feat_imp=model.feature_importances_
    imp=(feat_imp/feat_imp.sum())*100
    imp_df=pd.DataFrame({"columns":cols,"feat_imp":imp})

    return imp_df

imp_df=ft_imp(final_lgbm)
plt.figure(figsize=(12,6))
plt.title("Feature importance")
sns.barplot(data=imp_df.sort_values(by='feat_imp',ascending=False),y='columns',x='feat_imp',palette='viridis')
plt.show()


def submission(test,predictions):
    sub=pd.DataFrame({"id":test['id'],'Listening_Time_minutes':predictions})
    sub.to_csv("submission.csv",index=False)
    return sub
    
sub_df=submission(test_df,test_preds)


sub_df.head()

