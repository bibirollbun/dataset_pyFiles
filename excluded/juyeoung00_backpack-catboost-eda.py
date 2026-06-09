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


train_df= pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df= pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train_df.head()


train_df.info()


nn= pd.get_dummies(train_df, columns=["Waterproof","Laptop Compartment"],drop_first=True)
nn.info()


nb= nn[["Compartments","Weight Capacity (kg)","Waterproof_Yes","Laptop Compartment_Yes","Price"]]
nb.corr()


fig, axes= plt.subplots(1,2, figsize=(10,5))

sns.histplot(train_df["Price"],ax=axes[0], bins=30, kde=True)
sns.histplot(train_df["Weight Capacity (kg)"],ax=axes[1], bins=30, kde=True)

plt.show()


import scipy.stats as stats


no_naData= train_df.dropna()

brand_groups = [no_naData[no_naData['Brand'] == brand]['Price'] for brand in no_naData['Brand'].unique()]
size_groups = [no_naData[no_naData['Size'] == brand]['Price'] for brand in no_naData['Size'].unique()]
Color_groups = [no_naData[no_naData['Color'] == brand]['Price'] for brand in no_naData['Color'].unique()]
Style_groups = [no_naData[no_naData['Style'] == brand]['Price'] for brand in no_naData['Style'].unique()]
Material_groups = [no_naData[no_naData['Material'] == brand]['Price'] for brand in no_naData['Material'].unique()]

# ANOVA 분석 수행
f_stat, p_value = stats.f_oneway(*brand_groups)

# 결과 출력
print(f"Brand F-statistic: {f_stat}")
print(f"P-value: {p_value}\n")


# ANOVA 분석 수행
f_stat, p_value = stats.f_oneway(*size_groups)

# 결과 출력
print(f"Size F-statistic: {f_stat}")
print(f"P-value: {p_value}\n")


# ANOVA 분석 수행
f_stat, p_value = stats.f_oneway(*Color_groups)

# 결과 출력
print(f"Color F-statistic: {f_stat}")
print(f"P-value: {p_value}\n")


# ANOVA 분석 수행
f_stat, p_value = stats.f_oneway(*Style_groups)

# 결과 출력
print(f"Style F-statistic: {f_stat}")
print(f"P-value: {p_value}\n")


# ANOVA 분석 수행
f_stat, p_value = stats.f_oneway(*Material_groups)

# 결과 출력
print(f"Material F-statistic: {f_stat}")
print(f"P-value: {p_value}")


train_data= train_df[["Brand","Material","Color","Weight Capacity (kg)","Price"]]
train_data.head()


train_data.describe()


# 함수로 만들어서 처리하기
def data_conversion(df):
    df= pd.get_dummies(df, columns=["Waterproof","Laptop Compartment"],drop_first=True)
    
    df= df.fillna(df[["Weight Capacity (kg)"]].mean())
    
    mode_values= df[["Brand","Material","Color"]].mode().iloc[0]
    df= df.fillna(mode_values)

    df= df.drop(["Size","id","Compartments","Style"],axis=1)
    
    return df


data = train_df


n_train= data_conversion(data)
n_train.info()


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


X= n_train.drop('Price',axis=1)
y= n_train['Price']

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state= 100)

model= CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=3, cat_features=["Brand", "Color", "Material"], verbose=200)

model.fit(X_train, y_train)


y_pred= model.predict(X_test)

mae= mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


rmse= mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')


import optuna
from sklearn.model_selection import cross_val_score


# 데이터 셋팅 
train_df= pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df= pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

data = train_df

n_train= data_conversion(data)

X= n_train.drop('Price',axis=1)
y= n_train['Price'] 



X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state= 100)


def objective(trial):
    params={
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg',1,10),
        'random_strength': trial.suggest_int('random_strength',1,5),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1,1.0)
    }
    model= CatBoostRegressor(cat_features=["Brand", "Color", "Material"], verbose=100, **params)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=100)
    
    model.fit(X_train,y_train,
             eval_set=(X_valid, y_valid),
             early_stopping_rounds=20,
             verbose=500)
    
    y_pred= model.predict(X_valid)
    score= mean_squared_error(y_valid, y_pred, squared=False)
    return score




study= optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)


# 처음 조정 : 동일 값 10일때 컷, y값 조정 안함 
print("Best RMSE:", study.best_value)
print("Best Parameters:", study.best_params)


# y값을 로그로 변환 한 후, 동일한 값이 20일때, 컷
print("Best RMSE:", study.best_value)
print("Best Parameters:", study.best_params)


# y값을 원래대로 바꿈 : 20일때 컷 
print("Best RMSE:", study.best_value)
print("Best Parameters:", study.best_params)


# y값을 원래대로 바꿈 : 10일때 컷 : 값 설정 잘못했었음... 
print("Best RMSE:", study.best_value)
print("Best Parameters:", study.best_params)


# y값을 원래대로 바꿈 : 20일때 컷 
print("Best RMSE:", study.best_value)
print("Best Parameters:", study.best_params)


best_params = study.best_params  # Optuna에서 최적화한 값 사용


model= CatBoostRegressor(**best_params, cat_features=["Brand", "Color", "Material"],verbose=500)
model.fit(X_train, y_train)

predictions= model.predict(X_test)

rmse= mean_squared_error(y_test, predictions, squared=False)
print(rmse)


t = test_df

n_test= data_conversion(t)


n_test.info()


predictions= model.predict(n_test)


result_df = pd.DataFrame({'id': test_df["id"], 'Price': predictions})
result_df.to_csv('submission.csv', index=False)

