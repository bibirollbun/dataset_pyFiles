# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
original=pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


original


train_df


train_df.drop('id', axis=1, inplace=True)


train_data=pd.concat([train_df, original], ignore_index=True)

train_data


train_data.duplicated().value_counts()


train_data.drop_duplicates()


train_data.info()


train_data.isnull().sum().sort_values(ascending=True)


# Categorical Columns

cat_cols=[col for col in train_data.columns if train_data[col].dtype in ['object', 'bool']]

for col in cat_cols:
    plt.figure(figsize=(4,4))
    train_data[col].value_counts().plot(kind='bar')
    print(F"percentage {col} : {((train_data[col].value_counts()/len(train_data))*100).round(3)}")
    plt.title(f"Distibution {col} with frequency")
    plt.xlabel(f"{col}")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# Numerical Columns

num_col=[col for col in train_df.columns if train_data[col].dtype in ['float64', 'int64']]

for col in num_col:
    plt.figure(figsize=(4,4))
    train_data[col].value_counts().plot(kind='kde')
    print(f"Skewness: {train_data[col].skew()}")
    # print(f"Percentage {col}: {((train_df[col].value_counts()/len(train_df))*100).round(3)}")
    plt.xlabel(f"{col}")
    plt.ylabel("Frequency")
    plt.title(f"{col} distribution")
    plt.tight_layout()
    plt.show()


for col in num_col:
    print(f"{train_data[col].describe().round(4)}")


for col in num_col:
    if col=='accident_risk':
        continue
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=col, y='accident_risk', data=train_data, alpha=0.6)
    sns.regplot(x=col, y='accident_risk', data=train_data, scatter=False, color='red')
    plt.title(f'Accident Risk vs {col}')
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=col, y='accident_risk', data=train_data, palette='Pastel1')
    plt.title(f'Accident Risk vs {col}')
    plt.show()


# train["curvature_x_speed_limit"] = train["curvature"] * train["speed_limit"] 
# train["curvature**2"] = train["curvature"] ** 2
# train["curvature**3"] = train["curvature"] ** 3
# train["num_reported_accidents**2"] = train["num_reported_accidents"] ** 2
# train["num_reported_accidents**3"] = train["num_reported_accidents"] ** 3
# train["lighting_x_weather"] = train["lighting"].map({"night": 3, "dim": 1.05, "daylight": 1}) * train["weather"].map({"rainy": 2, "foggy": 2.5, "clear": 1}) # I gave more importance on fogginess (You can refer the plots)
# # I gave dim a slight more importance to test it out
# train["lighting_x_weather**2"] = train["lighting_x_weather"] ** 2
# train["lighting_x_weather**3"] = train["lighting_x_weather"] ** 3


from sklearn.preprocessing import LabelEncoder

def feature_engg(df):
    df['curv_speed']=df['curvature']*df['speed_limit']
    df["curvature**2"] = df["curvature"] ** 2
    df["curvature**3"] = df["curvature"] ** 3

    df["lighting_x_weather"] = df["lighting"].map({"night": 3, "dim": 1.05, "daylight": 1}) * df["weather"].map({"rainy": 2, "foggy": 2.5, "clear": 1})
    
    df['speed_curv_ratio'] = df['speed_limit'] / (df['curvature'] + 1e-6)

    df['speed_per_lane']=df['speed_limit']/(df['num_lanes']+1e-9)
    df['curb_per_lanes']=df['curvature']/(df['num_lanes']+1e-9)

    df['accident_density']=df['num_reported_accidents']/df['num_lanes']

    df["lighting_x_weather**2"] = df["lighting_x_weather"] ** 2
    df["lighting_x_weather**3"] = df["lighting_x_weather"] ** 3

    return df

train_data=feature_engg(train_data)
test_df=feature_engg(test_df)


train_data


from sklearn.preprocessing import StandardScaler, OrdinalEncoder

scaler=StandardScaler()
encoder=OrdinalEncoder()


from sklearn.model_selection import train_test_split

X=train_data.drop('accident_risk', axis=1)
y=train_data['accident_risk']

X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.2, random_state=42)

num_col = [col for col in num_col if col != 'accident_risk']

encoder.fit(X[cat_cols])
scaler.fit(X[num_col])

X_train[num_col]=scaler.transform(X_train[num_col])
X_val[num_col]=scaler.transform(X_val[num_col])
X_train[cat_cols]=encoder.transform(X_train[cat_cols])
X_val[cat_cols]=encoder.transform(X_val[cat_cols])
test_df[cat_cols]=encoder.transform(test_df[cat_cols])
test_df[num_col]=scaler.transform(test_df[num_col])


for col in cat_cols:
    X_train[col]=X_train[col].astype('category')
    X_val[col]=X_val[col].astype('category')
    test_df[col]=test_df[col].astype('category')


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE



    model = XGBRegressor(
        n_estimators=1800,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=4,
        reg_lambda=2,
        reg_alpha=1,
        booster='gbtree',
        enable_categorical=True,
        tree_method='gpu_hist',
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=True
    )
    
    
    y_pred=model.predict(X_val)
        
    rmse=np.sqrt(MSE(y_val, y_pred))
    print(f"RMSE  : {rmse}")


test_df


X_test = test_df.drop('id', axis=1)
y_test=model.predict(X_test)

submission=pd.DataFrame({
    'id':test_df['id'],
    'accident_risk':y_test
})

submission.to_csv("Submission.csv", index=False)

submission.head()


# !pip install optuna
# import optuna


# def objective(trial):
#     params={
#         'n_estimators': trial.suggest_categorical('n_estimators', [800, 900, 1000, 1100, 1200]),
#         'max_depth': trial.suggest_categorical('max_depth', [3, 4, 6, 7]),
#         'gamma': trial.suggest_categorical('gamma', [0, 1, 2, 3, 4, 5]),
#         'subsample': trial.suggest_categorical('subsample', [0.2, 0.5, 0.7, 0.8, 1.0]),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
#         'booster': 'dart',
#         'tree_method': 'gpu_hist',
#         'gpu_id': 0,
#         'verbosity': 3,
#         'enable_categorical': True,
#         'random_state': 42
#     }
    
#     model_opt=XGBRegressor(**params)
    
#     model_opt.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         early_stopping_rounds=50,
#         verbose=True
#     )
    
    
#     y_pred=model_opt.predict(X_val)
    
#     rmse=np.sqrt(MSE(y_val, y_pred))
#     return rmse

# # Create Optuna Study
# study=optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# print("Best RMSE: {:.4f}", format(study.best_value))
# print("Best params: ", study.best_params)




