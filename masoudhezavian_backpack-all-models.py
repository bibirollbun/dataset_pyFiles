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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDRegressor, LinearRegression
import xgboost as xgb
import catboost
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


# Define a function to find outliers based on IQR
numeric_columns = ['Compartments', 'Weight Capacity (kg)']
def find_outliers(df):
    outliers = {}
    imputed_df = df.copy()
    for col in df.columns:
        v = df[col]
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr  
        upper_bound = q3 + 1.5 * iqr  
        outliers_count = ((v < lower_bound) | (v > upper_bound)).sum()
        perc = outliers_count * 100.0 / len(df)
        outliers[col] = (perc, outliers_count)
        print(f"Column {col} outliers = {perc:.2f}% ({outliers_count} out of {len(df)})")

    return outliers

# Find outliers in the DataFrame
find_outliers(df_train[numeric_columns])
find_outliers(df_test[numeric_columns])


for numeric in numeric_columns : 
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_train[numeric])


    plt.title(f'Boxplot for {numeric} train dataframe')
    plt.xlabel('Values')
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_test[numeric])


    plt.title(f'Boxplot for {numeric} test dataframe')
    plt.xlabel('Values')


    plt.show()


#preprocessing : 
# 1- impute null values with decision tree classifire

def preprocessing_decision_tree_imputer(df_train, df_test):
    
    target = df_train['Price']
    df_train = df_train.drop(columns='Price')
    df_train= df_train.drop(['id'],axis=1)

    df_id = df_test['id']
    df_test= df_test.drop(['id'],axis=1)


    
    categorical_columns = df_train.select_dtypes(include=['object']).columns
    numeric_columns = ['Compartments', 'Weight Capacity (kg)']

    print('train shape : ', df_train.shape)
    print('test shape : ', df_test.shape)
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_columns),
        ('cat', LabelEncoder, categorical_columns)
    ])
    
    
    def impute_with_decision_tree(X_train, X_test, categorical_columns):
        
        for column in categorical_columns:
            train_data = X_train[X_train[column].notna()]
            X_train_non_null = train_data.drop(columns=categorical_columns)  
            y_train_non_null = train_data[column]
            

            test_data = X_train[X_train[column].isna()]
            X_test_for_imputation = test_data.drop(columns=categorical_columns)
            

            model = DecisionTreeClassifier()
            model.fit(X_train_non_null, y_train_non_null)
            
            
            predicted_values = model.predict(X_test_for_imputation)
            
            
            X_train.loc[X_train[column].isna(), column] = predicted_values
            
            
            X_test_for_imputation = X_test.drop(columns=categorical_columns)
            predicted_values_test = model.predict(X_test_for_imputation)
            X_test[column] = predicted_values_test

        return X_train, X_test
    
    
    df_train, df_test = impute_with_decision_tree(df_train, df_test, categorical_columns)

    print("Dtrain with predicted values:")
    print(df_train.shape)

    print("\nDtest with predicted values:")
    print(df_test.shape)
    print('------------------------------------')
    print("Dtrain nulls:")
    print(df_train.isnull().sum())

    print("\nDtest nulls:")
    print(df_test.isnull().sum())
    
    median_weight_test = df_test['Weight Capacity (kg)'].median()
    df_test['Weight Capacity (kg)'] = df_test['Weight Capacity (kg)'].fillna(median_weight_test)

    median_weight_train = df_train['Weight Capacity (kg)'].median()
    df_train['Weight Capacity (kg)'] = df_train['Weight Capacity (kg)'].fillna(median_weight_train)

    print('train nulls : ',df_train.isnull().sum())
    print('--------------------------------------------')
    print('test nulls : ',df_test.isnull().sum())
    
    encoder = LabelEncoder()
    object_columns_train = df_train.select_dtypes(include=['object']).columns
    for col in object_columns_train:

        encoder.fit(df_train[col])
        

        df_train[col] = encoder.transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
    
        
    X_train, X_temp, y_train, y_temp = train_test_split(df_train, target, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print('X_train shape : ', X_train.shape)
    print('y_train shape : ', y_train.shape)
    print('X_val shape : ', X_val.shape)
    print('y_val shape : ', y_val.shape)
    print('X_test shape : ', X_test.shape)
    print('y_test shape : ', y_test.shape)
    print('df_id shape : ', df_id.shape)
    print('df_test shape : ', df_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test, df_id, df_test


X_train_dt, X_val_dt, X_test_dt, y_train_dt, y_val_dt, y_test_dt, df_id_dt, df_test_dt= preprocessing_decision_tree_imputer(df_train, df_test)


#preprocessing : 
# 1- impute null values with Mode

def preprocessing_raplacing_mode(df_train, df_test):
    object_columns_train = df_train.select_dtypes(include=['object']).columns
    object_columns_test = df_test.select_dtypes(include=['object']).columns

    
    for col in object_columns_train:
        mode_value = df_train[col].mode()[0]  
        df_train[col].fillna(mode_value, inplace=True)
        
    for col in object_columns_test:
        mode_value = df_test[col].mode()[0]  
        df_test[col].fillna(mode_value, inplace=True)
        
        
        
    median_weight_test = df_test['Weight Capacity (kg)'].median()
    df_test['Weight Capacity (kg)'] = df_test['Weight Capacity (kg)'].fillna(median_weight_test)

    median_weight_train = df_train['Weight Capacity (kg)'].median()
    df_train['Weight Capacity (kg)'] = df_train['Weight Capacity (kg)'].fillna(median_weight_train)

    print('train nulls : ',df_train.isnull().sum())
    print('--------------------------------------------')
    print('test nulls : ',df_test.isnull().sum()) 
    print('--------------------------------------------')
    
    
    encoder = LabelEncoder()
    object_columns_train = df_train.select_dtypes(include=['object']).columns
    for col in object_columns_train:
        encoder.fit(df_train[col])
        df_train[col] = encoder.transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
        
    
    df_target = df_train['Price']
    df_train= df_train.drop(['Price'],axis=1)
    df_train= df_train.drop(['id'],axis=1)
    df_id = df_test['id']
    df_test= df_test.drop(['id'],axis=1)
    print('train shape : ', df_train.shape)
    print('test shape : ', df_test.shape)
    print('----------------------------------------')
    
    X_train, X_temp, y_train, y_temp = train_test_split(df_train, df_target, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print('X_train shape : ', X_train.shape)
    print('y_train shape : ', y_train.shape)
    print('----------------------------------------')
    print('X_val shape : ', X_val.shape)
    print('y_val shape : ', y_val.shape)
    print('----------------------------------------')
    print('X_test shape : ', X_test.shape)
    print('y_test shape : ', y_test.shape)
    print('----------------------------------------')
    print('df_id shape : ', df_id.shape)
    print('df_test shape : ', df_test.shape)
    print('----------------------------------------')
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df_id, df_test
    


X_train_mode, X_val_mode, X_test_mode, y_train_mode, y_val_mode, y_test_mode, df_id_mode, df_test_mode= preprocessing_raplacing_mode(df_train, df_test)


#preprocessing : 
# 1- impute null values with a word

def preprocessing_raplacing_word(df_train, df_test):
    replace_value = 'value'
    object_columns = df_train.select_dtypes(include=['object']).columns
    df_train[object_columns] = df_train[object_columns].fillna(replace_value)
    df_test[object_columns] = df_train[object_columns].fillna(replace_value)


    median_weight_test = df_test['Weight Capacity (kg)'].median()
    df_test['Weight Capacity (kg)'] = df_test['Weight Capacity (kg)'].fillna(median_weight_test)

    median_weight_train = df_train['Weight Capacity (kg)'].median()
    df_train['Weight Capacity (kg)'] = df_train['Weight Capacity (kg)'].fillna(median_weight_train)




    print('train nulls : ',df_train.isnull().sum())
    print('--------------------------------------------')
    print('test nulls : ',df_test.isnull().sum()) 
    print('----------------------------------------')
    
    
    encoder = LabelEncoder()
    for col in object_columns:
        encoder.fit(df_train[col])
        df_train[col] = encoder.transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
        
        
    df_target = df_train['Price']
    df_train= df_train.drop(['Price'],axis=1)
    df_train= df_train.drop(['id'],axis=1)
    df_id = df_test['id']
    df_test= df_test.drop(['id'],axis=1)
    print('train shape : ', df_train.shape)
    print('test shape : ', df_test.shape)
    print('----------------------------------------')
    
    X_train, X_temp, y_train, y_temp = train_test_split(df_train, df_target, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print('X_train shape : ', X_train.shape)
    print('y_train shape : ', y_train.shape)
    print('----------------------------------------')
    print('X_val shape : ', X_val.shape)
    print('y_val shape : ', y_val.shape)
    print('----------------------------------------')
    print('X_test shape : ', X_test.shape)
    print('y_test shape : ', y_test.shape)
    print('----------------------------------------')
    print('df_id shape : ', df_id.shape)
    print('df_test shape : ', df_test.shape)
    print('----------------------------------------')
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df_id, df_test


X_train_word, X_val_word, X_test_word, y_train_word, y_val_word, y_test_word, df_id_word, df_test_word= preprocessing_raplacing_word(df_train, df_test)


#preprocessing : 
# 1- impute null values with a word 
# 2- without X_val and y_val

def preprocessing_raplacing_word_without_xval(df_train, df_test):
    replace_value = 'value'
    object_columns = df_train.select_dtypes(include=['object']).columns
    df_train[object_columns] = df_train[object_columns].fillna(replace_value)
    df_test[object_columns] = df_train[object_columns].fillna(replace_value)


    median_weight_test = df_test['Weight Capacity (kg)'].median()
    df_test['Weight Capacity (kg)'] = df_test['Weight Capacity (kg)'].fillna(median_weight_test)

    median_weight_train = df_train['Weight Capacity (kg)'].median()
    df_train['Weight Capacity (kg)'] = df_train['Weight Capacity (kg)'].fillna(median_weight_train)




    print('train nulls : ',df_train.isnull().sum())
    print('--------------------------------------------')
    print('test nulls : ',df_test.isnull().sum()) 
    print('----------------------------------------')
    
    
    encoder = LabelEncoder()
    for col in object_columns:
        encoder.fit(df_train[col])
        df_train[col] = encoder.transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
        
        
    df_target = df_train['Price']
    df_train= df_train.drop(['Price'],axis=1)
    df_train= df_train.drop(['id'],axis=1)
    df_id = df_test['id']
    df_test= df_test.drop(['id'],axis=1)
    print('train shape : ', df_train.shape)
    print('test shape : ', df_test.shape)
    print('----------------------------------------')
    
    X_train, X_test, y_train, y_test = train_test_split(df_train, df_target, test_size=0.2, random_state=42)
    
    print('X_train shape : ', X_train.shape)
    print('y_train shape : ', y_train.shape)
    print('----------------------------------------')
    print('X_test shape : ', X_test.shape)
    print('y_test shape : ', y_test.shape)
    print('----------------------------------------')
    print('df_id shape : ', df_id.shape)
    print('df_test shape : ', df_test.shape)
    print('----------------------------------------')
    
    return X_train, X_test, y_train, y_test, df_id, df_test


X_train_word2, X_test_word2, y_train_word2, y_test_word2, df_id_word2, df_test_word2= preprocessing_raplacing_word_without_xval(df_train, df_test)


def Catboost(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word):
    results = []
    
    model = catboost.CatBoostRegressor(task_type="GPU",
                                    loss_function="RMSE", 
                                    devices='0',
                                    verbose=100,
                                    iterations= 1000,
                                    learning_rate=0.01,
                                    l2_leaf_reg= 10
                                    )
    
    model_dt = model.fit(X_train_dt, y_train_dt, eval_set=(X_val_dt, y_val_dt))
    y_train_pred_dt = model.predict(X_train_dt)
    y_test_pred_dt = model.predict(X_test_dt)
    rmse_train_dt = np.sqrt(mean_squared_error(y_train_dt, y_train_pred_dt))
    rmse_test_dt = np.sqrt(mean_squared_error(y_test_dt, y_test_pred_dt))
    
    results.append({'Dataset': f'Dataset_dt_catboost', 'RMSE Train': rmse_train_dt, 'RMSE Test': rmse_test_dt})
    
    model_mode = model.fit(X_train_mode, y_train_mode, eval_set=(X_val_mode, y_val_mode))
    y_train_pred_mode = model.predict(X_train_mode)
    y_test_pred_mode = model.predict(X_test_mode)
    rmse_train_mode = np.sqrt(mean_squared_error(y_train_mode, y_train_pred_mode))
    rmse_test_mode = np.sqrt(mean_squared_error(y_test_mode, y_test_pred_mode))
    results.append({'Dataset': f'Dataset_mode_catboost', 'RMSE Train': rmse_train_mode, 'RMSE Test': rmse_test_mode})
    
    model_word = model.fit(X_train_word, y_train_word, eval_set=(X_val_word, y_val_word))
    y_train_pred_word = model.predict(X_train_word)
    y_test_pred_word = model.predict(X_test_word)
    rmse_train_word = np.sqrt(mean_squared_error(y_train_word, y_train_pred_word))
    rmse_test_word = np.sqrt(mean_squared_error(y_test_word, y_test_pred_word))
    results.append({'Dataset': f'Dataset_word_catboost', 'RMSE Train': rmse_train_word, 'RMSE Test': rmse_test_word})
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df, model_dt,model_mode,model_word


results_cat_df, model_cat_dt,model_cat_mode,model_cat_word = Catboost(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word)


def XGboost(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word):
    results = []
    
    xgb_model = xgb.XGBRegressor(
    tree_method="gpu_hist",  
    predictor="gpu_predictor",
    n_estimators=400, 
    max_depth=2, 
    learning_rate=0.0155, 
    objective="reg:squarederror"
                )
    
    model_dt = xgb_model.fit(X_train_dt, y_train_dt, eval_set=[(X_val_dt, y_val_dt)],verbose=100)
    y_train_pred_dt = xgb_model.predict(X_train_dt)
    y_test_pred_dt = xgb_model.predict(X_test_dt)
    rmse_train_dt = np.sqrt(mean_squared_error(y_train_dt, y_train_pred_dt))
    rmse_test_dt = np.sqrt(mean_squared_error(y_test_dt, y_test_pred_dt))
    
    results.append({'Dataset': f'Dataset_dt_XGboost', 'RMSE Train': rmse_train_dt, 'RMSE Test': rmse_test_dt})
    
    model_mode = xgb_model.fit(X_train_mode, y_train_mode, eval_set=[(X_val_mode, y_val_mode)],verbose=100)
    y_train_pred_mode = xgb_model.predict(X_train_mode)
    y_test_pred_mode = xgb_model.predict(X_test_mode)
    rmse_train_mode = np.sqrt(mean_squared_error(y_train_mode, y_train_pred_mode))
    rmse_test_mode = np.sqrt(mean_squared_error(y_test_mode, y_test_pred_mode))
    results.append({'Dataset': f'Dataset_mode_XGboost', 'RMSE Train': rmse_train_mode, 'RMSE Test': rmse_test_mode})
    
    model_word = xgb_model.fit(X_train_word, y_train_word, eval_set=[(X_val_word, y_val_word)],verbose=100)
    y_train_pred_word = xgb_model.predict(X_train_word)
    y_test_pred_word = xgb_model.predict(X_test_word)
    rmse_train_word = np.sqrt(mean_squared_error(y_train_word, y_train_pred_word))
    rmse_test_word = np.sqrt(mean_squared_error(y_test_word, y_test_pred_word))
    results.append({'Dataset': f'Dataset_word_XGboost', 'RMSE Train': rmse_train_word, 'RMSE Test': rmse_test_word})
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df, model_dt,model_mode,model_word


results_xgb_df, model_xgb_dt,model_xgb_mode,model_xgb_word = XGboost(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word)


def Lightgbm(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word):
    results = []
    
    lgb_model = lgb.LGBMRegressor(
    boosting_type="gbdt",
    objective="RMSE",
    n_estimators=500,
    learning_rate=0.01,
    max_depth=35,
    device="gpu"  
)
    
    model_dt = lgb_model.fit(X_train_dt, y_train_dt, eval_set=[(X_val_dt, y_val_dt)])
    y_train_pred_dt = lgb_model.predict(X_train_dt)
    y_test_pred_dt = lgb_model.predict(X_test_dt)
    rmse_train_dt = np.sqrt(mean_squared_error(y_train_dt, y_train_pred_dt))
    rmse_test_dt = np.sqrt(mean_squared_error(y_test_dt, y_test_pred_dt))
    
    results.append({'Dataset': f'Dataset_dt_Lightgbm', 'RMSE Train': rmse_train_dt, 'RMSE Test': rmse_test_dt})
    
    model_mode = lgb_model.fit(X_train_mode, y_train_mode, eval_set=[(X_val_mode, y_val_mode)])
    y_train_pred_mode = lgb_model.predict(X_train_mode)
    y_test_pred_mode = lgb_model.predict(X_test_mode)
    rmse_train_mode = np.sqrt(mean_squared_error(y_train_mode, y_train_pred_mode))
    rmse_test_mode = np.sqrt(mean_squared_error(y_test_mode, y_test_pred_mode))
    results.append({'Dataset': f'Dataset_mode_Lightgbm', 'RMSE Train': rmse_train_mode, 'RMSE Test': rmse_test_mode})
    
    model_word = lgb_model.fit(X_train_word, y_train_word, eval_set=[(X_val_word, y_val_word)])
    y_train_pred_word = lgb_model.predict(X_train_word)
    y_test_pred_word = lgb_model.predict(X_test_word)
    rmse_train_word = np.sqrt(mean_squared_error(y_train_word, y_train_pred_word))
    rmse_test_word = np.sqrt(mean_squared_error(y_test_word, y_test_pred_word))
    results.append({'Dataset': f'Dataset_word_Lightgbm', 'RMSE Train': rmse_train_word, 'RMSE Test': rmse_test_word})
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df, model_dt,model_mode,model_word


results_lgb_df, model_lgb_dt,model_lgb_mode,model_lgb_word = Lightgbm(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word)


def ann(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word):
    results = []
    
    model1 = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_dt.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  
    ])
    model1.compile(optimizer='adam', loss='mse')  
    
    model2 = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_mode.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  
    ])
    model2.compile(optimizer='adam', loss='mse')
    
    model3 = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_word.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  
    ])
    model3.compile(optimizer='adam', loss='mse')

    checkpoint_loss_dt = ModelCheckpoint(
        'best_loss_model_dt.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )


    checkpoint_loss_mode = ModelCheckpoint(
        'best_loss_model_mode.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    checkpoint_loss_word = ModelCheckpoint(
        'best_loss_model_word.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    reduce_lr1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1)
    
    
    model_dt= model1.fit(X_train_dt, y_train_dt, epochs=200, batch_size=512,
            validation_data=(X_val_dt, y_val_dt),
            callbacks=[checkpoint_loss_dt,reduce_lr1])
    
    y_test_pred_dt = model1.predict(X_test_dt)
    y_train_pred_dt = model1.predict(X_train_dt)
    rmse_train_dt = np.sqrt(mean_squared_error(y_train_dt, y_train_pred_dt))
    rmse_test_dt = np.sqrt(mean_squared_error(y_test_dt, y_test_pred_dt))
    results.append({'Dataset': f'Dataset_dt_ann', 'RMSE Train': rmse_train_dt, 'RMSE Test': rmse_test_dt})
    print('-----------------------------------------------------------------------------')
    
    model_mode= model2.fit(X_train_mode, y_train_mode, epochs=200, batch_size=256,
            validation_data=(X_val_mode, y_val_mode),
            callbacks=[checkpoint_loss_mode,reduce_lr1])
    
    
    y_train_pred_mode = model2.predict(X_train_mode)
    y_test_pred_mode = model2.predict(X_test_mode)
    rmse_train_mode = np.sqrt(mean_squared_error(y_train_mode, y_train_pred_mode))
    rmse_test_mode = np.sqrt(mean_squared_error(y_test_mode, y_test_pred_mode))
    results.append({'Dataset': f'Dataset_mode_ann', 'RMSE Train': rmse_train_mode, 'RMSE Test': rmse_test_mode})
    print('-----------------------------------------------------------------------------')
    
    model_word= model3.fit(X_train_word, y_train_word, epochs=200, batch_size=256,
            validation_data=(X_val_word, y_val_word),
            callbacks=[checkpoint_loss_word,reduce_lr1])
    
    
    y_train_pred_word = model3.predict(X_train_word)
    y_test_pred_word = model3.predict(X_test_word)
    rmse_train_word = np.sqrt(mean_squared_error(y_train_word, y_train_pred_word))
    rmse_test_word = np.sqrt(mean_squared_error(y_test_word, y_test_pred_word))
    results.append({'Dataset': f'Dataset_word_ann', 'RMSE Train': rmse_train_word, 'RMSE Test': rmse_test_word})
    print('-----------------------------------------------------------------------------')
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df, model_dt,model_mode,model_word


results_ann_df, model_ann_dt,model_ann_mode,model_ann_word = ann(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word)

results_ann_df


def cnn (X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word):
    results = []
    
    scaler = StandardScaler()
    X_train_dt = scaler.fit_transform(X_train_dt)
    X_test_dt = scaler.transform(X_test_dt)
    X_val_dt = scaler.transform(X_val_dt)
    
    X_train_mode = scaler.fit_transform(X_train_mode)
    X_test_mode = scaler.transform(X_test_mode)
    X_val_mode = scaler.transform(X_val_mode)
    
    X_train_word = scaler.fit_transform(X_train_word)
    X_test_word = scaler.transform(X_test_word)
    X_val_word = scaler.transform(X_val_word)
    
    X_train_dt = X_train_dt.reshape(X_train_dt.shape[0], X_train_dt.shape[1], 1)  # تغییر ابعاد برای 3D
    X_test_dt = X_test_dt.reshape(X_test_dt.shape[0], X_test_dt.shape[1], 1)
    X_val_dt = X_val_dt.reshape(X_val_dt.shape[0], X_val_dt.shape[1], 1)
    
    X_train_mode = X_train_mode.reshape(X_train_mode.shape[0], X_train_mode.shape[1], 1)  # تغییر ابعاد برای 3D
    X_test_mode = X_test_mode.reshape(X_test_mode.shape[0], X_test_mode.shape[1], 1)
    X_val_mode = X_val_mode.reshape(X_val_mode.shape[0], X_val_mode.shape[1], 1)
    
    X_train_word = X_train_word.reshape(X_train_word.shape[0], X_train_word.shape[1], 1)  # تغییر ابعاد برای 3D
    X_test_word = X_test_word.reshape(X_test_word.shape[0], X_test_word.shape[1], 1)
    X_val_word = X_val_word.reshape(X_val_word.shape[0], X_val_word.shape[1], 1)
    
    model1 = keras.Sequential([
    
    layers.Conv1D(32, 1, activation='relu', input_shape=(X_train_dt.shape[1], 1)), 
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),  

    layers.Conv1D(16, 1, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),
    layers.Conv1D(8, 1, activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu'),
    

    layers.Dense(1)
    ])
    model1.compile(optimizer='adam', loss='mse')
    
    model2 = keras.Sequential([
    
    layers.Conv1D(32, 1, activation='relu', input_shape=(X_train_dt.shape[1], 1)), 
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),  

    layers.Conv1D(16, 1, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),
    layers.Conv1D(8, 1, activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu'),
    

    layers.Dense(1)
    ])
    model2.compile(optimizer='adam', loss='mse')
    
    model3 = keras.Sequential([
    
    layers.Conv1D(32, 1, activation='relu', input_shape=(X_train_dt.shape[1], 1)), 
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),  

    layers.Conv1D(16, 1, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),
    layers.Conv1D(8, 1, activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu'),
    

    layers.Dense(1)
    ])
    model3.compile(optimizer='adam', loss='mse')
    
    checkpoint_loss_dt = ModelCheckpoint(
        'best_loss_modelcnn_dt.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )


    checkpoint_loss_mode = ModelCheckpoint(
        'best_loss_modelcnn_mode.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    checkpoint_loss_word = ModelCheckpoint(
        'best_loss_modelcnn_word.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    reduce_lr1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1)
    
    
    model_dt= model1.fit(X_train_dt, y_train_dt, epochs=200, batch_size=512,
            validation_data=(X_val_dt, y_val_dt),
            callbacks=[checkpoint_loss_dt,reduce_lr1])
    
    y_test_pred_dt = model1.predict(X_test_dt)
    y_train_pred_dt = model1.predict(X_train_dt)
    rmse_train_dt = np.sqrt(mean_squared_error(y_train_dt, y_train_pred_dt))
    rmse_test_dt = np.sqrt(mean_squared_error(y_test_dt, y_test_pred_dt))
    results.append({'Dataset': f'Dataset_dt_cnn', 'RMSE Train': rmse_train_dt, 'RMSE Test': rmse_test_dt})
    print('-----------------------------------------------------------------------------')
    
    model_mode= model2.fit(X_train_mode, y_train_mode, epochs=200, batch_size=256,
            validation_data=(X_val_mode, y_val_mode),
            callbacks=[checkpoint_loss_mode,reduce_lr1])
    
    
    y_train_pred_mode = model2.predict(X_train_mode)
    y_test_pred_mode = model2.predict(X_test_mode)
    rmse_train_mode = np.sqrt(mean_squared_error(y_train_mode, y_train_pred_mode))
    rmse_test_mode = np.sqrt(mean_squared_error(y_test_mode, y_test_pred_mode))
    results.append({'Dataset': f'Dataset_mode_cnn', 'RMSE Train': rmse_train_mode, 'RMSE Test': rmse_test_mode})
    print('-----------------------------------------------------------------------------')
    
    model_word= model3.fit(X_train_word, y_train_word, epochs=200, batch_size=256,
            validation_data=(X_val_word, y_val_word),
            callbacks=[checkpoint_loss_word,reduce_lr1])
    
    
    y_train_pred_word = model3.predict(X_train_word)
    y_test_pred_word = model3.predict(X_test_word)
    rmse_train_word = np.sqrt(mean_squared_error(y_train_word, y_train_pred_word))
    rmse_test_word = np.sqrt(mean_squared_error(y_test_word, y_test_pred_word))
    results.append({'Dataset': f'Dataset_word_cnn', 'RMSE Train': rmse_train_word, 'RMSE Test': rmse_test_word})
    print('-----------------------------------------------------------------------------')
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df, model_dt,model_mode,model_word
    


results_cnn_df, model_cnn_dt,model_cnn_mode,model_cnn_word = cnn(X_train_dt,X_test_dt,X_val_dt,y_train_dt,y_test_dt,y_val_dt,
                X_train_mode,X_test_mode,X_val_mode,y_train_mode,y_test_mode,y_val_mode,
                X_train_word,X_test_word,X_val_word,y_train_word,y_test_word,y_val_word)

results_cnn_df


def sgd(X_train_word2,X_test_word2,y_train_word2,y_test_word2):
    results=[]
    
    scaler = StandardScaler()
    X_train_word2 = scaler.fit_transform(X_train_word2)
    X_test_word2 = scaler.transform(X_test_word2)

    
    model = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01, random_state=42)
    model.fit(X_train_word2, y_train_word2)
    
    y_train_pred = model.predict(X_train_word2)
    y_test_pred = model.predict(X_test_word2)
    rmse_train = np.sqrt(mean_squared_error(y_train_word2, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test_word2, y_test_pred))
    results.append({'Dataset': f'Dataset_sgd', 'RMSE Train': rmse_train, 'RMSE Test': rmse_test})
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df, model


results_df_sgd, model_sgd = sgd(X_train_word2,X_test_word2,y_train_word2,y_test_word2)

results_df_sgd


def linearregressor(X_train_word2,X_test_word2,y_train_word2,y_test_word2):
    results=[]
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_word2, y_train_word2)
    
    y_train_pred = model.predict(X_train_word2)
    y_test_pred = model.predict(X_test_word2)
    rmse_train = np.sqrt(mean_squared_error(y_train_word2, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test_word2, y_test_pred))
    results.append({'Dataset': f'Dataset_lr', 'RMSE Train': rmse_train, 'RMSE Test': rmse_test})
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df, model


results_df_lr, model_lr = linearregressor(X_train_word2,X_test_word2,y_train_word2,y_test_word2)

results_df_lr


total_result = pd.concat([results_cat_df,results_xgb_df,results_lgb_df,
                        results_df_lr,results_df_sgd,results_ann_df,results_cnn_df])

total_result


total_result[total_result['RMSE Test'] == total_result['RMSE Test'].min()]

