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

import re

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import gc

import warnings
warnings.filterwarnings('ignore')


extended_data=pd.read_csv("/kaggle/input/extended-dataset-for-used-car-prices-regression/extended_data.csv")
train=pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv", index_col=0)
test=pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv", index_col=0)


print('Training set')
print(f'Number of rows: {train.shape[0]} \nNumber of cols: {train.shape[1]}')

print('')

print('Testing set')
print(f'Number of rows: {test.shape[0]} \nNumber of cols: {test.shape[1]}')


train.head()


train.info()


train.describe(include='all').T.round(2)


train.isna().sum()


train[train.duplicated()]


# Data Imputation
def rep_na_cat(df):
    df['fuel_type'].fillna('Electric', inplace=True)
    df['accident'].fillna('None reported', inplace=True)
    df['clean_title'].fillna('No', inplace=True)    


rep_na_cat(train)
rep_na_cat(test)


# Extract Horse Power from engine
def ext_hp(df):     
    df['engine_hp'] = [', '.join(map(str, re.findall(r'^[0-9]+.0HP', v))) for v in df['engine']]
    df['engine_hp'] = df['engine_hp'].str.replace('HP', '')
    df['engine_hp'][df['engine_hp'] == ''] = np.nan
    df['engine_hp'] = df['engine_hp'].astype(float)
    df['engine_hp'][df['engine_hp'].isna()] = df['engine_hp'].mean()


ext_hp(train)
ext_hp(test)


# Extract Engine displacement from engine
def ext_displa(df):
    df['engine_displacement'] = [', '.join(map(str, re.findall(r'[0-9].[0-9]+L', v))) for v in df['engine']]
    df['engine_displacement'] = df['engine_displacement'].str.replace('L', '')
    df['engine_displacement'][df['engine_displacement'] == ''] = np.nan
    df['engine_displacement'] = df['engine_displacement'].astype(float)
    df['engine_displacement'][df['engine_displacement'].isna()] = df['engine_displacement'].mean()
    df.drop(columns='engine', inplace=True)


ext_displa(train)
ext_displa(test)


# Source -> https://www.kaggle.com/code/riachoudhari/car-price-prediction-all-regression-models

# Combine transmission into 5 categories

def map_transmission(transmission):
    # Standardize the input
    transmission = transmission.strip().lower()
    
    if any(kw in transmission for kw in ['a/t', 'automatic']):
        return 'Automatic'
    elif any(kw in transmission for kw in ['m/t', 'manual']):
        return 'Manual'
    elif any(kw in transmission for kw in ['cvt', 'variator']):
        return 'Variator'
    elif any(kw in transmission for kw in ['tiptronic']):
        return 'Tiptronic'
    else:
        return 'Other'


train['transmission'] = train['transmission'].apply(map_transmission)
test['transmission'] = test['transmission'].apply(map_transmission)


miles_per_gallon = (extended_data[['model', 'model_year','miles_per_gallon','msrp']]
                         .groupby(['model','model_year'])
                         .mean()
                         .reset_index()
                   )


# Merge datasets
def merge_ext(df):
    df = df.merge(miles_per_gallon, left_on=['model','model_year'], right_on=['model','model_year'])    
    return df


test_extended = merge_ext(test)
train_extended = merge_ext(train)


# extract model age
def model_age(df):
    df['model_age'] = pd.to_datetime('today').year - df['model_year']  


model_age(train_extended)
model_age(test_extended)


# replace '-' values with Unknown
def rep_unknown(df):
    for col_name in df.columns:
        df[col_name] = df[col_name].apply(lambda rep: 'Unknown' if rep == 'â€“' else rep)    


rep_unknown(train_extended)
rep_unknown(test_extended)


# Data imputation mode for categorical variables and mean for continous variables
def rep_na(df):
    for col_name in df.columns:
        if df[col_name].isna().sum() > 0 and df[col_name].dtype == 'object':
            df[col_name].fillna(df[col_name].mode()[0], inplace=True)
        elif df[col_name].isna().sum() > 0 and df[col_name].dtype == 'float64':
            df[col_name].fillna(df[col_name].mean(), inplace=True)



rep_na(train_extended)
rep_na(test_extended)


# Extract variable types

continous_var = list(train_extended.select_dtypes(['int','float']).columns)[1:]
categorical_var = list(train_extended.select_dtypes('object').columns)


train_viz = train_extended.copy()


# identify outliers

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


subset_price = remove_outlier(train_viz, 'price')


subset_milage = remove_outlier(train_viz, 'milage')


# Group price into 3 bins - 'Low-Medium-High'
subset_price['price_txt'] = pd.DataFrame(subset_price['price']).apply(lambda x:pd.cut(x, bins = 3, labels=['Low','Medium','High']), axis = 0)


# Group milage into 3 bins - 'Low-Medium-High'
subset_milage['milage_txt'] = pd.DataFrame(subset_milage['milage']).apply(lambda x:pd.cut(x, bins = 3, labels=['Low','Medium','High']), axis = 0)


train_viz['price_txt'] = train_viz.merge(subset_price, how='left')['price_txt'].astype('object')


train_viz['milage_txt'] = train_viz.merge(subset_milage, how='left')['milage_txt'].astype('object')


train_viz['price_txt'].fillna('Extremely High (Outlier)', inplace=True)


train_viz['milage_txt'].fillna('Extremely High (Outlier)', inplace=True)


train_viz['price_txt_sorted'] = [1 if pt == 'Low' else 2 if pt == 'Medium' else 3 if pt == 'High' else 4 
                                for pt in train_viz['price_txt']]


#train_viz.to_csv('datasets/train_viz.csv')


palette = ['green', 'red', 'orange', 'yellow' ]

for num_var in continous_var:    

    fig, ax = plt.subplots(1, 2, figsize=(15, 3))   
    
    # Histograms    
    sns.histplot(data=train_viz, 
                      x=num_var,
                      bins = 30,
                      hue = 'price_txt',
                      palette=palette,
                      #kde=True,
                      alpha=0.3,
                      ax=ax[0]                          
                ) 
    # Mean vertical line
    ax[0].axvline(np.mean(train_viz[num_var]), color="red")   
    
    # Boxplots
    sns.boxplot(data=train_viz, 
                     y=num_var,
                    # hue = 'price_txt',
                     palette=palette,
                     #showfliers=False,
                     ax=ax[1]
                   );


sns.pairplot(train_extended[continous_var]);


# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(8, 6))
sns.heatmap(train_extended[continous_var]
            .corr(), annot=True, cmap="coolwarm")
plt.title('Heatmap of the dataset')
plt.show()


enc_vars = ['fuel_type','transmission','accident','clean_title']

train_extended_dummies = pd.get_dummies(train_extended[enc_vars], dtype=int, drop_first=True)
train_extended = train_extended.drop(columns=enc_vars)
train_extended = pd.concat([train_extended, train_extended_dummies], axis=1)

test_extended_dummies = pd.get_dummies(test_extended[enc_vars], dtype=int, drop_first=True)
test_extended = test_extended.drop(columns=enc_vars)
test_extended = pd.concat([test_extended, test_extended_dummies], axis=1)


from sklearn.preprocessing import LabelEncoder

train_enc = train_extended.copy()

test_enc = test_extended.copy()

# Encode the categorical variables
enc = LabelEncoder()

lab_vars = ['brand','model','ext_col', 'int_col']

for var in lab_vars:
    train_enc[var] = enc.fit_transform(train_enc[var])
    
for var in lab_vars:
    test_enc[var] = enc.fit_transform(test_enc[var])


# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(18, 10))
sns.heatmap(train_enc
            .corr(), annot=True, cmap="coolwarm")
plt.title('Heatmap of the dataset')
plt.show()


from sklearn.model_selection import train_test_split


X = train_enc.drop(columns='price')

y = train_enc['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


%%time
rf = RandomForestRegressor(random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'RandomForest Regressor RMSE: {rmse:.3f}')


rf_param_grid = {
            'bootstrap': True,
            'max_depth': 20,
            'max_features': 6,
            'min_samples_leaf': 5,
            'min_samples_split': 10,
            'n_estimators': 1000
}


rf_opt = (RandomForestRegressor(**rf_param_grid,
                                random_state=42)
          .fit(X_train, y_train)
         )


y_pred_rf_opt = rf_opt.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_opt))
print(f'RandomForest Regressor RMSE: {rmse:.3f}')


xgb = XGBRegressor(random_state=42)

xgb.fit(X_train, y_train)


y_pred_xgb = xgb.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'RandomForest Regressor RMSE: {rmse:.3f}')


xgb_best_params =  {'colsample_bytree': 0.8,
                     'learning_rate': 0.01,
                     'max_depth': 5,
                     'min_child_weight': 7,
                     'n_estimators': 1000,
                     'subsample': 0.8}


xgb_opt = (XGBRegressor(**xgb_best_params,
                        random_state=42)
               .fit(X_train, y_train)
          )


y_pred_xgb_opt = xgb_opt.predict(X_test)


# mse = mean_squared_error(y_eval, y_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_opt))
print(f'XGBRegressor RMSE: {rmse:.3f}')


xgb_opt = (XGBRegressor(**xgb_best_params)
               .fit(X, y)
          )


predictions = xgb_opt.predict(test_enc)


submission_df = pd.DataFrame({'id':test_enc.index,
                              'price':predictions})


submission_df.to_csv('submission.csv', index=False)
submission_df.head()

