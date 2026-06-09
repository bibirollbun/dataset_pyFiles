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


X_full = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv',index_col='id')
X_test_full = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv',index_col = 'id')


X_full.head()


X_full = X_full.drop(['Podcast_Name', 'Episode_Title'],axis=1)
X_test = X_test_full.drop(['Podcast_Name', 'Episode_Title'],axis=1)


X_full.shape, X_test.shape


target = list(set(X_full.columns)- set(X_test.columns))[0]
target


X_full.info()


X_full.head()


X_full.isna().sum()


y = X_full[target]
X= X_full.drop(target,axis=1)


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train,y_valid = train_test_split(X,y, test_size=0.2, random_state=1)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


X_train.isna().sum()


num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = [col for col in X.columns if X[col].dtype=='object' ]
num_cols


abs(X[num_cols].skew())


skewness = abs(X[num_cols].skew())
less_skewed_cols = [col for col in skewness.index if skewness[col]<0.5]
more_skewed_cols = [col for col in skewness.index if skewness[col]>0.5]


# imputing null values
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(strategy='mean')
X_train[less_skewed_cols] =  mean_imputer.fit_transform(X_train[less_skewed_cols])
X_valid[less_skewed_cols] =  mean_imputer.transform(X_valid[less_skewed_cols])

median_imputer = SimpleImputer(strategy='median')
X_train[more_skewed_cols] =  median_imputer.fit_transform(X_train[more_skewed_cols])
X_valid[more_skewed_cols] =  median_imputer.transform(X_valid[more_skewed_cols])

cat_imputer = SimpleImputer(strategy='constant')
X_train[cat_cols] =  cat_imputer.fit_transform(X_train[cat_cols])
X_valid[cat_cols] =  cat_imputer.transform(X_valid[cat_cols])


X_train[cat_cols].nunique()


low_cardinality = [ col for col in cat_cols if X_train[col].nunique()<=10]
low_cardinality


oe_encode_cols = [cname for cname in low_cardinality if cname!='Genre']
oh_encode_cols = ['Genre']


# scaling and encoding
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

scaler = StandardScaler()
X_train[num_cols] =  scaler.fit_transform(X_train[num_cols])
X_valid[num_cols] =  scaler.transform(X_valid[num_cols])

ordinal_encoder = OrdinalEncoder()
X_train[oe_encode_cols] =  ordinal_encoder.fit_transform(X_train[oe_encode_cols])
X_valid[oe_encode_cols] =  ordinal_encoder.transform(X_valid[oe_encode_cols])


oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_train = pd.DataFrame(oh_encoder.fit_transform(X_train[oh_encode_cols]), columns= oh_encoder.get_feature_names_out(),index= X_train.index)
encoded_valid = pd.DataFrame(oh_encoder.transform(X_valid[oh_encode_cols]), columns= oh_encoder.get_feature_names_out(),index= X_valid.index)

X_train = pd.concat([X_train, encoded_train], axis=1)
X_valid = pd.concat([X_valid, encoded_valid], axis=1)

X_train.drop(oh_encode_cols,axis=1,inplace=True)
X_valid.drop(oh_encode_cols,axis=1,inplace=True)


X_train.head()


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)

model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

valid_preds = model.predict(X_valid) 

print('mean absolute error:', mean_absolute_error(y_valid,valid_preds))

rmse = np.sqrt(mean_squared_error(y_valid,valid_preds))
print('root mean sqaured error: ',rmse )


def rmse(true, preds):
    return np.sqrt(sum((true-preds)**2)/len(true))
    
rmse(y_valid,valid_preds)


# impute null values
X_test[less_skewed_cols] =  mean_imputer.transform(X_test[less_skewed_cols])
X_test[more_skewed_cols] =  median_imputer.transform(X_test[more_skewed_cols])
X_test[cat_cols] =  cat_imputer.transform(X_test[cat_cols])

# scaling and encoding
X_test[num_cols] =  scaler.transform(X_test[num_cols])
X_test[oe_encode_cols] =  ordinal_encoder.transform(X_test[oe_encode_cols])

encoded_test = pd.DataFrame(oh_encoder.transform(X_test[oh_encode_cols]), columns= oh_encoder.get_feature_names_out(),index= X_test.index)
X_test = pd.concat([X_test, encoded_test], axis=1)
X_test.drop(oh_encode_cols,axis=1,inplace=True)


y_preds = model.predict(X_test)


# Save test predictions to file
output = pd.DataFrame({'id': X_test.index,
                       'Listening_Time_minutes': y_preds})
output.to_csv('xgb_tuned_submission.csv', index=False)
print('output file created Sucessfully!')
output.head()


# importing Regression models
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


model_1 = RandomForestRegressor(n_estimators = 200,n_jobs=-1, random_state=1)
model_2 = LGBMRegressor(n_jobs=-1, random_state=1)
model_3 = XGBRegressor(n_estimators= 100, max_depth= 6, n_jobs=-1, random_state=1)

model_4 = RandomForestRegressor(n_estimators = 200,
                                max_depth=15,
                                min_samples_split=5,
                                # min_samples_leaf=2, 
                                n_jobs=-1, random_state=1)


# model_7 = RandomForestRegressor(n_estimators = 200,
#                                 max_depth=15,
#                                 min_samples_split=5,
#                                 # min_samples_leaf=2, 
#                                 n_jobs=-1, random_state=1)


def get_scores(model,X_train,X_valid,y_train,y_valid):
    print(model)
    model.fit(X_train,y_train)
    print('model trained!\n')
    
    valid_preds = model.predict(X_valid)
    
    rmse = np.sqrt(mean_squared_error(y_valid,valid_preds))
    print(f'root mean sqaured error: {rmse}' )
    
    print('mean absolute error:', mean_absolute_error(y_valid,valid_preds))


models = [model_2, model_3, model_1, model_4]

print('-------------------------------')
    get_scores(model,X_train,X_valid,y_train,y_valid)


model_5 = XGBRegressor(n_estimators= 200,n_jobs=-1, random_state=1)
model_6 = RandomForestRegressor(n_estimators = 300,n_jobs=-1, random_state=1)

models = [model_5,model_6]

for model in models:
    print('-------------------------------')
    get_scores(model,X_train,X_valid,y_train,y_valid)


better_models ={
    'RandomForest': model_6,
    'XGBREgressor' : model_5,
    'LGBMRegressor' :model_2,
}

for model_name in better_models:
    model = better_models[model_name]
    y_pred = model.predict(X_test)

    
    output =pd.DataFrame({'id': X_test.index,
                       'Listening_Time_minutes': y_pred})

    print('\n predictions of of X_test: \n',output.head())
    
    output.to_csv(f'{model_name}_ps5e4_submission.csv',index= False)
    print(f'\nSubmission file using {model_name} Regressor model created successfully!')
    print('-----------------------------------\n')




