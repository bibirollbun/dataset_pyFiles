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


!pip install -q scikit-learn==1.5.2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import category_encoders as ce
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import cross_val_score
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.linear_model import ElasticNet

import time


import warnings 
warnings.filterwarnings("ignore") 

%matplotlib inline


data_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


data_train.head(2)


# Cheking for data types
data_train.info()


# Checking for NaN data

data_train.isna().sum()


# Checking for values statistic info for numeric and object types
print('Statistic data for numeric values:')
display(data_train.describe(exclude='object').T)

print('\nStatistic data for Object values:')
display(data_train.describe(exclude=np.number).T)


numeric_feats_list = list(data_train.select_dtypes([np.number]).columns)
numeric_feats_list.remove('id')

# Subplot for distribution grpaphs of numeric data
fig, axes = plt.subplots(ncols=5, nrows=1, figsize = (25, 5))

# Var for column number calculations
col = 0

# Loop for graph building 
for feat in numeric_feats_list:
    
    # Condition if-else as for one feat need counplot and for other need hisplot
    if feat != 'Number_of_Ads':
        
        # Histplot building for feats
        hist = sns.histplot(
            data_train[feat],
            ax=axes[col]
        )
        
        hist.set_ylabel('')
        hist.set_xlabel(feat, fontsize = 14)
        hist.tick_params(rotation = 45)
    
    else:
        
        # Countplot building for feat
        countpl = sns.countplot(
            data_train,
            x = feat,
            ax=axes[col]
        )
        
        countpl.set_ylabel('')
        countpl.set_xlabel(feat, fontsize = 14)
        countpl.tick_params(rotation = 45)
        
    # Var plussing for graph index switching 
    col += 1
    
fig.suptitle('Distribution of Feature Values', fontsize = 20)

# outliers in data


# Subplot for distribution grpaphs of numeric data
fig, axes = plt.subplots(ncols=5, nrows=1, figsize = (25, 7))

# Var for column number calculations
col = 0

# Loop for graph building 
for feat in numeric_feats_list:
    box = sns.boxplot(
        data_train, 
        y = feat, 
        orient='v',
        ax=axes[col]
    )
    
    box.set_title(feat, fontsize = 14)
    box.set_ylabel('')
    
    col += 1
    
fig.suptitle('Boxplots for numeric features', fontsize = 20)


episode_lenght_out = data_train[data_train['Episode_Length_minutes'] > 125].value_counts().sum()
number_of_ads_out = data_train[data_train['Number_of_Ads'] > 3].value_counts().sum()

print('Numbers of outliers in episode_lenght_out feature is:', episode_lenght_out)
print('Numbers of outliers in Number_of_Ads feature is:', number_of_ads_out)



host_popularity_out = data_train[data_train['Host_Popularity_percentage'] > 100].value_counts().sum()
guest_popularity_out = data_train[data_train['Guest_Popularity_percentage'] > 100].value_counts().sum()

print('Numbers of probably oulliers in Host_Popularity_percentage feature is:', host_popularity_out)
print('Numbers of probably oulliers in Guest_Popularity_percentage feature is:', guest_popularity_out)


# Indexes lists to drop
episode_lenght_out_index = list(data_train[data_train['Episode_Length_minutes'] > 125].index)
number_of_ads_out_index = list(data_train[data_train['Number_of_Ads'] > 3].index)
host_popularity_out_index = list(data_train[data_train['Host_Popularity_percentage'] > 100].index)
guest_popularity_out_index = list(data_train[data_train['Guest_Popularity_percentage'] > 100].index)

index_to_drop = set(episode_lenght_out_index + 
                    number_of_ads_out_index + 
                    host_popularity_out_index + 
                    guest_popularity_out_index
                    )

index_to_drop = list(index_to_drop)


df_cleaned = data_train.copy()

df_cleaned = df_cleaned.drop(index=index_to_drop, axis=0)
df_cleaned = df_cleaned.drop('id', axis=1)
df_cleaned = df_cleaned.reset_index(drop=True)


df_cleaned.shape


print(f'Data Frame contains {df_cleaned.duplicated().sum()} duplicates')


display(data_test.describe(exclude='object').T)
display(data_test.describe(exclude=np.number).T)


data_test['Episode_Length_minutes'] = \
    data_test['Episode_Length_minutes'].apply(
        lambda x: data_test['Episode_Length_minutes'].median() if x > 125 else x
        )
    
data_test['Number_of_Ads'] = \
    data_test['Number_of_Ads'].apply(
        lambda x: data_test['Number_of_Ads'].mode() if x > 3 else x
    )
    
data_test['Host_Popularity_percentage'] = \
    data_test['Host_Popularity_percentage'].apply(
        lambda x: 100 if x > 100 else x
    ) 
    
data_test['Guest_Popularity_percentage'] = \
    data_test['Guest_Popularity_percentage'].apply(
        lambda x: 100 if x > 100 else x
    ) 


data_test = data_test.drop('id', axis=1)


print(df_cleaned.shape[1])
print(data_test.shape[1])


# Target data
y = df_cleaned['Listening_Time_minutes']

# predictors data
df_cleaned = df_cleaned.drop('Listening_Time_minutes', axis=1)


object_columns = list(df_cleaned.select_dtypes('object').columns)

one_hot_list = list()
binary_list = list()

for feat in object_columns:
    unique_values = df_cleaned[feat].nunique()
    print(f'Unique values for {feat} is {unique_values}')
    
    if unique_values > 15:
        binary_list.append(feat)
    
    else:
        one_hot_list.append(feat)
    


one_hot_encoder = OneHotEncoder()

data_onehot = one_hot_encoder.fit_transform(df_cleaned[one_hot_list]).toarray()

one_hot_col_names = one_hot_encoder.get_feature_names_out(one_hot_list)
df_onehot = pd.DataFrame(data_onehot, columns=one_hot_col_names)
df_onehotted = pd.concat([df_cleaned, df_onehot], axis=1)
df_onehotted = df_onehotted.drop(one_hot_list, axis=1)


bin_encoder = ce.BinaryEncoder(cols=binary_list)

feat_binned = bin_encoder.fit_transform(df_onehotted[binary_list])


df_encoded = pd.concat([df_onehotted, feat_binned], axis=1)

df_encoded = df_encoded.drop(binary_list, axis=1)


data_onehot_test = one_hot_encoder.transform(data_test[one_hot_list]).toarray()

one_hot_col_names = one_hot_encoder.get_feature_names_out(one_hot_list)
df_onehot_test = pd.DataFrame(data_onehot_test, columns=one_hot_col_names)
df_onehotted_test = pd.concat([data_test, df_onehot_test], axis=1)
df_onehotted_test = df_onehotted_test.drop(one_hot_list, axis=1)


feat_binned_test = bin_encoder.transform(df_onehotted_test[binary_list])

df_encoded_test = pd.concat([df_onehotted_test, feat_binned_test], axis=1)

df_encoded_test = df_encoded_test.drop(binary_list, axis=1)


df_encoded.isna().sum()

for n in range(df_encoded.shape[1]):
    
    if df_encoded.isna().sum()[n] != 0:
        print('\nFeature {} has {} missings'.format(
            df_encoded.isna().sum().index[n], 
            df_encoded.isna().sum()[n]
        ))


df_encoded_test.isna().sum()

for n in range(df_encoded_test.shape[1]):
    
    if df_encoded_test.isna().sum()[n] != 0:
        print('\nFeature {} has {} missings'.format(
            df_encoded_test.isna().sum().index[n], 
            df_encoded_test.isna().sum()[n]
        ))


df_encoded['Number_of_Ads'] = df_encoded['Number_of_Ads'].fillna(
    df_encoded['Number_of_Ads'].mode()[0]
)


# Make a copy - it will be missing-free and will be 
# used as train data for value predictions
data = df_encoded.copy()

# This DF has all missed data and from this will be generated missed values
test_data = data[data['Episode_Length_minutes'].isnull()]

# Clean it from missed data
data.dropna(inplace=True)

# Target data for training
y_train = data['Episode_Length_minutes']

# Predictrors for train and test data
X_train = data.drop(['Episode_Length_minutes', 'Guest_Popularity_percentage'], axis=1)
X_test = test_data.drop(['Episode_Length_minutes', 'Guest_Popularity_percentage'], axis=1)

# Making scaller
r_scaller_filling_ep_leng = preprocessing.RobustScaler()

X_train_scaled = r_scaller_filling_ep_leng.fit_transform(X_train)
X_test_scaled = r_scaller_filling_ep_leng.transform(X_test)

# Hyperparameters for Hyperopt which will be tested for choosing best ones
space = {
    'alpha': hp.loguniform('alpha', -3, 3),  # 0.05 до 20
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'max_iter': hp.quniform('max_iter', 500, 2000, 100),
    'fit_intercept': hp.choice('fit_intercept', [True, False])
}
   

random_state = 42
def hyperopt_rf(space, cv=5, X=X_train_scaled, y=y_train, random_state=random_state):
    """ Funstion for testing defferent parameters for the Model
    Input: Hyperrapameters for test, X - data, y - data
    Output: Metric score
    """
    params = {'alpha': float(space['alpha']),
        'l1_ratio': float(space['l1_ratio']),
        'max_iter': int(space['max_iter']),
        'fit_intercept': space['fit_intercept']}

  
  
    # Using random parameters to model
    model = ElasticNet(**params, random_state=random_state)

    # Model learning
    model.fit(X, y)
    score = cross_val_score(
        model, X, y, 
        cv=cv, 
        scoring='neg_mean_squared_error'
    ).mean()

    return -score

%time

# For the results logging
trials = Trials() 

best=fmin(hyperopt_rf,  
          space=space, 
          algo=tpe.suggest, 
          max_evals=20, 
          trials=trials,
          rstate=np.random.default_rng(random_state) 
         )
print("Best hyperparameters {}".format(best))


# making model with the best paramaters
model_ep_leng = ElasticNet(
    random_state=random_state, 
    alpha = float(best['alpha']),
    l1_ratio = float(best['l1_ratio']),
    max_iter = int(best['max_iter']),
    fit_intercept = bool(best['fit_intercept'])
    )

model_ep_leng.fit(X_train_scaled, y_train)
y_train_pred = model_ep_leng.predict(X_train_scaled)
print('MSE on test data: {:.2f}'.format(metrics.mean_squared_error(y_train, y_train_pred)))

# Predictions (previosly missed data)
y_test_pred = model_ep_leng.predict(X_test_scaled)

for i, ni in enumerate(test_data.index):
    df_encoded.loc[ni, 'Episode_Length_minutes'] = y_test_pred[i]


# For the comments here please look upper - all the same 
# Only exception is here dropping only one feature as other has been filled before

data = df_encoded.copy()

test_data = data[data['Guest_Popularity_percentage'].isnull()]

data.dropna(inplace=True)

y_train = data['Guest_Popularity_percentage']

X_train = data.drop(['Guest_Popularity_percentage'], axis=1)
X_test = test_data.drop(['Guest_Popularity_percentage'], axis=1)

r_scaller_filling_guest_pop = preprocessing.RobustScaler()

X_train_scaled = r_scaller_filling_guest_pop.fit_transform(X_train)
X_test_scaled = r_scaller_filling_guest_pop.transform(X_test)


space = {
    'alpha': hp.loguniform('alpha', -3, 3),  # 0.05 до 20
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'max_iter': hp.quniform('max_iter', 500, 2000, 100),
    'fit_intercept': hp.choice('fit_intercept', [True, False])
    }

random_state = 42

def hyperopt_rf(space, cv=5, X=X_train_scaled, y=y_train, random_state=random_state):
    
    params = {'alpha': float(space['alpha']),
        'l1_ratio': float(space['l1_ratio']),
        'max_iter': int(space['max_iter']),
        'fit_intercept': space['fit_intercept']}

    model = ElasticNet(**params, random_state=random_state)

    model.fit(X, y)
    score = cross_val_score(
        model, X, y, 
        cv=cv, 
        scoring='neg_mean_squared_error'  # для регрессии
    ).mean()
    
    return -score

%time

trials = Trials() 

best=fmin(hyperopt_rf, 
          space=space,
          algo=tpe.suggest,
          max_evals=20, 
          trials=trials,
          rstate=np.random.default_rng(random_state) 
         )
print("Best hyperparameters {}".format(best))

model_guest_pop = ElasticNet(
    random_state=random_state, 
    alpha = float(best['alpha']),
    l1_ratio = float(best['l1_ratio']),
    max_iter = int(best['max_iter']),
    fit_intercept = bool(best['fit_intercept'])
    )

model_guest_pop.fit(X_train_scaled, y_train)
y_train_pred = model_guest_pop.predict(X_train_scaled)
print('MSE on test data: {:.2f}'.format(metrics.mean_squared_error(y_train, y_train_pred)))

y_test_pred = model_guest_pop.predict(X_test_scaled)


for i, ni in enumerate(test_data.index):
    df_encoded.loc[ni, 'Guest_Popularity_percentage'] = y_test_pred[i]



# For the code comments please check upper

data = df_encoded_test.copy()

test_data = data[data['Episode_Length_minutes'].isnull()]

data.dropna(inplace=True)

y_train = data['Episode_Length_minutes']

X_train = data.drop(['Episode_Length_minutes', 'Guest_Popularity_percentage'], axis=1)
X_test = test_data.drop(['Episode_Length_minutes', 'Guest_Popularity_percentage'], axis=1)

X_train_scaled = r_scaller_filling_ep_leng.transform(X_train)
X_test_scaled = r_scaller_filling_ep_leng.transform(X_test)

# Here we using the model trained on the Train Data
y_pred = model_ep_leng.predict(X_test_scaled)

for i, ni in enumerate(test_data.index):
    df_encoded_test.loc[ni, 'Episode_Length_minutes'] = y_pred[i]


data = df_encoded_test.copy()

test_data = data[data['Guest_Popularity_percentage'].isnull()]

data.dropna(inplace=True)

y_train = data['Guest_Popularity_percentage']

X_train = data.drop(['Guest_Popularity_percentage'], axis=1)
X_test = test_data.drop(['Guest_Popularity_percentage'], axis=1)

X_train_scaled = r_scaller_filling_guest_pop.transform(X_train)
X_test_scaled = r_scaller_filling_guest_pop.transform(X_test)

# Here we using the model trained on the Train Data
y_pred = model_guest_pop.predict(X_test_scaled)

for i, ni in enumerate(test_data.index):
    df_encoded_test.loc[ni, 'Guest_Popularity_percentage'] = y_pred[i]


# Copy of cleaned data to following actions
X = df_encoded.copy()


# Splitting of Train data to train and test datasets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)



# creating lists with features name according to dtype
float_feats = list(X_train.select_dtypes('float64').columns)
int_feats = list(X_train.select_dtypes('int64').columns)


# Reducing numbers width 
for feat in float_feats:
    X_train[feat] = X_train[feat].astype('float16')
    X_test[feat] = X_test[feat].astype('float16')
    
for feat in int_feats:
    X_train[feat] = X_train[feat].astype('int16')
    X_test[feat] = X_test[feat].astype('int16')


# Creating polynomializer
poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)

# Study it
poly.fit(X_train)

# Creating of new feats
X_train_poly = poly.transform(X_train)
X_test_poly = poly.transform(X_test)


# Reducing numbers width 
for feat in float_feats:
    df_encoded_test[feat] = df_encoded_test[feat].astype('float16')
    
for feat in int_feats:
    df_encoded_test[feat] = df_encoded_test[feat].astype('int16')


df_encoded_test_poly = poly.transform(df_encoded_test)


space = {
    'n_estimators': hp.quniform('n_estimators', 1700, 1900, 10),  # Tree quantity
    'learning_rate': hp.loguniform('learning_rate', np.log(0.015), np.log(0.025)),  
    'max_depth': hp.quniform('max_depth', 11, 17, 1),  
    'subsample': hp.uniform('subsample', 0.85, 0.95),  # rate of quantity of examples for each tree
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 0.8),  # rate of quantity of feature for each tree
    'reg_lambda': hp.uniform('reg_lambda', 7, 9),  # L2-reg
    'reg_alpha': hp.uniform('reg_alpha', 0, 1)  # L1-reg
}

# Funtion for the model scoring
def hyperopt_xgb(space, cv=3, X=X_train_poly, y=y_train, random_state=random_state):
    print("Starting new iteration...")
    params = {
        'n_estimators': int(space['n_estimators']), 
        'learning_rate': float(space['learning_rate']), 
        'max_depth': int(space['max_depth']), 
        'subsample': float(space['subsample']),
        'colsample_bytree': float(space['colsample_bytree']),
        'reg_lambda': float(space['reg_lambda']),
        'reg_alpha': float(space['reg_alpha']),
    }

    # Initialization of model
    model = XGBRegressor(**params, tree_method='hist', booster='gbtree', device='cuda')

    # Model scorring with cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    
    # Returning of RMSE score for optimization
    return -scores.mean()



# Starting optimization
trials = Trials()

best = fmin(
    fn=hyperopt_xgb,
    space=space,
    algo=tpe.suggest,
    max_evals=2, 
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("Best parameters:")
print(best)

# Final model learning based on best hyperparameters
model_xgb = XGBRegressor(
    n_estimators=int(best['n_estimators']),
    learning_rate=float(best['learning_rate']),
    max_depth=int(best['max_depth']),
    subsample=float(best['subsample']),
    colsample_bytree=float(best['colsample_bytree']),
    reg_lambda=float(best['reg_lambda']),
    reg_alpha=float(best['reg_alpha']),
    booster='gbtree',
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    device='cuda'
)

model_xgb.fit(X_train_poly, y_train)
y_test_pred = model_xgb.predict(X_test_poly)

print('RMSE of test train data: {:.4f}'.format(metrics.root_mean_squared_error(y_test, y_test_pred)))


y_pred_test_data = model_xgb.predict(df_encoded_test_poly)


# downloading of sample data, reaching with predictions, saving
sample_submission = pd.read_csv(
    '/kaggle/input/playground-series-s5e4/sample_submission.csv'
    )

sample_submission_poly_df_xgb = pd.DataFrame(
    y_pred_test_data, columns=['y_pred']
    )

sample_submission_poly_xgb = pd.concat(
    [sample_submission, sample_submission_poly_df_xgb], 
    axis=1
    )

sample_submission_poly_xgb = sample_submission_poly_xgb.drop(
    'Listening_Time_minutes', axis=1
    )

sample_submission_poly_xgb = sample_submission_poly_xgb.rename(
    columns={'y_pred': 'Listening_Time_minutes'}
    )

sample_submission_poly_xgb = sample_submission_poly_xgb.set_index('id')

sample_submission_poly_xgb.to_csv('/kaggle/working/submission_v6.csv')


# model saving
model_xgb.save_model('/kaggle/working/model_xgb_poly_hyperopt.model')

