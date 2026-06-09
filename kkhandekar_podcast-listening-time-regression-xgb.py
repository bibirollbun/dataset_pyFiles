# Upgrade SKLearn
!pip install -qq scikit-learn==1.6.1
!pip install --upgrade xgboost -qq


#
# Libraries
#

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json
from itertools import *
from datetime import date
from tqdm import tqdm
import xgboost as xgb

#Optuna
import optuna

# Sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.impute import *
import sklearn

# Stats
import scipy
from scipy.stats import *
from scipy.sparse import csr_matrix

# Setting
pd.set_option('max_colwidth',None)
seed = 2304
warnings.simplefilter('ignore')

data_path = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('csv'):
            data_path.append(os.path.join(dirname, filename))


#
# Custom Function -- Imputation
#

def impute(df):
    """
    Impute numerical columns in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with imputed numerical columns.
    """
    
    impute_cols = []
    
    # find & extract columns with NaN
    res = df.isna().sum()

    for i, j in zip(res,list(df.columns)):
        if i > 0:  # no. of rows with NaN > 0
            impute_cols.append(j)
        else:
            pass

    print(f"Found \" {', '.join(impute_cols)} \" columns in \"{df.name}\" dataset \n")
    
    # loop through each column & replace NaN with 75% quantile
    for c in impute_cols:
        qnt = df[c].quantile(0.75)   # 75% quantile
        df[[c]] = df[[c]].fillna(value=qnt,axis=1)
    
    print(f"** Imputation Completed **\n")


#
# Custom Function -- Extract "New" Features
#

def new_features(df):
    """
    Extract new features

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new feature columns
    """
    
    # when the condition is true
    choicelist = ['yes']
    
    # Weekday / Weekend
    df['Is_Weekend'] = np.select([df['Publication_Day'].isin(['Saturday', 'Sunday'])], choicelist,default='no')
    
    # Host's popularity > 75%
    df['Is_High_Host_Popularity'] = np.select([df['Host_Popularity_percentage'] > 75.00], choicelist, default='no') 
    
    # Guest's popularity > 75%
    df['Is_High_Guest_Popularity'] = np.select([df['Guest_Popularity_percentage'] > 75.00], choicelist, default='no') 
    
    # Listening Experience
    df['Good_Listening_Exp'] = np.select([df['Number_of_Ads'] == 0], choicelist, default='no')
    
    # Ad Density
    df['Ad_Density'] = np.where (df['Episode_Length_minutes'] != 0, round((df['Number_of_Ads'] / df['Episode_Length_minutes']),2), 0)
    
    # Is Genre Popular? 
    df['Popular_Genre'] = np.select([df['Genre'].isin(['Comedy','True Crime','News'])], choicelist,default='no')
    
    # Genre = True Crime & Publication_Time = Night
    df['TrueCrime_at_Night'] = np.select([( (df['Genre'] =='True Crime') & (df['Publication_Time']=='Night') )], choicelist,default='no')
    
    # Episode Length
    df['Is_Long_Episode'] = np.select([df['Episode_Length_minutes'] > 60.00], choicelist, default='no')
    
    # Is host popular than guest?
    df['Is_Host_Popular_than_Guest'] = np.select([df['Host_Popularity_percentage'] > df['Guest_Popularity_percentage']], choicelist, default='no')

    return df

    
#
# Custom Function - Scaling
#

def scale_num_cols(df, scaler='standard',columns=None):
    """
    Scales only the numerical columns in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        scaler (str, optional): The type of scaling to apply. 
            Either 'minmax' for Min-Max scaling or 'standard' for standardization.
            Defaults to 'standard'.
        cols: list of columns to be scaled

    Returns:
        pd.DataFrame: The DataFrame with numerical columns scaled.
    """
    
    df_scaled = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_scaled[col]):
            if scaler == 'minmax':
                s = MinMaxScaler()
            elif scaler == 'standard':
                 s = StandardScaler()
            else:
                raise ValueError("Scaler must be 'minmax' or 'standard'")
            df_scaled[col] = s.fit_transform(df_scaled[[col]])
    return df_scaled


#
# Custom Function - Calculate RMSE
#

def calc_rmse(y_true,y_pred):
    """
    Calculate RMSE

    Args:
        y_true: actual value of target
        y_pred: predicted value of target

    Returns:
        rmse: root mean squared error value
    """
    rmse = root_mean_squared_error(y_true,y_pred)
    return '{:.5}'.format(rmse)
    


#
#  Load Data
#

sample = pd.read_csv(data_path[0],index_col='id')
train = pd.read_csv(data_path[1],index_col='id')
test = pd.read_csv(data_path[2],index_col='id')

# drop columns
unwanted = ['Episode_Title']
train_ds = train.drop(unwanted,axis=1)
train_ds.name = "train"

test_ds = test.drop(unwanted,axis=1)
test_ds.name = 'test'

# View
train_ds.head()


#
# Impute
#

# Imputing 
impute(train_ds)
impute(test_ds)

# View
train_ds.head()


#
# Extracting "New" Features
#

# train
train_ds = new_features(train_ds)
test_ds = new_features(test_ds)


# View
train_ds.head()


#
# Filtering Columns, Encoding, Feature Engineering
#

# Filter
cat_cols = train_ds.select_dtypes(include=['object']).columns.tolist()
num_cols = [ c for c in train_ds.columns if ((c not in cat_cols) and (c != 'Listening_Time_minutes')) ]

# Encoding
le = LabelEncoder()

for c in cat_cols:
    train_ds[c] = le.fit_transform(train_ds[c])
    test_ds[c] = le.fit_transform(test_ds[c])

# Feature Engineering
x = train_ds.loc[:, train_ds.columns != 'Listening_Time_minutes']
y = train_ds[['Listening_Time_minutes']]


#
# Data Split, Regression Matrix
#

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)

# Scaling
x_train_sc = scale_num_cols(x_train, scaler='standard',columns=num_cols)
x_test_sc = scale_num_cols(x_test, scaler='standard',columns=num_cols)


# Regression Matrix
dtrain_reg = xgb.DMatrix(x_train_sc, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(x_test_sc, y_test, enable_categorical=True)

# Model
params = {'verbosity':0}
reg = xgb.XGBRegressor(**params)


#
# Train, Predict & Evaluate
#

# Train
reg.fit(x_train_sc, y_train)

# Prediction
pred = reg.predict(x_test_sc)

# Evaluate
rmse = calc_rmse(y_test,pred)
print(f"The RMSE achieved: {rmse}")


# Get the Index
id_idx = list(test_ds.index)

# Scaling & Prediction
test_ds_sc = scale_num_cols(test_ds, scaler='standard',columns=num_cols)
pred_tst = reg.predict(test_ds_sc)
pred_tst = [round(p,3) for p in pred_tst]

# Submission 
submission_ds = pd.DataFrame({'id': id_idx,
                           'Listening_Time_minutes': pred_tst})

# export
submission_ds.to_csv('submission.csv',index=False)

