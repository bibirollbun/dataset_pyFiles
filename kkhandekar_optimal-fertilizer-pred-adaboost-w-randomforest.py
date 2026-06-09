# Upgrade sklearn
!pip install --upgrade --quiet scikit-learn


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
from tqdm.keras import TqdmCallback
from tqdm import tqdm

# Sklearn
import sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.ensemble import *

# Optuna
import optuna

# Setting
pd.set_option('max_colwidth',None)
seed = 945
warnings.simplefilter('ignore')

data_path = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('csv'):
            data_path.append(os.path.join(dirname, filename))


#
# Data
#

# files
train = pd.read_csv(data_path[1],index_col='id')
test = pd.read_csv(data_path[2],index_col='id')
sub = pd.read_csv(data_path[0])

# stats
print(f"Train shape: {train.shape} | Test shape: {test.shape}\n")

# column formating - opt.
for df in [train, test]:
    df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    df.columns = ['_'.join(c.split(' ')).lower() for c in df.columns]

# target-encoder (for submission!)
trgt_en = LabelEncoder()
trgt_en.fit(train['fertilizer_name'])

# view
train.head()


#
# Custom Function - Extract New Features
#
def new_features(df):
    '''
        args: dataframe
        return: dataframe with additional feature columns
    '''
    eps = 1e-6
    df['n_p_ratio'] = df['nitrogen'] / (df['phosphorous'] + eps)
    df['p_k_ratio'] = df['phosphorous'] / (df['potassium'] + eps)
    df['n_k_ratio'] = df['nitrogen'] / (df['potassium'] + eps)
    df['total_nutrients'] = df['nitrogen'] + df['phosphorous'] + df['potassium']
    df['temp_humidity_index'] = df['temperature'] * df['humidity']
    df['soil_quality_index'] = df['moisture'] / (df['temperature'] + eps)

    return df


#
# Custom Function - Encoding
#
def encoding(df, feat= False):
    '''
        args#1: dataframe
        args#2: feature flag
        return: dataframe with encoded feature columns
    '''
    
    le = LabelEncoder()

    # only features
    if feat == True:
        cat_cols = [c for c in df.columns if ((df[c].dtype == 'object') and (c != 'fertilizer_name'))]

        for c in cat_cols:
            le.fit(df[c])
            df[c] = le.transform(df[c])
        
        return df

    # only target
    else:
        le.fit(df['fertilizer_name'])
        df['fertilizer_name'] = le.transform(df['fertilizer_name'])
        decoded_labels = le.inverse_transform(list(df['fertilizer_name']))
        return df, decoded_labels


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
# Custom Function - Calculate AP
#

def calc_ap(y_true,y_score):
    """
    Calculate AP (average precision)

    Args:
        y_true: actual value of target
        y_score: confidence score of predicted target

    Returns:
        ap: average precision value
    """
    
    # get the top3
    y_score_top3 = np.argsort(y_score, axis=1)[:, -3:][:, ::-1]
    
    # Calculate AP (AP score)
    ap_score = average_precision_score(y_true,y_score_top3)
    
    return '{:.5}'.format(ap_score)


#
# Pre-Processing - new features
#

# extract additional features
train_df = new_features(train)
test_df= new_features(test)

# view
train_df.head()


#
# Pre-Processing - encoding (features)
#

# encode
train_df = encoding(train_df,feat=True)
test_df = encoding(test_df,feat=True)

# view
train_df.head()


#
# Pre-Processing - encoding (target)
#

# encode target
train_df, decoded_labels = encoding(train_df)

# view
train_df.head()


#
# Pre-Processing - encoding (frequency)
#

# frequency columns
freq_cols = ['soil_type','crop_type']

# encoding
for c in freq_cols:
    combined = pd.concat([train_df[c], test_df[c]])
    freq = combined.value_counts(normalize=True)
    train_df[f'{c}_freq'] = train_df[c].map(freq)
    test_df[f'{c}_freq'] = test_df[c].map(freq)

# view
train_df.head()


#
# Feature Engineering
#

# feature & target
x = train_df.loc[:, train_df.columns != 'fertilizer_name']
y = train_df[['fertilizer_name']]

# same column order
test_df = test_df[x.columns]

# train & validation split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

# view
print(f"training size: {x_train.shape} | validation size: {x_val.shape}")


#
# Scaling
#

x_train_sc = scale_num_cols(x_train, scaler='standard',columns=x_train.columns)
x_val_sc = scale_num_cols(x_val, scaler='standard',columns=x_val.columns)

# view
x_train_sc.head()


#
# Training
#

# base estimator
base_est = RandomForestClassifier(random_state=seed)

# model config
params = {
            'estimator': base_est,
            'random_state': seed
         }

clf = AdaBoostClassifier(**params)

# summary
clf


# fit
clf.fit(x_train_sc,y_train)

# prediction (proba)
pred_score = clf.predict_proba(x_val_sc)

# AP Score
ap_score = calc_ap(y_val, pred_score)
print(f"Base AP score: {ap_score}")


#
# Submission File
#

# prediction (test_df)
pred_test = clf.predict_proba(test_df)

# take top3
pred_test_top3 = np.argsort(pred_test, axis=1)[:, -3:][:, ::-1]

# get decoded list
decoded_labels = [trgt_en.inverse_transform(row) for row in pred_test_top3]

# submission df
submission = pd.DataFrame({
     'id': test.index.tolist(),
     'Fertilizer Name': [' '.join(pred_list) for pred_list in decoded_labels]
})

# view 
submission.head()

# export to csv
submission.to_csv("submission.csv", index=False)

