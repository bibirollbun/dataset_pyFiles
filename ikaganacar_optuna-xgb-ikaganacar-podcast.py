import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tqdm import tqdm
from itertools import combinations

#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from warnings import filterwarnings
filterwarnings('ignore')

import gc
gc.collect()



are_you_on_kaggle = True

# Load data from kaggle
if are_you_on_kaggle: 
    train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
    test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
    subm  = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
else: # load data from local
    train = pd.read_csv('train.csv',index_col='id')
    test  = pd.read_csv('test.csv', index_col='id')
    subm  = pd.read_csv('sample_submission.csv')

train


def feature_eng(df,train=True):
    le = LabelEncoder() 

    # New Columns
    df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '', regex=False).astype('category')
    
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'].replace([np.inf, -np.inf], 0, inplace=True)

    df['Weekend'] = df['Publication_Day'].isin(['Saturday','Sunday']).astype('category') 

    df['Lenght_Type'] = pd.cut(df['Episode_Length_minutes'], bins=[0, 30, 60, 90, 200],labels=['short', 'medium', 'long', 'very_long'])

    sentiments = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df['Sentiment_Score'] = df['Episode_Sentiment'].map(sentiments)

    df['Genre_Sentiment'] = df['Genre'].astype(str) + "_" + df['Episode_Sentiment'].astype(str)

    
    if train:
        # Get rid of outliers
        df = df[df['Number_of_Ads']<10]

    
    # Fill NULL values with median
    df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(), inplace=True) 
    df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)
    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)

    
    # Preprocess Categorical Columns
    categorical_cols = ['Podcast_Name','Genre','Publication_Day','Publication_Time','Episode_Sentiment','Lenght_Type','Weekend','Genre_Sentiment']
    for c in categorical_cols:
        #df[c]=le.fit_transform(df[c]) # Converts categorical column into int format
        df[c] = df[c].astype('category') # Define column type as category 

    return df

# Apply
train = feature_eng(train)
test = feature_eng(test,train=False)



def optimize_data(df):
    # Reduce Data sizes
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes.astype('int16')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def create_submission(final_model):
    test_processed = optimize_data(test)
    xgb_test = xgb.DMatrix(test_processed, enable_categorical=True)
    test_results = final_model.predict(xgb_test)

    submission = pd.DataFrame({
        "id": subm["id"],
        "Listening_Time_minutes": test_results
    })
    submission.to_csv("submission.csv", index=False)



def optuna_search(X, y, trial_count=20):       
    def objective(trial):
        params = {
            'device': 'cuda',
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'max_bin': trial.suggest_int('max_bin', 512, 4096),
            'max_depth': trial.suggest_int('max_depth', 7, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-5, 1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 2.0, log=True),
        }
   
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

        results = xgb.cv(
            params,
            dtrain,
            num_boost_round=10000,
            nfold=3,
            early_stopping_rounds=50,
            metrics={'rmse'},
            seed=42,
            as_pandas=True,
            verbose_eval=False
        )
        return results['test-rmse-mean'].min()

    
    study = optuna.create_study(direction='minimize')
    
    study.optimize(objective, n_trials=trial_count, show_progress_bar=True, gc_after_trial=True)
    
    return study.best_params 


%%time

y = train["Listening_Time_minutes"]
X = optimize_data(train.drop("Listening_Time_minutes", axis=1))

#optuna_search(X,y,30)


%%time

y = train["Listening_Time_minutes"]
X = optimize_data(train.drop("Listening_Time_minutes", axis=1))

result_params = {
                 'max_bin': 3643,
                 'max_depth': 13,
                 'learning_rate': 0.005147579446392961,
                 'subsample': 0.891461905992914,
                 'colsample_bytree': 0.5117775558597328,
                 'gamma': 0.6835247087833203,
                 'min_child_weight': 2,
                 'reg_alpha': 0.010299782108777746,
                 'reg_lambda': 0.022954826795186848,
                 'objective': 'reg:squarederror',
                 'device': 'cuda',
                 'grow_policy': 'depthwise',
                 'sampling_method': 'gradient_based',
                }

dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

model = xgb.train(
            result_params,
            dtrain,
            num_boost_round=13000,
        )

create_submission(model)

