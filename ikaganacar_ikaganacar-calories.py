import numpy as np
from sklearn.metrics import log_loss,accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd 
import xgboost as xgb
import json
from collections import Counter

import gc
from tqdm import tqdm
from itertools import combinations

from warnings import filterwarnings
filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv",index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

train_df


le = LabelEncoder()

def feature_engineering(df):  
    all_cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    categorical_cols = ['Sex']
    
    for c in categorical_cols:
        df[c] = le.fit_transform(df[c]) # Converts categorical column into int format
        df[c] = df[c].astype('category') # Define column type as category 

    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    
    # Adding combinations of categorical cols as columns
    gc.collect()
    
            
    return df


test_df = feature_engineering(test_df)
train_df = feature_engineering(train_df)

labels = train_df['Calories']
train_df.drop('Calories',axis=1,inplace=True)

train_df.describe()


def prep_submission(ids,preds):

    submission_df = pd.DataFrame({
        'id': ids,
        'preds' : preds[:],

    })

    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved successfully!")


import optuna
from optuna import Trial
from functools import partial
import xgboost as xgb

def objective(trial: Trial, train_df, labels):
    # Define hyperparameter search space
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda:0',
        'eval_metric': 'rmsle',
        'max_depth': trial.suggest_int('max_depth', 7, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01),
        'subsample': trial.suggest_float('subsample', 0.40, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.95),
        'gamma': trial.suggest_float('gamma', 0.0001, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5),
        'max_bin': trial.suggest_int('max_bin', 100, 5000)  
    }
    dtrain = xgb.DMatrix(train_df, labels, enable_categorical=True)
    results = xgb.cv(
        params,
        dtrain,
        num_boost_round=10000,
        nfold=3,
        early_stopping_rounds=25,
        metrics={'rmsle'},
        seed=42,
        as_pandas=True,
        verbose_eval=False
    )
    
    return results['test-rmsle-mean'].min()

if True:
    # Optimization execution
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, train_df, labels),
        n_trials=200,
        n_jobs=-1,
        gc_after_trial=True,
        show_progress_bar=True
    )


%%time

params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'rmsle',
        'max_depth': 11, 'learning_rate': 0.00909079906956036, 'subsample': 0.5909047228318444, 'colsample_bytree': 0.7439577268901377, 'gamma': 0.8310204647109323, 'min_child_weight': 1, 'reg_alpha': 7.325755580395283, 'reg_lambda': 7.637041918113586, 'max_bin': 2632
        

}


dtrain = xgb.DMatrix(train_df, labels, enable_categorical=True)

result = xgb.cv(
        params,
        dtrain,
        num_boost_round=10000,
        nfold=5,
        early_stopping_rounds=25,
        metrics={'rmsle'},
        seed=42,
        as_pandas=True,
        verbose_eval=250
)



best_rounds = result.shape[0]
final_model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_rounds+200
    )

ids= test_df["id"]
try:
    test_df_without_id = test_df.drop("id",axis=1)
except:
    pass

final_test = xgb.DMatrix(test_df_without_id,enable_categorical=True)
preds = final_model.predict(final_test)

prep_submission(ids,preds)

final_model.save_model('XGB_.json')

