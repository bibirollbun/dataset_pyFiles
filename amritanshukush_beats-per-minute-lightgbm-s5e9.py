import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from scipy.stats import skew

import lightgbm as lgb
import xgboost as xgb
import catboost as catb

import optuna 

import warnings
warnings.filterwarnings('ignore')


# Dataset paths
sample_submission_path = '/kaggle/input/playground-series-s5e9/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e9/train.csv'
test_path = '/kaggle/input/playground-series-s5e9/test.csv'


samp_sub = pd.read_csv(sample_submission_path)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("--------TRAIN DATA---------")
display(train_data.head(3))
print("Size of train data : " , train_data.shape)

print('\n')
print("\n--------TEST DATA----------")
display(test_data.head(3))
print("Size of test data : " , test_data.shape)

print('\n')
print("\n---------SAMPLE SUBMISSION---------")
display(samp_sub.head(3))


print("Train Data Statistics :\n")
display(train_data.describe())
train_data.info()

print("\n-------Null Values count:---------")

train_data.isnull().sum()


def outlier_plot(data, exclude_columns, box_color='pink', median_color='brown', whisker_color='purple'):
    columns = data.drop(exclude_columns, axis=1, errors='ignore').columns
    
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2 
    
    plt.figure(figsize=(15, 5 * n_rows)) 
    
    for i, column in enumerate(columns, 1):
        plt.subplot(n_rows, 2, i)
        plt.boxplot(
            data[column].dropna(), 
            vert=False,
            patch_artist=True,  
            boxprops=dict(facecolor=box_color, color=whisker_color),
            medianprops=dict(color=median_color),
            whiskerprops=dict(color=whisker_color),
            capprops=dict(color=whisker_color),
            flierprops=dict(marker='o', color=whisker_color, markersize=5)
        )
        plt.title(f'{column}')
        plt.xlabel(column)
        plt.grid(False)

    plt.tight_layout()
    plt.show()


outlier_plot(train_data , ['BeatsPerMinute' , 'id'])


def remove_outliers(data , outlier_columns):
    
    for column in outlier_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        data = data[
            (data[column] >= lower_bound) & (data[column] <= upper_bound)
        ]
    
    return data


OUTLIER_COLUMNS = ['AudioLoudness' , 'VocalContent' , 'AcousticQuality' , 'InstrumentalScore' , 'LivePerformanceLikelihood' , 'TrackDurationMs']

train_data = remove_outliers(train_data , OUTLIER_COLUMNS)


outlier_plot(train_data , ['BeatsPerMinute' , 'id'])


# Creating new features 

def new_features(df):    
    # Interaction features
    df['Rhythm_Audio_Interaction'] = df['RhythmScore'] * df['AudioLoudness']
    df['Vocal_Acoustic_Ratio'] = df['VocalContent'] / (df['AcousticQuality'] + 1e-6)
    df['Energy_Mood_Product'] = df['Energy'] * df['MoodScore']
    df['Instrumental_Live_Interaction'] = df['InstrumentalScore'] * df['LivePerformanceLikelihood']
    
    # Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[['RhythmScore', 'AudioLoudness', 'Energy']])
    poly_cols = [f'poly_{i}' for i in range(poly_features.shape[1])]
    df[poly_cols] = poly_features
    
    # Log transformation for skewed features
    for col in ['TrackDurationMs', 'AudioLoudness', 'VocalContent']:
        if col in df.columns and skew(df[col].dropna()) > 0.5:
            if df[col].min() < 0:
                shift = abs(df[col].min()) + 1
                df[f'log_{col}'] = np.log1p(df[col] + shift)
            else:
                df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    # Binning features
    df['Duration_Bin'] = pd.qcut(df['TrackDurationMs'], q=10, labels=False, duplicates='drop')
    df['Energy_Bin'] = pd.qcut(df['Energy'], q=5, labels=False, duplicates='drop')
    
    return df


train_data = new_features(train_data)
test_data = new_features(test_data)


print('Train size : ' , train_data.shape)
print('Test shape : ',test_data.shape)


# Training data after adding new features

print("-----New Train data with fresh new Features----\n")
train_data.head()


TARGET = 'BeatsPerMinute'
N_SPLITS = 5
lgb_models , lgb_scores = [] , []

def rmse(y , preds):
    return np.sqrt(mean_squared_error(y , preds))


train_x = train_data.drop([TARGET , 'id'] , axis=1)
train_y = train_data[TARGET]

test_x = test_data.drop('id' , axis=1)


# Using optuna to find the best parameters for LightGBM

def lgb_objective(trial):
    lgb_params = {
    'learning_rate' : trial.suggest_float('learning_rate' , 0.01 , 0.1 , log=True),
    'num_leaves' : trial.suggest_int('num_leaves' , 5 , 50),
    'max_depth' : trial.suggest_int('max_depth' , 3 , 15),
    'reg_alpha' : trial.suggest_float('reg_alpha' , 0.01 , 0.1 , log=True),
    'reg_lambda' : trial.suggest_float('reg_lambda' , 0.01 , 0.1 , log=True),
    'subsample' : trial.suggest_float('subsample' , 0 , 1)
             }

    lgb_model = lgb.LGBMRegressor(**lgb_params)

    val_rmse = 0.0
    oof_preds = np.zeros(len(train_x))
    fold_rmses=[]

    kf = KFold(n_splits = 5 , shuffle=True , random_state=0)

    for train_idx , val_idx in kf.split(train_x , train_y):

        X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
        Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]

        lgb_model.fit(X_train , Y_train , eval_set = [(X_val , Y_val)])
        val_pred = lgb_model.predict(X_val)
        oof_preds[val_idx] = val_pred
    
        val_rmse = rmse(Y_val , val_pred)
        fold_rmses.append(val_rmse)

    return float(np.mean(fold_rmses))

# To run, you can un-comment below lines and run it
        
#study = optuna.create_study(direction='minimize')
#study.optimize(lgb_objective, n_trials=5, show_progress_bar=True)
#best_params = study.best_trial.params
#print("\nBest Hyperparameters from Optuna:")
#print(best_params)


# Optimal parameters for LightGBM from optuna
best_params = {
    'learning_rate' : 0.02493303813836915,
    'num_leaves' : 34,
    'max_depth' : 13,
    'reg_alpha' : 0.07319459385373907,
    'reg_lambda' : 0.08870554420170793,
    'subsample' : 0.5514983899131313,
}


lgb_model = lgb.LGBMRegressor(**best_params)

oof_preds = np.zeros(len(train_x))

kf = KFold(n_splits = N_SPLITS , shuffle=True , random_state=0)

for train_idx , val_idx in kf.split(train_x , train_y):
    print('\nFold:' , len(lgb_models) + 1)
    X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
    Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]

    lgb_model.fit(X_train , Y_train , eval_set = [(X_val , Y_val)])
    val_pred = lgb_model.predict(X_val)
    oof_preds[val_idx] = val_pred

    lgb_model_rmse = rmse(Y_val , val_pred)

    lgb_models.append(lgb_model) , lgb_scores.append(lgb_model_rmse)

print('\nScores :', lgb_scores)


# Deploying trained model on test set
lgb_test_preds = sum(lgb_model.predict(test_x) for lgb_model in lgb_models) / len(lgb_models)


# Rounding off the result as shown in sample submission

final_preds = np.round(lgb_test_preds , 3)
final_preds


submission = pd.DataFrame({'id': test_data['id'], 'BeatsPerMinute': final_preds})
submission.to_csv('submission.csv', index=False)
display(submission.head())

