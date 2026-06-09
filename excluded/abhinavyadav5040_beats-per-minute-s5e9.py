import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from scipy.stats import skew

import lightgbm as lgb
import xgboost as xgb
import catboost as catb

import warnings
warnings.filterwarnings('ignore')


samp_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
train_data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

print("--------TRAIN DATA---------")
display(train_data.head(3))

print('\n')
print("\n--------TEST DATA----------")
display(test_data.head(3))

print('\n')
print("\n---------SAMPLE SUBMISSION---------")
display(samp_sub.head(3))


print('Train size : ' , train_data.shape)
print('Test size : ' , test_data.shape)


display(train_data.describe())
train_data.info()
train_data.isnull().sum()


plt.figure(figsize=(14,8))
plt.plot(train_data['BeatsPerMinute'].head(500))


num_cols = train_data.select_dtypes(include=['int' , 'float']).columns.drop('id')

plt.figure(figsize=(10,8))
corr_matrix = train_data[num_cols].corr()

sns.heatmap(corr_matrix , annot = True, cmap='viridis' , fmt = ".2f")
plt.title("Correlation between Numerical features")
plt.show()


def new_features(df):
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
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


train_X = train_data.drop(['BeatsPerMinute' , 'id'] , axis=1)
train_y = train_data['BeatsPerMinute']

test_x = test_data.drop('id' , axis=1)


X_train , X_test , y_train , y_test = train_test_split(train_X , train_y , test_size=0.2 , random_state=42)


lgb_model = lgb.LGBMRegressor(
    learning_rate = 0.03,
    num_leaves = 30,
    max_depth = 10,
    reg_alpha = 0.08,
    reg_lambda = 0.6,
    subsample = 0.45
)
lgb_model.fit(X_train , y_train , eval_set = [(X_test , y_test)] , callbacks=[lgb.early_stopping(stopping_rounds=100)])

lgb_preds = lgb_model.predict(X_test)

lgb_rmse = np.sqrt(mean_squared_error(y_test , lgb_preds))
print('rmse : ' , lgb_rmse)


lgb_pred = lgb_model.predict(test_x)

print("lgb preds : " , lgb_pred)


final_preds = np.round(lgb_pred , 3)

print('final :' , final_preds)


submission = pd.DataFrame({"id": test_data["id"], "BeatsPerMinute": final_preds})
submission.to_csv("submission.csv", index=False)
submission.head()

