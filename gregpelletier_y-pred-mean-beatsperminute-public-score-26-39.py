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


!pip install git+https://github.com/gjpelletier/PyMLR.git --upgrade


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import skew
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


sample_submission_path = '/kaggle/input/playground-series-s5e9/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e9/train.csv'
test_path = '/kaggle/input/playground-series-s5e9/test.csv'

samp_sub = pd.read_csv(sample_submission_path)
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("Size of train data : " , df_train.shape)
print("Size of test data : " , df_test.shape)


# show the dtypes of the train data and check for missing data
# note: there are no categorical features and no missing data

from PyMLR import show_dtypes
dtypes_train = show_dtypes(df_train)


# describe the dataset
df_train.describe().T


# show histrograms of the data
df_train.hist(bins=20, figsize=(15, 10));


# Compute the correlation matrix
correlation_matrix = df_train.corr()

# Visualize the correlation matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


y_pred = df_train['BeatsPerMinute'].mean()

# Save your predictions as a CSV
to_save = df_test[['id']].copy()
to_save.loc[:, 'BeatsPerMinute'] = y_pred
to_save.to_csv('/kaggle/working/submission_mean.csv', index=False)


# Creating new features 
# adapted from https://www.kaggle.com/code/amritanshukush/beats-per-minute-lightgbm-s5e9

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

# add the new features into the train and test dataframes

df_train = new_features(df_train)
df_test = new_features(df_test)

print('Train size : ' , df_train.shape)
print('Test shape : ',df_test.shape)


# Compute the correlation matrix including the engineered features
# note that BeatsPerMinute has practically no correlation with the new features

correlation_matrix = df_train.corr()

# Visualize the correlation matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


print('Correlation coefficients between BeatsPerMinute and all features,')
print('including engineered features.')
print('Note: all correlation coefficients are <=0.01\n')
print(correlation_matrix['BeatsPerMinute'])


TARGET = 'BeatsPerMinute'
X_train = df_train.drop([TARGET , 'id'] , axis=1)
y_train = df_train[TARGET]

X_test = df_test.drop('id' , axis=1)


from PyMLR import xgb
train_model_xgb, train_output_xgb = xgb(X_train, y_train, gpu=False)


# Fit the model with the training data
from PyMLR import lgbm
train_model_lgbm, train_output_lgbm = lgbm(X_train, y_train, gpu=False)


from PyMLR import catboost
train_model_cat, train_output_cat = catboost(X_train, y_train, gpu=False)

