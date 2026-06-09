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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as mpimg
import math
import plotly.express as px
import plotly.graph_objects as go

from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from IPython.display import Image
from colorama import Fore, Back, Style

y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')



from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


custom_colors = ["#ff6b6b", "#95d5b2", "#a2d2ff", "#72efdd"]
sns.set_palette(sns.color_palette(custom_colors))


sns.palplot(sns.color_palette(custom_colors), size=1)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.head()


# Missiong Values
print('Missing values in train set:')
print(train.isnull().sum())
print('\nMissing values in test set:')
print(test.isnull().sum())


# Visualization layout
fig, axes = plt.subplots(2, 5, figsize=(25, 12))
fig.suptitle('Feautre Distributions', fontsize=18)

# Plot histograms for all features
all_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy', 'BeatsPerMinute']

for i, feature in enumerate(all_features):
    row, col = i // 5, i % 5
    train[feature].hist(bins=30, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} Distribution')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# Visualization layout
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feautre Distributions', fontsize=18)

# Plot histograms for key features
features_to_plt = ['RhythmScore', 'AudioLoudness', 'MoodScore', 'TrackDurationMs', 'Energy', 'BeatsPerMinute']

for i, feature in enumerate(features_to_plt):
    row, col = i // 3, i % 3
    train[feature].hist(bins=30, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} Distribution')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# Correlation heatmap
plt.figure(figsize=(14, 12))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='mako', center=1)
plt.title('All Feature Correlation Matrix')
plt.show()


# Correlation with target feature

features_to_compare = ['TrackDurationMs', 'MoodScore', 
                       'LivePerformanceLikelihood', 'VocalContent', 'RhythmScore']

correlations = train[features_to_compare + ['BeatsPerMinute']].corr()
bpm_corr = correlations['BeatsPerMinute'].drop('BeatsPerMinute')

print("Correlation of BeatsPerMinute with selected features:")
print(bpm_corr)


plt.figure(figsize=(8, 6))
sns.heatmap(bpm_corr.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation of BeatsPerMinute with Selected Features')
plt.show()


# Target variable distribution
plt.figure(figsize=(10, 6))
sns.histplot(train['BeatsPerMinute'], kde=True)
plt.title('Distribution of Beats Per Minute (BPM)')
plt.xlabel('BPM')
plt.ylabel('Frequency')
plt.show()


# Relationship between key features and target
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

sns.scatterplot(data=train, x='RhythmScore', y='BeatsPerMinute', ax=axes[0, 0])
axes[0, 0].set_title('RhythmScore vs BPM')

sns.scatterplot(data=train, x='Energy', y='BeatsPerMinute', ax=axes[0, 1])
axes[0, 1].set_title('Energy vs BPM')

sns.scatterplot(data=train, x='AudioLoudness', y='BeatsPerMinute', ax=axes[1, 0])
axes[1, 0].set_title('AudioLoudness vs BPM')

sns.scatterplot(data=train, x='MoodScore', y='BeatsPerMinute', ax=axes[1, 1])
axes[1, 1].set_title('MoodScore vs BPM')

sns.scatterplot(data=train, x='TrackDurationMs', y='BeatsPerMinute', ax=axes[2, 0])
axes[2, 0].set_title('TrackDurationMs vs BPM')

sns.scatterplot(data=train, x='VocalContent', y='BeatsPerMinute', ax=axes[2, 1])
axes[2, 1].set_title('VocalContent vs BPM')

plt.tight_layout()
plt.show()


X = train.drop(['id', 'BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']
X_test = test.drop('id', axis=1)

def create_enhanced_features(df):
    
    df_enhanced = df.copy()
    
    # Interaction features
    df_enhanced['Rhythm_Energy'] = df_enhanced['RhythmScore'] * df_enhanced['Energy']
    df_enhanced['Loudness_Energy'] = df_enhanced['AudioLoudness'] * df_enhanced['Energy']
    df_enhanced['Mood_Energy'] = df_enhanced['MoodScore'] * df_enhanced['Energy']
    
    # Polynomial features
    df_enhanced['RhythmScore_sq'] = df_enhanced['RhythmScore'] ** 2
    df_enhanced['Energy_sq'] = df_enhanced['Energy'] ** 2
    df_enhanced['AudioLoudness_sq'] = df_enhanced['AudioLoudness'] ** 2
    
    # Ratio features (with small epsilon to avoid division by zero)
    df_enhanced['Energy_per_Rhythm'] = df_enhanced['Energy'] / (df_enhanced['RhythmScore'] + 1e-6)
    df_enhanced['Mood_per_Energy'] = df_enhanced['MoodScore'] / (df_enhanced['Energy'] + 1e-6)

    # Log / Root Transformations (Handle Skewness)
    df_enhanced['Log_Duration'] = np.log1p(df_enhanced['TrackDurationMs'])
    df_enhanced['Sqrt_Energy'] = np.sqrt(df_enhanced['Energy'])

    # Standardized Ratios (Relative Measures)
    df_enhanced['Duration_per_Mean'] = df_enhanced['TrackDurationMs'] / df_enhanced['TrackDurationMs'].mean()
    df_enhanced['Energy_per_MeanRhythm'] = df_enhanced['Energy'] / (df_enhanced['RhythmScore'].mean() + 1e-6)

    
    return df_enhanced


# Applying the features
X_enhanced = create_enhanced_features(X)
X_test_enhanced = create_enhanced_features(X_test)


X_train, X_val, y_train, y_val = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

print(f'Training set shape: {X_train.shape}')
print(f'Validation set shape: {X_val.shape}')


# CatBoost Model

cat_model=CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0
)

cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

cat_train_pred = cat_model.predict(X_train)
cat_val_pred = cat_model.predict(X_val)

# Metrices Calculation
cat_train_mx = mean_absolute_error(y_train, cat_train_pred)
cat_val_mx = mean_absolute_error(y_val, cat_val_pred)

cat_train_rmx = np.sqrt(mean_absolute_error(y_train, cat_train_pred))
cat_val_rmx = np.sqrt(mean_absolute_error(y_val, cat_val_pred))

cat_train_r2 = r2_score(y_train, cat_train_pred)
cat_val_r2 = r2_score(y_val, cat_val_pred)

print("CatBoost Result:")
print(f'Training MX: {cat_train_mx:.4f}')
print(f'Validation MX: {cat_val_mx:.4f}')
print(f'Training RMX: {cat_train_rmx:.4f}')
print(f'Validation RMX: {cat_val_rmx:.4f}')
print(f'Training R²: {cat_train_r2:.4f}')
print(f'Validation R²: {cat_val_r2:.4f}')


# Feature importance for CatBoost
cat_importance = pd.DataFrame({
    'feature': X_enhanced.columns,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=cat_importance)
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.show()


test_predictions = cat_model.predict(X_test_enhanced)

submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': test_predictions
})

print("\nSubmission BPM stats:")
print(submission['BeatsPerMinute'].describe())

submission.to_csv('/kaggle/working/submission.csv', index=False)




