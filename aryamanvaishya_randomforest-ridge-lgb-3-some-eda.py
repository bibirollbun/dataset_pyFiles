# Kaggle Playground — BPM Prediction
# Starter notebook / script for EDA, modeling and submission
# Save this file as Kaggle Notebook (.ipynb)

# 1. Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb


# 2. Paths
TRAIN_PATH = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e9/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/playground-series-s5e9/sample_submission.csv"


# 3. Utility functions

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 4. Load data
print('Loading data...')
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print('Train shape:', train.shape)
print('Test shape:', test.shape)

# Quick peek
train.head()


# 5. Basic EDA — distributions, missing values, correlations

def basic_eda(df):
    print('--- shape ---')
    print(df.shape)
    print('\n--- dtypes ---')
    print(df.dtypes)
    print('\n--- missing ---')
    print(df.isnull().sum())
    print('\n--- descriptive ---')
    display(df.describe().T)

basic_eda(train)


# Distribution of target
plt.figure(figsize=(8,4))
plt.hist(train['BeatsPerMinute'].dropna(), bins=60, color='skyblue', edgecolor='black')
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BPM')
plt.ylabel('Count')
plt.show()


# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(train.corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()


# Pairplots for key features
sns.pairplot(train[['BeatsPerMinute','RhythmScore','Energy','AudioLoudness','MoodScore']])
plt.show()


# Boxplots to check feature distributions vs BPM bins
train['BPM_bin'] = pd.qcut(train['BeatsPerMinute'], q=4, labels=False)
plt.figure(figsize=(10,6))
sns.boxplot(x='BPM_bin', y='Energy', data=train)
plt.title('Energy vs BPM bins')
plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(x='BPM_bin', y='RhythmScore', data=train)
plt.title('RhythmScore vs BPM bins')
plt.show()


# Scatterplots for relationships
fig, axs = plt.subplots(2,2, figsize=(12,10))
axs[0,0].scatter(train['RhythmScore'], train['BeatsPerMinute'], alpha=0.3)
axs[0,0].set_title('RhythmScore vs BPM')
axs[0,1].scatter(train['Energy'], train['BeatsPerMinute'], alpha=0.3)
axs[0,1].set_title('Energy vs BPM')
axs[1,0].scatter(train['AudioLoudness'], train['BeatsPerMinute'], alpha=0.3)
axs[1,0].set_title('AudioLoudness vs BPM')
axs[1,1].scatter(train['MoodScore'], train['BeatsPerMinute'], alpha=0.3)
axs[1,1].set_title('MoodScore vs BPM')
plt.tight_layout()
plt.show()


# More Pairplots
# Pairplots for key features
sns.pairplot(train[['RhythmScore',	'AudioLoudness',	'VocalContent',	'AcousticQuality',	'InstrumentalScore',	'LivePerformanceLikelihood',	'MoodScore',	'TrackDurationMs',	'Energy']])
plt.show()


# KDE plots for continuous features
for col in ['RhythmScore','AudioLoudness','Energy','MoodScore','TrackDurationMs']:
    plt.figure(figsize=(8,4))
    sns.kdeplot(train[col], fill=True, color='teal')
    plt.title(f'Distribution of {col}')
    plt.show()


# 6. Feature engineering
train['TrackDurationSec'] = train['TrackDurationMs'] / 1000.0
test['TrackDurationSec'] = test['TrackDurationMs'] / 1000.0

for df in (train, test):
    df['Energy_x_Mood'] = df['Energy'] * df['MoodScore']
    df['Rhythm_x_Energy'] = df['RhythmScore'] * df['Energy']
    df['Loudness_per_sec'] = df['AudioLoudness'] / (df['TrackDurationSec'] + 1e-6)

# 7. Prepare features
FEATURES = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore',
    'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationSec', 'Energy',
    'Energy_x_Mood', 'Rhythm_x_Energy', 'Loudness_per_sec'
]

X = train[FEATURES]
y = train['BeatsPerMinute']
X_test = test[FEATURES]


# KDE plots for continuous features
for col in ['VocalContent','AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','Energy','TrackDurationSec',"Energy_x_Mood","Rhythm_x_Energy","Loudness_per_sec"]:
    plt.figure(figsize=(8,4))
    sns.kdeplot(train[col], fill=True, color='teal')
    plt.title(f'Distribution of {col}')
    plt.show()


# 8. Baseline mean predictor
print('Baseline RMSE (mean):', rmse(y, np.repeat(y.mean(), len(y))))


# 9. Models
numeric_features = FEATURES
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)],
    remainder='drop'
)

ridge = Pipeline(steps=[('pre', preprocessor), ('model', Ridge(alpha=1.0, random_state=42))])
rf = Pipeline(steps=[('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))])


# 10. LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': 42,
    'verbosity': -1
}
train_set = lgb.Dataset(X, label=y)
model_lgb = lgb.train(params, train_set, num_boost_round=500)


# 11. Fit RandomForest and predict
rf.fit(X, y)
preds_rf = rf.predict(X_test)


# 12. Fit Ridge and predict
ridge.fit(X, y)
preds_ridge = ridge.predict(X_test)


# 13. LightGBM predict
preds_lgb = model_lgb.predict(X_test)


# 14. Simple ensemble (average)
preds_final = (preds_rf + preds_ridge + preds_lgb) / 3


# 15. Submission
submission = pd.DataFrame({'ID': test['id'], 'BeatsPerMinute': preds_final})
submission.to_csv('submission.csv', index=False)
submission.head()




