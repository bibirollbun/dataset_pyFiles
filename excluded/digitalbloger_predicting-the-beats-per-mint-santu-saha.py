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


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
sns.set_style('whitegrid')
print("Libraries imported successfully and warnings are suppressed!")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
sns.set_style('whitegrid')
print("Libraries imported successfully!")


TRAIN_PATH = '/kaggle/input/playground-series-s5e9/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e9/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/playground-series-s5e9/sample_submission.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission_df = pd.read_csv(SAMPLE_SUB_PATH)

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print("\nTraining Data Head:")
train_df.head()



print("Training Data Info:")
train_df.info()
print("\n" + "="*50 + "\n")
print("Test Data Info:")
test_df.info()


plt.figure(figsize=(16, 6))
sns.histplot(train_df['BeatsPerMinute'], kde=True, bins=50)
plt.title('Distribution of BeatsPerMinute in the Training Data', fontsize=16)
plt.xlabel('BeatsPerMinute', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
mean_bpm = train_df['BeatsPerMinute'].mean()
median_bpm = train_df['BeatsPerMinute'].median()
plt.axvline(mean_bpm, color='r', linestyle='--', label=f'Mean: {mean_bpm:.2f}')
plt.axvline(median_bpm, color='g', linestyle='-', label=f'Median: {median_bpm:.2f}')
plt.legend()

plt.show()


features = [col for col in train_df.columns if col not in ['id', 'BeatsPerMinute']]
X_train = train_df[features]
y_train = train_df['BeatsPerMinute']
X_test = test_df[features]

params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = np.zeros(X_train.shape[0])
test_predictions = np.zeros(X_test.shape[0])
oof_rmse_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    print(f"===== FOLD {fold+1} =====")
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    val_preds = model.predict(X_val_fold)
    oof_predictions[val_index] = val_preds
    
    rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
    oof_rmse_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse}")
    
    test_predictions += model.predict(X_test) / N_SPLITS
mean_oof_rmse = np.mean(oof_rmse_scores)
print(f"\nAverage Cross-Validation RMSE: {mean_oof_rmse}")


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'BeatsPerMinute': test_predictions
})

submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' has been created successfully!")
print("Here is a preview of your submission file with the correct 'id' header:")
print(submission_df.head())


corr_matrix = train_df.corr()
bpm_corr = corr_matrix[['BeatsPerMinute']].sort_values(by='BeatsPerMinute', ascending=False)
plt.figure(figsize=(10, 12))
sns.heatmap(bpm_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation with BeatsPerMinute', fontsize=16)
plt.show()


lgb.plot_importance(model, figsize=(10, 10), max_num_features=20, height=0.8)
plt.title('Top 20 Most Important Features (from last CV fold)', fontsize=16)
plt.show()


oof_df = pd.DataFrame({
    'Actual': y_train,
    'Predicted': oof_predictions
})
plt.figure(figsize=(10, 10))
sns.scatterplot(data=oof_df, x='Actual', y='Predicted', alpha=0.3)
max_val = max(oof_df['Actual'].max(), oof_df['Predicted'].max())
min_val = min(oof_df['Actual'].min(), oof_df['Predicted'].min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Perfect Prediction')

plt.title('Actual vs. Predicted BeatsPerMinute (Out-of-Fold)', fontsize=16)
plt.xlabel('Actual BPM', fontsize=12)
plt.ylabel('Predicted BPM', fontsize=12)
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()


plt.figure(figsize=(16, 8))

sns.kdeplot(y_train, label='Actual Target (Train)', color='blue', linewidth=2)

sns.kdeplot(oof_predictions, label='OOF Predictions', color='orange', linestyle='--', linewidth=2)

sns.kdeplot(test_predictions, label='Test Predictions', color='green', linestyle=':', linewidth=2)

plt.title('Distribution Comparison: Target vs. Predictions', fontsize=16)
plt.xlabel('BeatsPerMinute', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.show()


residuals = y_train - oof_predictions

plt.figure(figsize=(12, 7))
sns.scatterplot(x=y_train, y=residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--', lw=2) # The zero-error line

plt.title('Residuals (Errors) vs. Actual Values', fontsize=16)
plt.xlabel('Actual BeatsPerMinute', fontsize=12)
plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
plt.grid(True)
plt.show()


importance_df = pd.DataFrame({
    'feature': model.feature_name_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

TOP_FEATURE = importance_df.iloc[0]['feature']
print(f"The most important feature is: {TOP_FEATURE}\n")


low_bpm_threshold = train_df['BeatsPerMinute'].quantile(0.25) # Bottom 25%
high_bpm_threshold = train_df['BeatsPerMinute'].quantile(0.75) # Top 25%

low_bpm_songs = train_df[train_df['BeatsPerMinute'] <= low_bpm_threshold]
high_bpm_songs = train_df[train_df['BeatsPerMinute'] >= high_bpm_threshold]


plt.figure(figsize=(12, 7))
sns.kdeplot(low_bpm_songs[TOP_FEATURE], label=f'Low BPM (<= {low_bpm_threshold:.0f})', color='skyblue', fill=True)
sns.kdeplot(high_bpm_songs[TOP_FEATURE], label=f'High BPM (>= {high_bpm_threshold:.0f})', color='salmon', fill=True)

plt.title(f'Distribution of "{TOP_FEATURE}" for Low vs. High BPM Songs', fontsize=16)
plt.xlabel(f'Value of {TOP_FEATURE}', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.show()


plt.figure(figsize=(12, 7))

sns.kdeplot(train_df[TOP_FEATURE], label='Train Set', color='blue', linewidth=2)
sns.kdeplot(test_df[TOP_FEATURE], label='Test Set', color='red', linestyle='--', linewidth=2)

plt.title(f'Distribution of "{TOP_FEATURE}" in Train vs. Test Data', fontsize=16)
plt.xlabel(f'Value of {TOP_FEATURE}', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.show()



submission_df = pd.DataFrame({
    'id': test_df['id'],  # <-- This is the corrected part
    'BeatsPerMinute': test_predictions
})


submission_df.to_csv('submission.csv', index=True)

print("\nSubmission file 'submission.csv' has been created successfully!")
print("Here is a preview of your submission file ")
print(submission_df.head())

