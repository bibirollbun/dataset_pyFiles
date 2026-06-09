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


# Complete Workflow - Predicting BeatsPerMinute

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# Load Dataset
# Replace path if needed
df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df = df.replace([np.inf, -np.inf], np.nan)
df.head()

# EDA
print(df.info())
print(df.describe())

# Histogram of target
plt.figure(figsize=(8,5))
sns.histplot(df['BeatsPerMinute'], kde=True)
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Count')
plt.show()

# Feature distributions
features = ['RhythmScore','AudioLoudness','VocalContent','AcousticQuality','InstrumentalScore',
            'LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']
df[features].hist(figsize=(12,8), bins=30)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='viridis', annot=False)
plt.title('Correlation Heatmap')
plt.show()

# Simple Linear Regression (Energy)
X = df[['Energy']]
y = df['BeatsPerMinute']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('RMSE (Simple LR):', mean_squared_error(y_test, y_pred, squared=False))
print('R² (Simple LR):', r2_score(y_test, y_pred))

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Energy')
plt.ylabel('BeatsPerMinute')
plt.title('Simple Linear Regression: Energy → BeatsPerMinute')
plt.legend()
plt.show()

# Multiple Linear Regression
X = df[features]
y = df['BeatsPerMinute']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
y_pred_multi = multi_model.predict(X_test)

print('RMSE (Multiple LR):', mean_squared_error(y_test, y_pred_multi, squared=False))
print('R² (Multiple LR):', r2_score(y_test, y_pred_multi))

# Residual Analysis
residuals = y_test - y_pred_multi

# Residuals vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_pred_multi, residuals)
plt.axhline(0, linestyle='--')
plt.xlabel('Predicted BeatsPerMinute')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()

# Histogram of residuals
plt.figure(figsize=(7,5))
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.xlabel('Residuals')
plt.show()

# Q-Q plot
plt.figure(figsize=(6,6))
stats.probplot(residuals, dist='norm', plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()

# Feature Importance / Coefficients
coef = pd.DataFrame({'Feature': features, 'Coefficient': multi_model.coef_})
coef_sorted = coef.sort_values(by='Coefficient', ascending=False)
print(coef_sorted)

plt.figure(figsize=(10,6))
plt.barh(coef_sorted['Feature'], coef_sorted['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.gca().invert_yaxis()
plt.show()



test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df.head()
X_test_submission = test_df[features]  # same features used in training
predictions = multi_model.predict(X_test_submission)
submission = pd.DataFrame({
    'id': test_df['id'],
    'BeatsPerMinute': predictions
})

submission.to_csv('submission.csv', index=False)
submission.head()


