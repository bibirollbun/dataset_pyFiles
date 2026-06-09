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


# Import libraries
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler



# Load the dataset
data = pd.read_csv('/kaggle/input/emory/BTC_USDT_1h.csv')


# Handle missing values in 'tradecount' without dropping rows
data['tradecount'] = data['tradecount'].fillna(0)


# Feature Engineering (avoid dropping rows)
data['close_lag_1'] = data['close'].shift(1).fillna(method='ffill')  # Forward fill for lag
data['price_change_prev'] = data['close'].pct_change().fillna(0) * 100  # Fill NaN with 0
data['high_low_ratio'] = data['high'] / data['low']  # Tỷ lệ high/low
data['volume_ratio'] = data['Volume BTC'] / data['Volume USDT'].replace(0, 1e-10)  # Tránh chia cho 0
data['volatility'] = data['high'] - data['low']  # Biến động giá


# Features for prediction
features = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT', 'tradecount', 
            'close_lag_1', 'price_change_prev', 'high_low_ratio', 'volume_ratio', 'volatility']
X = data[features]


# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Get indices for train-test split
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=42)


# Calculate target for training and test sets (handle NaN at edges)
y_train_full = np.where(data['close'].pct_change().shift(-1).fillna(0) * 100 >= 0.02, 1, 0)
y_train = y_train_full[train_idx]
y_test = y_train_full[test_idx]


# Align X_train and X_test with the calculated targets
X_train = pd.DataFrame(X_scaled[train_idx], columns=features)
X_test = pd.DataFrame(X_scaled[test_idx], columns=features)


# Train LightGBM model with GridSearchCV
model = LGBMClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)


# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")


# Predict probabilities on the full dataset
X_full_scaled = scaler.transform(X)
y_pred_proba = best_model.predict_proba(X_full_scaled)[:, 1]  # Xác suất lớp 1


# Tune threshold to maximize F1 Score
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Tránh chia cho 0
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold}")


# Apply optimal threshold to full predictions
y_pred_full = (y_pred_proba >= optimal_threshold).astype(int)


# Create submission DataFrame using all dates
submission = pd.DataFrame({
    'date': data['date'],  # Sử dụng toàn bộ 39,392 dates
    '0.02%': y_pred_full  # Dự đoán cho toàn bộ dữ liệu
})

# Verify the number of rows
print(f"Number of rows in submission: {len(submission)}")


# Calculate F1 Score on validation set with optimal threshold
y_pred_test_adjusted = (best_model.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)
f1 = f1_score(y_test, y_pred_test_adjusted)
print(f'F1 Score on validation set with adjusted threshold: {f1}')


# Save to CSV
submission.to_csv('submission.csv', index=False)

