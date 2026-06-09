# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical operations
import numpy as np
# Import time for measuring preprocessing duration
import time
# Import XGBRegressor for modeling
from xgboost import XGBRegressor
# Import train_test_split for validation split
from sklearn.model_selection import train_test_split
# Import mean_squared_error for evaluation
from sklearn.metrics import mean_squared_error
# Import pearsonr for competition metric
from scipy.stats import pearsonr
# Import os for file system operations
import os
# Iterate through directories and files in /kaggle/input
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# Set display options for full output
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load data
train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# Preview the structure
print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print("\nTrain Data Head:")
print(train_data.head())
print("\nTrain Data Info:")
print(train_data.info())
print("\nMissing Values in Train Data:")
print(train_data.isnull().sum())
print("\nTrain Data Description:")
print(train_data.describe())
print("\nTest Data Head:")
print(test_data.head())


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')

# Suppress warnings
np.seterr(invalid='ignore')

# Plot label distribution
plt.figure(figsize=(10, 5))
plt.hist(train_data['label'].clip(lower=-5, upper=5), bins=50)
plt.title('Clipped Distribution of Label')
plt.xlabel('Label Value')
plt.ylabel('Frequency')
plt.show()

# Plot distributions of key features
plt.figure(figsize=(10, 5))
plt.hist(train_data['bid_qty'].clip(lower=0, upper=100), bins=50)
plt.title('Clipped Distribution of bid_qty')
plt.xlabel('bid_qty')
plt.ylabel('Frequency')
plt.show()

# Correlation of key features with label
key_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
correlations = train_data[key_features + ['label']].corr()['label'].drop('label')
plt.figure(figsize=(8, 4))
sns.barplot(x=correlations.index, y=correlations.values)
plt.title('Correlation of Market Features with Label')
plt.xlabel('Feature')
plt.ylabel('Pearson Correlation')
plt.show()
print("Correlations with label:\n", correlations)


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Load data
train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# Feature engineering
train_data['bid_ask_spread'] = train_data['bid_qty'] - train_data['ask_qty']
train_data['buy_volume_ratio'] = train_data['buy_qty'] / (train_data['volume'] + 1e-6)
test_data['bid_ask_spread'] = test_data['bid_qty'] - test_data['ask_qty']
test_data['buy_volume_ratio'] = test_data['buy_qty'] / (test_data['volume'] + 1e-6)

# Select features (exclude timestamp, label)
features = [col for col in train_data.columns if col not in ['timestamp', 'label']]
X = train_data[features]
y = train_data['label']
X_test = test_data[features]

# Feature selection with SelectKBest
selector = SelectKBest(score_func=f_regression, k=50)
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)
selected_features = [features[i] for i in selector.get_support(indices=True)]
print("Selected features:", selected_features)

# Standardize features
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)
X_test_selected = scaler.transform(X_test_selected)

# Time-based train-validation split
train_size = int(0.8 * len(X_selected))
X_train, X_val = X_selected[:train_size], X_selected[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
corr, _ = pearsonr(y_val, y_pred)
print("Validation Pearson correlation:", corr)

# Predict on test set
test_pred = model.predict(X_test_selected)

# Create submission
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submission['label'] = test_pred
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created: /kaggle/working/submission.csv")


# Remove existing 'level_0' if present, then reset the index to ensure 'timestamp' is a column
import numpy as np
np.seterr(invalid='ignore')

if 'level_0' in train_data.columns:
    train_data = train_data.drop(columns=['level_0'])
train_data = train_data.reset_index()

# Create a lag feature by shifting the 'label' column down by one row to use the previous value as a predictor
train_data['lag_1'] = train_data['label'].shift(1)

# Display the first few rows of 'timestamp', 'label', and 'lag_1' to verify the lag feature
print(train_data[['timestamp', 'label', 'lag_1']].head())

# Select key features ('bid_qty', 'ask_qty', 'volume', 'lag_1', 'label') and remove rows with NaN values (from the first lag)
X = train_data[['bid_qty', 'ask_qty', 'volume', 'lag_1', 'label']].dropna()

# Extract the 'label' column from X as the target variable y, removing it from the feature set
y = X.pop('label')

# Display the first few rows of the feature set X to confirm the structure
print(X.head())

# Display the first few values of the target variable y to verify it matches X
print(y.head())


import numpy as np
np.seterr(invalid='ignore')
# Adding more from competition data
X_expanded = train_data[['bid_qty', 'ask_qty', 'volume', 'lag_1', 'label', 'X1', 'X2', 'X3']].dropna()
y_expanded = X_expanded.pop('label')
from sklearn.model_selection import train_test_split
X_train_exp, X_val_exp, y_train_exp, y_val_exp = train_test_split(X_expanded, y_expanded, test_size=0.2, shuffle=False)
from xgboost import XGBRegressor
model_exp = XGBRegressor()
model_exp.fit(X_train_exp, y_train_exp)
from scipy.stats import pearsonr
predictions = model_exp.predict(X_val_exp)
corr, _ = pearsonr(predictions, y_val_exp)
print("Expanded Pearson correlation:", corr)


import numpy as np 
np.seterr(invalid='ignore')
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

#Test Robot Settings 
param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
model = XGBRegressor()
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')  # Note: Use Pearson if possible
grid_search.fit(X_train_exp, y_train_exp)

#Best Settings and Score
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_val_exp)
from scipy.stats import pearsonr
corr, _ = pearsonr(predictions, y_val_exp)
print("Tuned Pearson correlation:", corr)


import numpy as np
np.seterr(invalid='ignore')
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Build two robots
xgb_model = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=100)  # Use tuned params
lin_model = LinearRegression()

# Make a super team
ensemble = VotingRegressor([('xgb', xgb_model), ('lin', lin_model)])
ensemble.fit(X_train_exp, y_train_exp)

# Check super team score
predictions = ensemble.predict(X_val_exp)
corr, _ = pearsonr(predictions, y_val_exp)
print("Ensemble Pearson correlation:", corr)


import pandas as pd
np.seterr(invalid='ignore')

# Take a smaller sample to save resources
train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet').sample(frac=0.1, random_state=42)
print(train_data.shape)
test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
print(test_data.columns)




