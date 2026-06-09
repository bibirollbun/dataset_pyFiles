import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Encode 'Sex'
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])  # male=1, female=0
test['Sex'] = le.transform(test['Sex'])

# Prepare features and target
X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])  # log1p for RMSLE
X_test = test.drop(columns=['id'])

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Validation
y_pred_val = model.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_val)))
print("Validation RMSLE:", rmsle)

# Predict on test
y_test_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_log)

# Submission
submission = pd.DataFrame({'id': test['id'], 'Calories': y_test_pred})
submission.to_csv('submission.csv', index=False)


