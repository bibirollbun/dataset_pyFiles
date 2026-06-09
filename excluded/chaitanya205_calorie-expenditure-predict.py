import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

import os



train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train.head()



print(train.info())
print(train.describe())

# Check for missing values
print(train.isnull().sum())

# Visualize target
sns.histplot(train['Calories'], kde=True, bins=50)
plt.title("Target Distribution - Calories Burned")
plt.show()




X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
X_test = test.drop(['id'], axis=1)



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.preprocessing import LabelEncoder

# Combine train and test for consistent encoding
combined = pd.concat([X, X_test], axis=0)
categorical_cols = combined.select_dtypes(include=['object']).columns

# Encode all categorical features
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Split back
X = combined.iloc[:len(X), :]
X_test = combined.iloc[len(X):, :]




X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)



# RMSLE Evaluation
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

print("Validation RMSLE:", rmsle(y_val, val_preds))


test_preds = model.predict(X_test)

submission['Calories'] = test_preds
submission.to_csv('submission.csv', index=False)



# Concatenate X and X_test for consistent one-hot encoding
combined = pd.concat([X, X_test], axis=0)

# Perform One-Hot Encoding
combined_encoded = pd.get_dummies(combined, drop_first=True)

# Split back
X = combined_encoded.iloc[:len(X), :]
X_test = combined_encoded.iloc[len(X):, :]



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)

# RMSLE Calculation
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

print("Validation RMSLE:", rmsle(y_val, val_preds))



test_preds = model.predict(X_test)

submission['Calories'] = test_preds
submission.to_csv('submission.csv', index=False)





