# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Step 1: Data Cleaning
print("Missing values in train:\n", train.isnull().sum())
print("\nMissing values in test:\n", test.isnull().sum())
print("\nData types:\n", train.dtypes)

# Fill missing values if any
train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)

# Step 2: EDA
plt.figure(figsize=(12, 6))
sns.histplot(train['rainfall'], bins=30)
plt.title('Distribution of Rainfall')
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(train[['pressure', 'maxtemp', 'mintemp', 'humidity', 'rainfall']])
plt.show()

# Insights:
# - Humidity, cloud, and dewpoint seem positively correlated with rainfall.
# - Sunshine is negatively correlated with rainfall.

# Step 3: Feature Engineering
features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

X = train[features]
y = train['rainfall']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test[features])

# Step 4: Model Training (Random Forest for binary classification)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model RMSE: {rmse:.4f}")

# Step 5: Predict on test data and prepare submission
test['rainfall'] = model.predict(test_scaled)

# Submission file
#submission = test[['id', 'rainfall']]
#submission.to_csv('submission.csv', index=False)
#print("Submission file created: submission.csv")



# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Step 1: Data Cleaning
train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)

# Optional: Check class balance
print(train['rainfall'].value_counts())

# If imbalanced, apply SMOTE
X = train[['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']]
y = train['rainfall']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test[X.columns])

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 2: Model Training using XGBoost
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"Improved Model Accuracy: {accuracy:.4f}")
print(f"Improved Model RMSE: {rmse:.4f}")

# Step 3: Predict on test data and prepare submission
test['rainfall'] = model.predict(test_scaled)

# Submission file
#submission = test[['id', 'rainfall']]
#submission.to_csv('submission.csv', index=False)
#print("Submission file created: submission.csv")



# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Data Cleaning
train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)

# Feature and target
features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
X = train[features]
y = train['rainfall']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test[features])

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# GridSearchCV for XGBoost
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
}

xgb = XGBClassifier(random_state=42)
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                    cv=3, scoring='accuracy', n_jobs=-1, verbose=1)

grid.fit(X_train, y_train)
print(f"Best Parameters: {grid.best_params_}")

# Best model after tuning
best_model = grid.best_estimator_
y_pred = best_model.predict(X_val)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"Tuned Model Accuracy: {accuracy:.4f}")
print(f"Tuned Model RMSE: {rmse:.4f}")

# Predict on test data and prepare submission
test['rainfall'] = best_model.predict(test_scaled)
submission = test[['id', 'rainfall']]
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")





