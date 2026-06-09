import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


# Basic information about the datasets
print("Train Data Info:")
print(train.info())
print("\nTest Data Info:")
print(test.info())


# Display the first few rows
print("\nFirst few rows of the training data:")
print(train.head())

# Check for missing values
print("\nMissing values in training data:")
print(train.isnull().sum())

print("\nMissing values in test data:")
print(test.isnull().sum())


test['winddirection'].fillna(test['winddirection'].median(), inplace=True)


# Descriptive statistics of numerical columns
print("\nDescriptive statistics of the training data:")
print(train.describe())


# train['wind_x'] = np.sin(np.radians(train['winddirection']))
# train['wind_y'] = np.cos(np.radians(train['winddirection']))
# test['wind_x'] = np.sin(np.radians(test['winddirection']))
# test['wind_y'] = np.cos(np.radians(test['winddirection']))
# train.drop(columns=['winddirection'], inplace=True)
# test.drop(columns=['winddirection'], inplace=True)


scaler = StandardScaler()
num_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed','winddirection']
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


X = train.drop(columns=['id', 'rainfall'])  # Features
y = train['rainfall']  # Target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))


test_X = test.drop(columns=['id'])
test_preds = model.predict(test_X)


submission = pd.DataFrame({'id': test['id'], 'rainfall': test_preds})
submission.to_csv('submission.csv', index=False)

