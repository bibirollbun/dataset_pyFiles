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


import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import xgboost as xgb



INPUT_PATH = '/kaggle/input/playground-series-s5e6/'


df_train = pd.read_csv(INPUT_PATH + 'train.csv')
df_test = pd.read_csv(INPUT_PATH + 'test.csv')
submission = pd.read_csv(INPUT_PATH + 'sample_submission.csv')


df_train.head()


df_test.head()


submission.head()


df_train.info()


df_test.info()


df_train.describe()


categorical_var = df_train.select_dtypes(include='object').columns
categorical_var


df_train['Soil Type'].value_counts()


df_train['Crop Type'].value_counts()


df_train['Fertilizer Name'].value_counts()


non_categorical = df_train.select_dtypes(include='int64').columns


df_train.groupby('Fertilizer Name')[non_categorical].mean()


for non_cat in non_categorical:
    plt.figure(figsize=(12, 6))

    sns.boxplot(x='Fertilizer Name', y=non_cat, data=df_train)
    
    plt.show()


df_sample = df_train.sample(n=5000, random_state=42)

plt.figure(figsize=(12, 8))

# Plot using the sample
sns.scatterplot(x='Temparature', y='Potassium', hue='Fertilizer Name', data=df_sample)

plt.show()


categorical_features = ['Soil Type', 'Crop Type']
df_prepared = pd.get_dummies(df_train, columns=categorical_features, drop_first=True)

print("Original number of columns:", df_train.shape[1])
print("New number of columns after encoding:", df_prepared.shape[1])
print("\nFirst 5 rows of the prepared data:")
print(df_prepared.head())


# X contains all columns EXCEPT the target and any ID columns.
X = df_prepared.drop(columns=['id', 'Fertilizer Name'])

# y contains ONLY our target column.
y = df_prepared['Fertilizer Name']


X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)


model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


model_rf.fit(X_train, y_train)
print("Model training complete.")


# Make predictions on the validation set
predictions = model_rf.predict(X_val)

# Calculate the accuracy
accuracy = accuracy_score(y_val, predictions)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")


le = LabelEncoder()
# Fit on the full y to learn all possible fertilizers
le.fit(y)

# Transform our training and validation sets
y_train_encoded = le.transform(y_train)
y_val_encoded = le.transform(y_val)


# Create and train the XGBoost model
# 'multi:softprob' tells it to return probabilities for each class
powerful_xgb = xgb.XGBClassifier(
    n_estimators=500,         # More trees
    max_depth=8,              # Deeper trees to find interactions
    learning_rate=0.05,       # A smaller learning rate
    objective='multi:softprob',
    n_jobs=-1,
    random_state=42
)

print("Training a more powerful XGBoost model...")
powerful_xgb.fit(X_train, y_train_encoded)



new_probabilities = powerful_xgb.predict_proba(X_val)
new_loss = log_loss(y_val_encoded, new_probabilities)

print(f"Old Validation Log Loss: 1.9179")
print(f"New Validation Log Loss: {new_loss:.4f}")


categorical_features = ['Soil Type', 'Crop Type']
df_test_prepared = pd.get_dummies(df_test, columns=categorical_features, drop_first=True)

print(df_test_prepared.head())


X_test = df_test_prepared.drop(columns=['id'])


probabilities = powerful_xgb.predict_proba(X_test)


top_3_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
top_3_labels = le.classes_[top_3_indices]



predictions_formatted = [" ".join(labels) for labels in top_3_labels]



submission_df = pd.DataFrame({
    'id': df_test['id'], # Get the id from the original test dataframe
    'Fertilizer Name': predictions_formatted
})


# Save to a .csv file, without the pandas index
submission_df.to_csv('submission.csv', index=False)


print("Submission file 'submission.csv' created successfully!")
print(submission_df.head())




