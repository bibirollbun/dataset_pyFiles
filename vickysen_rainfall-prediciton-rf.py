
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np # Used for handling NaNs
import os

# Define file paths in the Kaggle environment
train_path = '/kaggle/input/playground-series-s5e3/train.csv'
test_path = '/kaggle/input/playground-series-s5e3/test.csv'
sample_sub_path = '/kaggle/input/playground-series-s5e3/sample_submission.csv'

# Load the data
try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_sub_path)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files. Ensure the data is attached to the notebook. {e}")


train_df.columns


#Check for missing values in training data
train_df.isnull().sum().sum()


#Check for missing values in test data
test_df.isnull().sum().sum()


test_df[test_df.isnull().any(axis=1)]


# Fill missing values with the mean
test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mean())


test_df.isnull().sum().sum()


X_train = train_df.drop(['rainfall','id'],axis=1)
y_train = train_df['rainfall']
X_test = test_df.drop('id',axis=1)


rf_model = RandomForestClassifier(
    max_depth=5, max_features='sqrt', min_samples_leaf=2, min_samples_split=5, n_estimators=400
)


# Train the model
rf_model.fit(X_train, y_train)
print("Model training complete")


# Predict on the test set
test_predictions = rf_model.predict(X_test)

# Check the first few predictions
print("\nFirst 10 Test Predictions:")
print(test_predictions[:10])





# Create the submission DataFrame
# Competition requires 'id' and the prediction column
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': test_predictions 
})

# Save the submission file to the working directory
# This file is what Kaggle commits and uses for scoring
submission_df.to_csv('submission2.csv', index=False)

print("\nSubmission file 'submission2.csv' created successfully in /kaggle/working/")




