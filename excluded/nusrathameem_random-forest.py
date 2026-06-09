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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_log_error
import os



train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv') # Good to check format


print("Train Data Info:")
train_df.info()
print("\nTest Data Info:")
test_df.info()


print("\nMissing Values in Train Data:")
print(train_df.isnull().sum())
print("\nMissing Values in Test Data:")
print(test_df.isnull().sum())



# Keep the original test IDs safe!
test_ids = test_df['id']


# Drop 'id' from training data features as it's just an identifier
train_df = train_df.drop('id', axis=1)
# Keep 'id' in test_df for now, we'll drop it before preprocessing features


#Basic EDA
print("\nTrain Data Description:")
print(train_df.describe())


# Distribution of Target Variable
sns.histplot(train_df['Calories'], kde=True)
plt.title('Distribution of Calories')
plt.show()


 #Preprocessing

# Identify Categorical and Numerical Columns (excluding target and ID)
categorical_cols = ['Sex']
numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'] # Explicitly list them


# One-Hot Encoding for 'Sex'
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)



# Apply encoder to train data
encoded_train = encoder.fit_transform(train_df[categorical_cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols), index=train_df.index) # Preserve index


 #Apply encoder to test data
encoded_test = encoder.transform(test_df[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols), index=test_df.index) # Preserve index



# Combine numerical and encoded categorical features
# Drop original categorical columns and concatenate encoded ones
train_processed = pd.concat([train_df[numerical_cols], encoded_train_df], axis=1)
# Important: Drop 'id' and 'Sex' from test_df *before* concatenating
test_processed = pd.concat([test_df[numerical_cols], encoded_test_df], axis=1)



# Ensure columns are in the same order (usually handled by concat, but good practice)
test_processed = test_processed[train_processed.columns]


print("\nProcessed Training Features Head:")
print(train_processed.head())
print("\nProcessed Test Features Head:")
print(test_processed.head())


#  Prepare Data for Modeling ---
X = train_processed
y = train_df['Calories']
X_test = test_processed # Features for final prediction


# Split Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=25,         # Very few trees
    random_state=42,
    n_jobs=-1,               # Still parallel, but low load
    max_depth=5,             # Very shallow trees
    min_samples_split=20,    # Fewer splits
    min_samples_leaf=10      # Larger leaves
)

print("\nTraining Model...")
model.fit(X_train, y_train)
print("Training Complete.")



# Validation ---
y_val_pred = model.predict(X_val)


# IMPORTANT: Handle negative predictions BEFORE calculating RMSLE
y_val_pred = np.maximum(0, y_val_pred) # Set negative predictions to 0


# Calculate RMSLE
# Need to ensure y_val doesn't have negatives either (shouldn't if it's original data)
y_val_non_negative = np.maximum(0, y_val) # Just in case, though calories shouldn't be negative
rmsle = np.sqrt(mean_squared_log_error(y_val_non_negative, y_val_pred))
print(f"RMSLE on Validation Set: {rmsle}")



#  Prediction on Test Set ---
print("\nPredicting on Test Set...")
test_predictions = model.predict(X_test)

# IMPORTANT: Handle negative predictions
test_predictions = np.maximum(0, test_predictions)

# If you log-transformed y, transform predictions back:
# test_predictions = np.expm1(test_predictions)

print("Prediction Complete.")



# Create Submission File ---
# Use the ORIGINAL test_ids saved earlier
submission_df = pd.DataFrame({'id': test_ids, 'Calories': test_predictions})

print("\nSubmission File Head:")
print(submission_df.head())



# Save the submission file
submission_df.to_csv('submission.csv', index=False)


print("\nSubmission file created: submission.csv")
print(f"Number of rows in submission: {len(submission_df)}")
print(f"Number of rows expected in test: {len(test_ids)}")
# Check if number of rows match sample submission
print(f"Number of rows in sample submission: {len(sample_submission)}")

