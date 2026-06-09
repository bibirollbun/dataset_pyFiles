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


# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error




# Load the main training dataset
train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')

# Load the test dataset
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# Load the data dictionary to understand column names
data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')



train_data.head()


# Handle missing values in categorical columns by replacing them with the mode (most frequent value)
categorical_columns = train_data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    train_data[column].fillna(train_data[column].mode()[0], inplace=True)

# Handle missing values in numerical columns by replacing them with the median
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    train_data[column].fillna(train_data[column].median(), inplace=True)

# Check if there are any missing values left
train_data.isnull().sum()



from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
label_encoder = LabelEncoder()

# Apply label encoding for binary columns (e.g., 'Yes'/'No' columns)
binary_columns = ['psych_disturb', 'arrhythmia', 'diabetes', 'hla_match_c_high', 'hla_high_res_8']
for column in binary_columns:
    train_data[column] = label_encoder.fit_transform(train_data[column].astype(str))

# For non-binary categorical columns, you can apply similar encoding
# If there are other categorical columns to encode, use the same process
categorical_columns = ['tbi_status', 'hla_match_dqb1_high', 'tce_imm_match', 'rituximab']  # add other columns if needed
for column in categorical_columns:
    train_data[column] = label_encoder.fit_transform(train_data[column].astype(str))

# Check the encoded data
train_data.head()



from sklearn.preprocessing import StandardScaler

# List of numerical columns
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns

# Initialize scaler
scaler = StandardScaler()

# Scale numerical columns
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])

# Check the scaled data
train_data.head()



from sklearn.model_selection import train_test_split

# Define features and target
X = train_data.drop(columns=['efs', 'efs_time', 'ID'])  # Drop target columns and ID
y = train_data['efs']  # Let's say 'efs' is the target column

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the splits
X_train.shape, X_test.shape, y_train.shape, y_test.shape



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')





# Apply one-hot encoding to categorical columns
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Make sure to align the train and test datasets
train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1, fill_value=0)

# Separate features (X) and target (y)
X_train_encoded = train_data_encoded.drop(columns=['efs', 'efs_time'])  # Drop target columns
y_train_encoded = train_data_encoded['efs']

X_test_encoded = test_data_encoded.drop(columns=['efs', 'efs_time'])  # Drop target columns
y_test_encoded = test_data_encoded['efs']



# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train_encoded, y_train_encoded)

# Predict on the test set
y_pred_encoded = model.predict(X_test_encoded)

# Evaluate the model
mse = mean_squared_error(y_test_encoded, y_pred_encoded)
r2 = r2_score(y_test_encoded, y_pred_encoded)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')



from sklearn.impute import SimpleImputer

# Impute missing values with the median for numerical columns
imputer = SimpleImputer(strategy='median')
X_train_encoded_imputed = imputer.fit_transform(X_train_encoded)
X_test_encoded_imputed = imputer.transform(X_test_encoded)

# Now proceed to train the model
model = LinearRegression()

# Train the model
model.fit(X_train_encoded_imputed, y_train_encoded)

# Predict on the test set
y_pred_encoded = model.predict(X_test_encoded_imputed)

# Evaluate the model
mse = mean_squared_error(y_test_encoded, y_pred_encoded)
r2 = r2_score(y_test_encoded, y_pred_encoded)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Standardize the features
X_train_encoded_scaled = scaler.fit_transform(X_train_encoded_imputed)
X_test_encoded_scaled = scaler.transform(X_test_encoded_imputed)



from sklearn.ensemble import RandomForestRegressor

# Initialize a Random Forest Regressor model
rf_model = RandomForestRegressor()

# Train the model
rf_model.fit(X_train_encoded_scaled, y_train_encoded)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test_encoded_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test_encoded, y_pred_rf)
r2_rf = r2_score(y_test_encoded, y_pred_rf)

print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest R-squared: {r2_rf}')



# 1. Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_encoded_scaled, y_train_encoded)

# 2. Evaluate model performance (as you have done earlier)
y_pred_encoded = rf_model.predict(X_test_encoded_scaled)
mse = mean_squared_error(y_test_encoded, y_pred_encoded)
r2 = r2_score(y_test_encoded, y_pred_encoded)
print(f"Random Forest Mean Squared Error: {mse}")
print(f"Random Forest R-squared: {r2}")

# 3. Feature Importance
importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]

# Show the top 10 important features
features = X_train_encoded.columns
pd.DataFrame({'Feature': features[indices], 'Importance': importances[indices]}).head(10)

# 4. Hyperparameter tuning (optional)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_encoded_scaled, y_train_encoded)

print("Best parameters found:", grid_search.best_params_)



import numpy as np

print("Variance of y_train_encoded:", np.var(y_train_encoded))
print("Unique values in y_train_encoded:", np.unique(y_train_encoded))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_encoded_scaled, y_train_encoded)

# Predict on the test set
y_pred_encoded = rf_model.predict(X_test_encoded_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Random Forest Accuracy: {accuracy}")

# Detailed classification report
print(classification_report(y_test_encoded, y_pred_encoded))



import numpy as np

print("Class distribution in training set:")
unique, counts = np.unique(y_train_encoded, return_counts=True)
print(dict(zip(unique, counts)))

print("\nClass distribution in test set:")
unique, counts = np.unique(y_test_encoded, return_counts=True)
print(dict(zip(unique, counts)))



from sklearn.model_selection import train_test_split

# Re-split with stratification
X_train_encoded_scaled, X_test_encoded_scaled, y_train_encoded, y_test_encoded = train_test_split(
    X_encoded_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Verify the new distribution
import numpy as np
print("New Class Distribution in Training Set:", dict(zip(*np.unique(y_train_encoded, return_counts=True))))
print("New Class Distribution in Test Set:", dict(zip(*np.unique(y_test_encoded, return_counts=True))))



print("X_encoded_scaled exists:", 'X_encoded_scaled' in locals())
print("y_encoded exists:", 'y_encoded' in locals())



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)  # Convert categorical columns to numerical



scaler = StandardScaler(with_mean=False)  # ✅ Prevent centering sparse data
X_encoded_scaled = scaler.fit_transform(X_encoded)



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Convert labels to numeric values



X_train_encoded_scaled, X_test_encoded_scaled, y_train_encoded, y_test_encoded = train_test_split(
    X_encoded_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)



from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train_encoded_scaled, y_train_encoded)



from sklearn.metrics import classification_report, accuracy_score

# Predict on the test set
y_pred_encoded = rf_model.predict(X_test_encoded_scaled)

# Evaluate the model performance
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Random Forest Accuracy: {accuracy}")

# Display classification report for more detailed metrics
print(classification_report(y_test_encoded, y_pred_encoded))



import numpy as np
import pandas as pd

# Get feature importances
importances = rf_model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Create a DataFrame to display the top 10 important features based on index positions
feature_importance_df = pd.DataFrame({
    'Feature Index': indices[:10], 
    'Importance': importances[indices[:10]]
})

print(feature_importance_df)



from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, X_train_encoded_scaled, y_train_encoded, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {scores}")
print(f"Average Cross-Validation Accuracy: {scores.mean()}")


